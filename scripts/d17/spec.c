/* D17 part B: how much of a union-loading sweep's reads a speculative read
 * could cover, from a DS4_ROCM_ROUTE_TRACE file (format and lane handling as
 * in D16's scripts/d16b/union.c, whose parsing this follows).
 *
 * Per chunk lane and layer:
 *   U    = union over rows of the unbiased top-6 of s + bias (today's routing)
 *   need = U minus experts resident at the chunk's start (what must be read)
 * Speculative candidates, each read before the layer's router has run (so
 * they can overlap the previous layer's compute), minus residents:
 *   prev    the same layer's union in the previous chunk
 *   hotK    the K experts this layer was routed to most often in all earlier
 *           chunks (K = 32, 64, 128)
 *   prev+hot64
 * For each: covered = |spec & need| / |need| (reads that become overlapped),
 * wasted = |spec - need| (reads not needed). Reported per chunk-size bin
 * (warm chunks only; a cold chunk has no history and reads whole layers).
 * Usage: spec TRACE */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NL 40
#define NE 384
#define NC 5   /* candidates */
static const char *CN[NC] = {"prev", "hot32", "hot64", "hot128", "prev+hot64"};

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) x = s;
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}
static int better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }
static void top6(const float *c, int out[6]) {
    float v[6];
    int n = 0;
    for (int e = 0; e < NE; e++) {
        if (n < 6 || better(c[e], e, v[5], out[5])) {
            int i = n < 6 ? n++ : 5;
            while (i > 0 && better(c[e], e, v[i - 1], out[i - 1])) { v[i] = v[i - 1]; out[i] = out[i - 1]; i--; }
            v[i] = c[e];
            out[i] = e;
        }
    }
}
static void topk(const double *freq, int k, unsigned char *out) {
    memset(out, 0, NE);
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < NE; e++)
            if (!out[e] && freq[e] > 0 && (best < 0 || freq[e] > freq[best])) best = e;
        if (best < 0) break;
        out[best] = 1;
    }
}

typedef struct { double layers, rows, uni, need, cov[NC], waste[NC]; } bin_t;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: spec TRACE\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    static float bias[NL][NE];
    enum { MAXLANE = 16 };
    static unsigned char gmask[MAXLANE][NL * NE];
    uint32_t grows[MAXLANE];
    uint64_t gres[MAXLANE];
    int glanes = 0, gseen[NL];
    int last_l_il = 1, have = 0;
    /* history, updated when a group of lanes is finished (so a chunk never sees itself) */
    static double freq[NL][NE], gfreq[NL][NE];
    static unsigned char prev[NL][NE], gprev[NL][NE];
    static int have_prev[NL], g_have_prev[NL];
    static bin_t bins[4];
    const char *bn[4] = {"warm, <=512 rows", "warm, 513-1023", "warm, 1024-2048", "warm, 2049+"};
    float *s = NULL; uint16_t *h = NULL; size_t cap = 0;
    float c[NE];
    for (;;) {
        int tag = fgetc(f);
        if (tag == EOF) break;
        uint32_t hdr[2];
        if (fread(hdr, sizeof(hdr), 1, f) != 1) break;
        if (tag == 'B') {
            if (hdr[0] >= NL || hdr[1] != NE || fread(bias[hdr[0]], 4, NE, f) != NE) break;
        } else if (tag == 'C') {
            uint64_t nres;
            static unsigned char tmp[NL * NE];
            if (fread(&nres, 8, 1, f) != 1 || fread(tmp, 1, NL * NE, f) != NL * NE) break;
            if (last_l_il != 0) {   /* a new group: fold the finished group's routing into history */
                for (int l = 0; l < NL; l++) {
                    for (int e = 0; e < NE; e++) freq[l][e] += gfreq[l][e];
                    if (g_have_prev[l]) { memcpy(prev[l], gprev[l], NE); have_prev[l] = 1; }
                }
                memset(gfreq, 0, sizeof(gfreq));
                memset(gprev, 0, sizeof(gprev));
                memset(g_have_prev, 0, sizeof(g_have_prev));
                glanes = 0;
                memset(gseen, 0, sizeof(gseen));
            }
            memcpy(gmask[glanes], tmp, NL * NE);
            grows[glanes] = hdr[1];
            gres[glanes] = nres;
            glanes++;
            have = 1;
            last_l_il = 0;
        } else if (tag == 'L') {
            const uint32_t il = hdr[0], rows = hdr[1];
            const size_t n = (size_t)rows * NE;
            if (n > cap) { cap = n; h = realloc(h, n * 2); s = realloc(s, n * 4); }
            if (fread(h, 2, n, f) != n) break;
            last_l_il = (int)il;
            if (!have || il >= NL) continue;
            const int lane = il == 0 ? glanes - 1 : gseen[il]++;
            if (lane < 0 || lane >= glanes || rows != grows[lane]) continue;
            const unsigned char *res = gmask[lane] + (size_t)il * NE;
            unsigned char uni[NE] = {0};
            for (uint32_t r = 0; r < rows; r++) {
                for (int e = 0; e < NE; e++) c[e] = h2f(h[(size_t)r * NE + e]) + bias[il][e];
                int ex[6];
                top6(c, ex);
                for (int k = 0; k < 6; k++) { uni[ex[k]] = 1; gfreq[il][ex[k]] += 1; }
            }
            for (int e = 0; e < NE; e++) gprev[il][e] |= uni[e];
            g_have_prev[il] = 1;
            if (gres[lane] == 0) continue;   /* cold chunk: no residents, whole-layer path */
            const int b = rows <= 512 ? 0 : rows < 1024 ? 1 : rows <= 2048 ? 2 : 3;
            unsigned char cand[NC][NE];
            memset(cand, 0, sizeof(cand));
            if (have_prev[il]) memcpy(cand[0], prev[il], NE);
            topk(freq[il], 32, cand[1]);
            topk(freq[il], 64, cand[2]);
            topk(freq[il], 128, cand[3]);
            for (int e = 0; e < NE; e++) cand[4][e] = cand[0][e] | cand[2][e];
            double nu = 0, nn = 0, cov[NC] = {0}, wst[NC] = {0};
            for (int e = 0; e < NE; e++) {
                const int need = uni[e] && !res[e];
                nu += uni[e];
                nn += need;
                for (int k = 0; k < NC; k++) {
                    if (!cand[k][e] || res[e]) continue;
                    if (need) cov[k] += 1; else wst[k] += 1;
                }
            }
            bins[b].layers += 1;
            bins[b].rows += rows;
            bins[b].uni += nu;
            bins[b].need += nn;
            for (int k = 0; k < NC; k++) { bins[b].cov[k] += cov[k]; bins[b].waste[k] += wst[k]; }
        }
    }
    printf("| chunk bin | layer-chunks | mean rows | union / 384 | need (not resident) |");
    for (int k = 0; k < NC; k++) printf(" %s covered / wasted |", CN[k]);
    printf("\n|---|---|---|---|---|");
    for (int k = 0; k < NC; k++) printf("---|");
    printf("\n");
    for (int b = 0; b < 4; b++) {
        if (!bins[b].layers) continue;
        const double L = bins[b].layers;
        printf("| %s | %.0f | %.0f | %.0f | %.0f |", bn[b], L, bins[b].rows / L, bins[b].uni / L, bins[b].need / L);
        for (int k = 0; k < NC; k++)
            printf(" %.0f%% / %.0f |", 100.0 * bins[b].cov[k] / (bins[b].need > 0 ? bins[b].need : 1), bins[b].waste[k] / L);
        printf("\n");
    }
    printf("(per layer; one MXFP4 expert = 17.93 MiB; 40 layers per chunk)\n");
    return 0;
}
