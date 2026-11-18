/* D16b stage 1: offline prefill-union model from a DS4_ROCM_ROUTE_TRACE file.
 *
 * For every traced prefill chunk (the 'C' record: residency at the chunk's
 * start, then 'L' records with every row's gating scores per layer) and every
 * setting (unbiased; Cache-Prior lambda 0.05 / 0.09 / 0.2 with keep 3 and 2):
 *   - each row's top-6 of c = s + bias, with +lambda*R for experts resident at
 *     the chunk's start or in the row's unbiased top-keep (ties: lower index);
 *     R = per-layer mean range (max c - min c) of all rows traced before this
 *     chunk (frozen for the chunk; a chunk with no history uses each row's own
 *     range), as the recipe's prefill transaction;
 *   - the layer's union over rows; "reads" = union experts not resident at the
 *     chunk's start (what the sweep must read; in-chunk evictions ignored);
 *   - swapped row-layers (biased set != unbiased set) and drift (weight share
 *     of the dropped unbiased picks).
 * Output: per chunk-size bin, mean reads per chunk, GiB, modelled read time at
 * READ_GIBS, change against unbiased, swap share and drift; plus per-chunk
 * lines with -v.
 * Usage: union TRACE [READ_GIBS=7.5] [-v] */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NL 40
#define NE 384
#define NS 7
static const float LAM[NS] = {0.0f, 0.05f, 0.09f, 0.2f, 0.05f, 0.09f, 0.2f};
static const int KEEP[NS] = {0, 3, 3, 3, 2, 2, 2};
static const double EXPERT_BYTES = 18800640.0; /* one MXFP4 expert (gate+up+down) */

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) x = s; /* the writer flushes subnormals to zero */
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
            while (i > 0 && better(c[e], e, v[i - 1], out[i - 1])) {
                v[i] = v[i - 1];
                out[i] = out[i - 1];
                i--;
            }
            v[i] = c[e];
            out[i] = e;
        }
    }
}

typedef struct {
    double chunks, rows, reads[NS], union_sz[NS], rl, swapped[NS], drift[NS];
} bin_t;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: union TRACE [READ_GIBS] [-v]\n");
        return 2;
    }
    double gibs = 7.5;
    int verbose = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) verbose = 1;
        else gibs = atof(argv[i]);
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("open");
        return 1;
    }
    static float bias[NL][NE];
    /* A multi-chunk cohort writes its lanes' 'C' records first (layer 0 of
     * each lane), then layer 1 of every lane, and so on: lane k of the
     * group gets the k-th record of each layer. */
    enum { MAXLANE = 16 };
    static unsigned char gmask[MAXLANE][NL * NE];
    uint32_t gpos[MAXLANE], grows[MAXLANE];
    uint64_t gres[MAXLANE];
    int glanes = 0, gseen[NL];
    double gread[MAXLANE][NS];
    int last_l_il = 1; /* layer of the last 'L' record; a 'C' after a layer >= 1 starts a group */
    double rsum[NL] = {0}, rcnt[NL] = {0};
    double rbar[NL];
    int have_chunk = 0, chunk_id = -1;
    static bin_t bins[5];
    const char *bin_name[5] = {"warm, 2-512 rows", "warm, 513-1023 rows", "warm, 1024-2048 rows", "warm, 2049+ rows", "cold (nothing resident at the chunk start)"};
    float *c = NULL, *s = NULL;
    uint16_t *h = NULL;
    size_t cap = 0;
    long nrec = 0;
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
            if (last_l_il != 0) {
                /* a new group: report the finished one */
                for (int g = 0; g < glanes && verbose; g++) {
                    printf("chunk %d pos %u rows %u resident %llu lanes %d reads", chunk_id - glanes + 1 + g,
                           gpos[g], grows[g], (unsigned long long)gres[g], glanes);
                    for (int k = 0; k < NS; k++) printf(" %.0f", gread[g][k]);
                    printf("\n");
                }
                glanes = 0;
                memset(gseen, 0, sizeof(gseen));
                for (int l = 0; l < NL; l++) rbar[l] = rcnt[l] > 0 ? rsum[l] / rcnt[l] : 0.0;
            }
            if (glanes >= MAXLANE) { fprintf(stderr, "too many lanes\n"); return 1; }
            memcpy(gmask[glanes], tmp, NL * NE);
            gpos[glanes] = hdr[0];
            grows[glanes] = hdr[1];
            gres[glanes] = nres;
            memset(gread[glanes], 0, sizeof(gread[0]));
            glanes++;
            have_chunk = 1;
            chunk_id++;
            last_l_il = 0;
            const int b = nres == 0 ? 4 : hdr[1] <= 512 ? 0 : hdr[1] < 1024 ? 1 : hdr[1] <= 2048 ? 2 : 3;
            bins[b].chunks++;
            bins[b].rows += hdr[1];
            continue;
        } else if (tag == 'L') {
            const uint32_t il = hdr[0], rows = hdr[1];
            const size_t n = (size_t)rows * NE;
            if (n > cap) {
                cap = n;
                h = realloc(h, n * 2);
                s = realloc(s, n * 4);
                c = realloc(c, NE * 4);
            }
            if (fread(h, 2, n, f) != n) break;
            nrec++;
            last_l_il = (int)il;
            if (!have_chunk || il >= NL) continue;
            const int lane = il == 0 ? glanes - 1 : gseen[il]++;
            if (lane < 0 || lane >= glanes || rows != grows[lane]) {
                fprintf(stderr, "lane mismatch at layer %u rows %u\n", il, rows);
                continue;
            }
            const unsigned char *mask = gmask[lane];
            const uint32_t crows = grows[lane];
            const int b = gres[lane] == 0 ? 4 : crows <= 512 ? 0 : crows < 1024 ? 1 : crows <= 2048 ? 2 : 3;
            for (size_t i = 0; i < n; i++) s[i] = h2f(h[i]);
            unsigned char uni[NS][NE];
            memset(uni, 0, sizeof(uni));
            const unsigned char *lm = mask + (size_t)il * NE;
            double lr_sum = 0;
            for (uint32_t r = 0; r < rows; r++) {
                const float *sr = s + (size_t)r * NE;
                float mx = -INFINITY, mn = INFINITY;
                for (int e = 0; e < NE; e++) {
                    c[e] = sr[e] + bias[il][e];
                    if (c[e] > mx) mx = c[e];
                    if (c[e] < mn) mn = c[e];
                }
                const float robs = mx - mn;
                lr_sum += robs;
                const float R = rbar[il] > 0 ? (float)rbar[il] : robs;
                int ex[6];
                top6(c, ex);
                for (int k = 0; k < 6; k++) uni[0][ex[k]] = 1;
                float es = 0;
                for (int k = 0; k < 6; k++) es += sr[ex[k]];
                for (int k = 1; k < NS; k++) {
                    float cb[NE];
                    const float bonus = LAM[k] * R;
                    for (int e = 0; e < NE; e++) cb[e] = c[e] + (lm[e] ? bonus : 0.0f);
                    for (int j = 0; j < KEEP[k]; j++) if (!lm[ex[j]]) cb[ex[j]] += bonus;
                    int pk[6];
                    top6(cb, pk);
                    float drop = 0;
                    int sw = 0;
                    for (int j = 0; j < 6; j++) {
                        uni[k][pk[j]] = 1;
                        int kept = 0;
                        for (int q = 0; q < 6; q++) kept |= pk[q] == ex[j];
                        if (!kept) { sw = 1; drop += sr[ex[j]]; }
                    }
                    bins[b].swapped[k] += sw;
                    bins[b].drift[k] += es > 0 ? drop / es : 0;
                }
            }
            bins[b].rl += rows;
            for (int k = 0; k < NS; k++) {
                int u = 0, rd = 0;
                for (int e = 0; e < NE; e++) if (uni[k][e]) { u++; if (!lm[e]) rd++; }
                bins[b].reads[k] += rd;
                bins[b].union_sz[k] += u;
                gread[lane][k] += rd;
            }
            /* this chunk's rows commit their ranges after the chunk */
            rsum[il] += lr_sum;
            rcnt[il] += rows;
        } else {
            fprintf(stderr, "bad tag %d after %ld layer records\n", tag, nrec);
            return 1;
        }
    }
    for (int g = 0; g < glanes && verbose; g++) {
        printf("chunk %d pos %u rows %u resident %llu lanes %d reads", chunk_id - glanes + 1 + g,
               gpos[g], grows[g], (unsigned long long)gres[g], glanes);
        for (int k = 0; k < NS; k++) printf(" %.0f", gread[g][k]);
        printf("\n");
    }
    printf("layer records %ld, chunks %d, read rate %.2f GiB/s (modelled time = reads x %.2f MiB / rate)\n",
           nrec, chunk_id + 1, gibs, EXPERT_BYTES / 1048576.0);
    for (int b = 0; b < 5; b++) {
        bin_t *x = &bins[b];
        if (!x->chunks) continue;
        printf("\n%s: %.0f chunks, mean %.0f rows; per chunk (all 40 layers):\n", bin_name[b], x->chunks,
               x->rows / x->chunks);
        printf("| setting | union experts | experts read | GiB read | modelled read s | reads vs unbiased | swapped row-layers | drift |\n");
        printf("|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for (int k = 0; k < NS; k++) {
            const double rd = x->reads[k] / x->chunks, gib = rd * EXPERT_BYTES / 1073741824.0;
            char name[48];
            if (k == 0) snprintf(name, sizeof(name), "unbiased");
            else snprintf(name, sizeof(name), "lambda %.2f keep %d", LAM[k], KEEP[k]);
            char sw[32] = "-", dr[32] = "-";
            if (k) {
                snprintf(sw, sizeof(sw), "%.4f", x->swapped[k] / x->rl);
                snprintf(dr, sizeof(dr), "%.4f", x->drift[k] / x->rl);
            }
            printf("| %s | %.0f | %.0f | %.2f | %.2f | %+.1f%% | %s | %s |\n", name,
                   x->union_sz[k] / x->chunks, rd, gib, gib / gibs,
                   k ? 100.0 * (x->reads[k] - x->reads[0]) / (x->reads[0] > 0 ? x->reads[0] : 1) : 0.0,
                   sw, dr);
        }
    }
    free(c); free(s); free(h);
    fclose(f);
    return 0;
}
