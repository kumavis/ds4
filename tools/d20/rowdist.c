/* D20: rows per expert in real prefill chunks, from a D16b routing trace
 * (DS4_ROCM_ROUTE_TRACE).  Unbiased top-6 per row (c = s + bias, ties to the
 * lower index), then per layer the count of rows routed to each of the 384
 * experts.  Prints, per chunk-size bin: experts used per layer, rows per used
 * expert (mean, p50, p90, max), and the share of used experts with fewer
 * than 8 / 16 / 32 rows, and the share of pairs in experts with < 16 rows.
 * Usage: rowdist TRACE */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define NE 384
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) x = s; else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static int cmpu(const void *a, const void *b) { return (int)(*(const uint32_t *)a) - (int)(*(const uint32_t *)b); }
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    static float bias[NL][NE];
    const char *bn[4] = {"256-511 rows", "512-1023 rows", "1024-2047 rows", "2048 rows"};
    double used[4] = {0}, layers[4] = {0}, lt8[4] = {0}, lt16[4] = {0}, lt32[4] = {0}, pairs_lt16[4] = {0}, pairs[4] = {0};
    static uint32_t all[4][NL * NE * 64];
    size_t nall[4] = {0};
    uint16_t *h = NULL; size_t cap = 0;
    for (;;) {
        int tag = fgetc(f);
        if (tag == EOF) break;
        uint32_t hdr[2];
        if (fread(hdr, 8, 1, f) != 1) break;
        if (tag == 'B') { if (fread(bias[hdr[0]], 4, NE, f) != NE) break; continue; }
        if (tag == 'C') { uint64_t n; static unsigned char m[NL * NE]; if (fread(&n, 8, 1, f) != 1 || fread(m, 1, NL * NE, f) != NL * NE) break; continue; }
        if (tag != 'L') return 1;
        const uint32_t il = hdr[0], rows = hdr[1];
        const size_t n = (size_t)rows * NE;
        if (n > cap) { cap = n; h = realloc(h, n * 2); }
        if (fread(h, 2, n, f) != n) break;
        if (rows < 256) continue;
        const int b = rows < 512 ? 0 : rows < 1024 ? 1 : rows < 2048 ? 2 : 3;
        uint32_t cnt[NE] = {0};
        for (uint32_t r = 0; r < rows; r++) {
            float c[NE];
            for (int e = 0; e < NE; e++) c[e] = h2f(h[(size_t)r * NE + e]) + bias[il][e];
            for (int k = 0; k < 6; k++) {
                int best = 0;
                for (int e = 1; e < NE; e++) if (c[e] > c[best]) best = e;
                cnt[best]++;
                c[best] = -INFINITY;
            }
        }
        layers[b]++;
        for (int e = 0; e < NE; e++) {
            if (!cnt[e]) continue;
            used[b]++;
            pairs[b] += cnt[e];
            if (cnt[e] < 8) lt8[b]++;
            if (cnt[e] < 16) { lt16[b]++; pairs_lt16[b] += cnt[e]; }
            if (cnt[e] < 32) lt32[b]++;
            if (nall[b] < sizeof(all[0]) / sizeof(all[0][0])) all[b][nall[b]++] = cnt[e];
        }
    }
    printf("| chunk rows | layer calls | experts used / layer | rows per used expert: mean / p50 / p90 / max | used experts < 8 rows | < 16 | < 32 | pairs in experts < 16 rows |\n");
    printf("|---|---:|---:|---|---:|---:|---:|---:|\n");
    for (int b = 0; b < 4; b++) {
        if (!layers[b]) continue;
        qsort(all[b], nall[b], sizeof(uint32_t), cmpu);
        printf("| %s | %.0f | %.0f | %.1f / %u / %u / %u | %.1f%% | %.1f%% | %.1f%% | %.1f%% |\n", bn[b], layers[b],
               used[b] / layers[b], pairs[b] / used[b], all[b][nall[b] / 2], all[b][(size_t)(nall[b] * 0.9)],
               all[b][nall[b] - 1], 100 * lt8[b] / used[b], 100 * lt16[b] / used[b], 100 * lt32[b] / used[b],
               100 * pairs_lt16[b] / pairs[b]);
    }
    return 0;
}
