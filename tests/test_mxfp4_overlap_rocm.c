/* Model-free bit-equality check of the D2 overlap launch order.
 *
 * D2 runs the routed MXFP4 experts' gate/up in pieces (the resident (row,
 * expert) pairs first, then the pairs of each missing expert as its gate and
 * up land) through a pair-list twin of the decode gate/up kernel, then the
 * unchanged mid quantize and fused down in route order.  This test drives the
 * launcher's own helpers (ds4_gpu_test_mxfp4_overlap in
 * rocm/ds4_rocm_moe_launch.cuh) with V4.1 Flash expert shapes, random MXFP4
 * weights, activations and routes, and compares on identical inputs with
 *   - one row: the sequential single-token zero-copy decode launches
 *     (plain decode's reference arithmetic);
 *   - 2..6 rows: D1's verify-rows launch, in both down modes.
 * For every row count, several random partitions of the pairs into launch
 * groups (1 group = everything at once, up to one group per pair, random
 * order) are checked.  mid and out must be bitwise equal (memcmp); outputs
 * are poisoned differently beforehand so an unwritten element differs. */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_gpu_test_mxfp4_overlap(
        const void *gate_host, const void *up_host, const void *down_host,
        uint32_t n_union, const float *x_host, const int32_t *selected_host,
        const float *weights_host, uint32_t n_tokens, uint32_t n_expert,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        float clamp, int down_per_row, const uint8_t *group, uint32_t n_groups,
        float *mid_d2_host, float *mid_ref_host, float *out_d2_host, float *out_ref_host);

#define IN_DIM 5120u
#define MID_DIM 2304u
#define OUT_DIM 5120u
#define N_EXPERT 6u
#define MAX_ROWS 6u
#define N_UNION 24u
#define CLAMP 10.0f
#define PARTITIONS 4u

static uint32_t g_rng = 0x9e3779b9u;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13u;
    g_rng ^= g_rng >> 17u;
    g_rng ^= g_rng << 5u;
    return g_rng;
}
static float rndf(void) { return (float)(rnd() & 0xffffffu) / 8388608.0f - 1.0f; }

static void fill_mxfp4(uint8_t *p, uint64_t bytes) {
    for (uint64_t i = 0; i < bytes; i++) {
        p[i] = (i % 17u) == 0u ? (uint8_t)(120u + rnd() % 8u) : (uint8_t)rnd();
    }
}

int main(void) {
    const uint64_t gate_eb = (uint64_t)IN_DIM / 32u * 17u * MID_DIM;
    const uint64_t down_eb = (uint64_t)MID_DIM / 32u * 17u * OUT_DIM;
    uint8_t *gate = malloc(gate_eb * N_UNION);
    uint8_t *up = malloc(gate_eb * N_UNION);
    uint8_t *down = malloc(down_eb * N_UNION);
    float *x = malloc(sizeof(float) * MAX_ROWS * IN_DIM);
    float *w = malloc(sizeof(float) * MAX_ROWS * N_EXPERT);
    int32_t *sel = malloc(sizeof(int32_t) * MAX_ROWS * N_EXPERT);
    float *mid_a = malloc(sizeof(float) * MAX_ROWS * N_EXPERT * MID_DIM);
    float *mid_b = malloc(sizeof(float) * MAX_ROWS * N_EXPERT * MID_DIM);
    float *out_a = malloc(sizeof(float) * MAX_ROWS * OUT_DIM);
    float *out_b = malloc(sizeof(float) * MAX_ROWS * OUT_DIM);
    if (!gate || !up || !down || !x || !w || !sel || !mid_a || !mid_b || !out_a || !out_b) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    fill_mxfp4(gate, gate_eb * N_UNION);
    fill_mxfp4(up, gate_eb * N_UNION);
    fill_mxfp4(down, down_eb * N_UNION);
    for (uint32_t i = 0; i < MAX_ROWS * IN_DIM; i++) x[i] = rndf() * 4.0f;
    for (uint32_t t = 0; t < MAX_ROWS; t++) {
        float s = 0.0f;
        for (uint32_t j = 0; j < N_EXPERT; j++) {
            int32_t e;
            int dup;
            do {
                e = (int32_t)(rnd() % N_UNION);
                dup = 0;
                for (uint32_t k = 0; k < j; k++) dup |= sel[t * N_EXPERT + k] == e;
            } while (dup);
            sel[t * N_EXPERT + j] = e;
            w[t * N_EXPERT + j] = 0.05f + (float)(rnd() % 1000u) / 1000.0f;
            s += w[t * N_EXPERT + j];
        }
        for (uint32_t j = 0; j < N_EXPERT; j++) w[t * N_EXPERT + j] /= s;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }
    int failures = 0, cases = 0;
    for (int mode = 0; mode < 2; mode++) {
        for (uint32_t n = 1; n <= MAX_ROWS; n++) {
            if (n == 1u && mode == 1) continue; /* one row: decode tail only */
            const uint32_t pairs = n * N_EXPERT;
            for (uint32_t part = 0; part < PARTITIONS; part++) {
                uint8_t group[MAX_ROWS * N_EXPERT];
                uint32_t n_groups;
                if (part == 0) {
                    n_groups = 1;
                    memset(group, 0, sizeof(group));
                } else if (part == 1) {
                    /* one launch per pair, in reverse pair order */
                    n_groups = pairs;
                    for (uint32_t p = 0; p < pairs; p++) group[p] = (uint8_t)(pairs - 1u - p);
                } else {
                    /* "resident" group 0, then the "missing" union slots one
                     * group each, in random order: like D2's launches. */
                    uint8_t slot_group[N_UNION];
                    n_groups = 1;
                    for (uint32_t u = 0; u < N_UNION; u++) {
                        slot_group[u] = (rnd() % 3u) == 0u ? (uint8_t)(1u + rnd() % 5u) : 0u;
                        if (slot_group[u] + 1u > n_groups) n_groups = slot_group[u] + 1u;
                    }
                    for (uint32_t p = 0; p < pairs; p++) group[p] = slot_group[sel[p]];
                }
                cases++;
                if (!ds4_gpu_test_mxfp4_overlap(gate, up, down, N_UNION, x, sel, w,
                                                n, N_EXPERT, IN_DIM, MID_DIM, OUT_DIM,
                                                CLAMP, mode, group, n_groups,
                                                mid_a, mid_b, out_a, out_b)) {
                    fprintf(stderr, "launch failed (mode=%d rows=%u part=%u)\n", mode, n, part);
                    failures++;
                    continue;
                }
                const size_t mid_n = (size_t)pairs * MID_DIM;
                const int mid_equal = memcmp(mid_a, mid_b, mid_n * sizeof(float)) == 0;
                const int out_equal = memcmp(out_a, out_b, (size_t)n * OUT_DIM * sizeof(float)) == 0;
                int finite = 1;
                for (size_t i = 0; i < (size_t)n * OUT_DIM; i++) {
                    if (!isfinite(out_a[i]) || !isfinite(out_b[i])) finite = 0;
                }
                printf("ref=%s rows=%u groups=%u mid=%s out=%s finite=%d\n",
                       n == 1u ? "decode" : (mode ? "d1-per-row" : "d1-batched"),
                       n, n_groups, mid_equal ? "bitwise" : "DIFFERS",
                       out_equal ? "bitwise" : "DIFFERS", finite);
                if (!mid_equal || !out_equal || !finite) failures++;
            }
        }
    }
    ds4_gpu_cleanup();
    printf("%s (%d of %d cases failing)\n", failures ? "FAIL" : "PASS", failures, cases);
    return failures ? 1 : 0;
}
