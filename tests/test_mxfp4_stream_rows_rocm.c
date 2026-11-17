/* Model-free check of the SSD-streamed MXFP4 verify rows (D1).
 *
 * routed_moe_launch sends DSpark verification rows (2..6 tokens) on streamed
 * MXFP4 experts through routed_moe_mxfp4_ptrs_rows_launch, which reads the
 * resident slots through a union pointer table.  This test drives that helper
 * and, on identical inputs, the one-token decode launches that plain decode
 * runs for each row (ds4_gpu_test_mxfp4_stream_rows in
 * rocm/ds4_rocm_moe_launch.cuh), with the V4.1 Flash expert shapes and
 * random weights and routes.
 *
 * Checks, per row count 1..6 and per down mode (one batched launch, or one
 * decode-instantiation launch per row):
 *   - mid (gate/up/SwiGLU, per (row, expert) pair) is bitwise equal;
 *   - out is finite everywhere (outputs are poisoned beforehand);
 *   - out is bitwise equal, or else within a tolerance that only floating-
 *     point reordering could explain: max |diff| <= 1e-5 * max |ref| of the
 *     row.  The measured max abs difference per row is printed either way.
 * A wrong expert, row or stale slot produces O(1) relative differences and
 * fails the tolerance.  The per-row mode must be bitwise equal (it is the
 * decode instantiation). */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_gpu_test_mxfp4_stream_rows(
        const void *gate_host, const void *up_host, const void *down_host,
        uint32_t n_union, const float *x_host, const int32_t *selected_host,
        const float *weights_host, uint32_t n_tokens, uint32_t n_expert,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        float clamp, int down_per_row, float *mid_rows_host, float *mid_ref_host,
        float *out_rows_host, float *out_ref_host);

#define IN_DIM 5120u
#define MID_DIM 2304u
#define OUT_DIM 5120u
#define N_EXPERT 6u
#define MAX_ROWS 6u
#define N_UNION 24u
#define CLAMP 10.0f

static uint32_t g_rng = 0x12345678u;
static uint32_t rnd(void) {
    g_rng ^= g_rng << 13u;
    g_rng ^= g_rng >> 17u;
    g_rng ^= g_rng << 5u;
    return g_rng;
}
static float rndf(void) { return (float)(rnd() & 0xffffffu) / 8388608.0f - 1.0f; }

static void fill_mxfp4(uint8_t *p, uint64_t bytes) {
    /* 17-byte blocks: one E8M0 scale, 16 bytes of nibbles. */
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
            /* Six distinct union slots per row; rows overlap like real
             * verify rows do (shared experts across adjacent tokens). */
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
    int failures = 0;
    for (int mode = 0; mode < 2; mode++) {
        for (uint32_t n = 1; n <= MAX_ROWS; n++) {
            if (!ds4_gpu_test_mxfp4_stream_rows(gate, up, down, N_UNION, x, sel, w,
                                                n, N_EXPERT, IN_DIM, MID_DIM, OUT_DIM,
                                                CLAMP, mode, mid_a, mid_b, out_a, out_b)) {
                fprintf(stderr, "launch failed (mode=%d rows=%u)\n", mode, n);
                failures++;
                continue;
            }
            const size_t mid_n = (size_t)n * N_EXPERT * MID_DIM;
            const int mid_equal = memcmp(mid_a, mid_b, mid_n * sizeof(float)) == 0;
            int out_equal = 1, finite = 1, within = 1;
            printf("mode=%s rows=%u mid=%s",
                   mode ? "per-row" : "batched", n, mid_equal ? "bitwise" : "DIFFERS");
            for (uint32_t t = 0; t < n; t++) {
                double max_ref = 0.0, max_diff = 0.0;
                const float *a = out_a + (size_t)t * OUT_DIM, *b = out_b + (size_t)t * OUT_DIM;
                for (uint32_t i = 0; i < OUT_DIM; i++) {
                    if (!isfinite(a[i]) || !isfinite(b[i])) finite = 0;
                    const double d = fabs((double)a[i] - (double)b[i]);
                    if (d > max_diff) max_diff = d;
                    if (fabs((double)b[i]) > max_ref) max_ref = fabs((double)b[i]);
                }
                if (memcmp(a, b, OUT_DIM * sizeof(float)) != 0) out_equal = 0;
                if (!(max_diff <= 1e-5 * max_ref)) within = 0;
                printf(" row%u:maxabs=%.3g(ref %.3g)", t, max_diff, max_ref);
            }
            printf(" out=%s finite=%d\n", out_equal ? "bitwise" : (within ? "within-tol" : "MISMATCH"), finite);
            if (!mid_equal || !finite || !within || (mode == 1 && !out_equal)) failures++;
        }
    }
    ds4_gpu_cleanup();
    printf("%s (%d failing cases)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
