/* D0: model-free bit-equality of the DSpark verifier's multi-row kernels
 * against the one-token decode kernels on identical inputs.
 *
 * For each kernel the verifier calls with rows = 2..6 (gfx1151 V4.1 shapes),
 * the same synthetic weights and the same per-row inputs are run once with
 * all rows and once per row with rows = 1, and every output element is
 * compared bitwise. Outputs are poisoned with NaN before each launch so a
 * row or tail that a kernel fails to write is reported.
 *
 * Kernels marked "exact" are expected to be bit-identical (they share the
 * decode kernel's lane ownership and reduction order); a difference there is
 * a failure (exit 1). Kernels marked "info" are known to use a different
 * summation geometry; their maximum difference is reported only.
 *
 * Usage: tests/test_d0_verify_rows_rocm   (needs the GPU; run under
 *        flock /data/ds4-bench/gpu.lock on kumanix) */

#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define EMBD 5120u
#define HC 4u
#define LORA_Q 1280u
#define HEAD_DIM 512u
#define HEADS 64u
#define N_EXPERT 384u
#define FF_EXP 2304u
#define MAX_ROWS 6u
#define MXFP4_TYPE 39u

static int g_fail;
/* The verifier runs with ds4_gpu_set_dspark_verify_mode(true); kernels may
 * dispatch differently outside it (prefill/MTP batches). */
static bool g_verify_mode = true;
/* Production inputs to every checked kernel are BF16-rounded activations
 * (norms, heads, low, residual). With BF16 inputs the int8 x activation
 * products are exact in F32, so compiler contraction differences inside a
 * lane's 4-element dot cannot show; random F32 inputs expose them. */
static bool g_bf16_inputs = true;

static float bf16_round(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    x += 0x7fffu + ((x >> 16) & 1u);
    x &= 0xffff0000u;
    memcpy(&f, &x, 4);
    return f;
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static float frand(uint32_t *s) {
    *s = mix32(*s + 0x9e3779b9u);
    return ((float)(*s >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static uint16_t f32_to_f16(float f) {
    /* Inputs here are small normal numbers; truncating conversion is fine. */
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7bffu);
    return (uint16_t)(sign | ((uint32_t)e << 10) | ((x >> 13) & 0x3ffu));
}

/* Q8_0: per 32 values an F16 scale then 32 int8. */
static void fill_q8(uint8_t *w, uint64_t rows, uint32_t width, uint32_t seed) {
    const uint32_t blocks = width / 32u;
    for (uint64_t r = 0; r < rows; r++)
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 34u;
            uint32_t s = mix32(seed ^ (uint32_t)(r * 7919u + b));
            const uint16_t d = f32_to_f16(0.002f + 0.004f * (frand(&s) * 0.5f + 0.5f));
            memcpy(blk, &d, 2);
            for (uint32_t i = 0; i < 32u; i++) blk[2 + i] = (uint8_t)(int8_t)(frand(&s) * 127.0f);
        }
}

static void fill_f32(float *w, uint64_t n, uint32_t seed, float scale) {
    uint32_t s = seed;
    for (uint64_t i = 0; i < n; i++) w[i] = frand(&s) * scale;
}

static void fill_f16(uint16_t *w, uint64_t n, uint32_t seed, float scale) {
    uint32_t s = seed;
    for (uint64_t i = 0; i < n; i++) w[i] = f32_to_f16(frand(&s) * scale);
}

/* MXFP4 blocks: one E8M0 scale byte and 16 bytes of packed nibbles. */
static void fill_mxfp4(uint8_t *w, uint64_t blocks, uint32_t seed) {
    uint32_t s = seed;
    for (uint64_t b = 0; b < blocks; b++) {
        s = mix32(s + (uint32_t)b);
        w[b * 17u] = (uint8_t)(118u + (s % 6u));
        for (uint32_t i = 1; i < 17u; i++) { s = mix32(s); w[b * 17u + i] = (uint8_t)s; }
    }
}

typedef struct {
    void *map;
    uint64_t size, used;
} arena;

static uint64_t take(arena *a, uint64_t bytes) {
    const uint64_t off = (a->used + 4095u) & ~UINT64_C(4095);
    a->used = off + bytes;
    if (a->used > a->size) { fprintf(stderr, "arena overflow\n"); exit(2); }
    return off;
}

typedef int (*launch_fn)(void *ctx, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out,
                         uint32_t rows);

/* Compare outputs of `rows` rows at once against rows = 1 per row. ins/outs
 * are per-tensor row widths (floats). */
static void check(const char *name, bool exact, launch_fn fn, void *ctx,
                  const uint32_t *in_w, uint32_t n_in, const uint32_t *out_w, uint32_t n_out,
                  uint32_t seed) {
    for (uint32_t rows = 2; rows <= MAX_ROWS; rows++) {
        ds4_gpu_tensor *in[4] = {0}, *outb[4] = {0}, *in1[4] = {0}, *out1[4] = {0};
        float *host_in[4] = {0}, *hb[4] = {0}, *h1[4] = {0};
        bool ok = true;
        for (uint32_t t = 0; t < n_in; t++) {
            const uint64_t n = (uint64_t)rows * in_w[t];
            host_in[t] = malloc(n * 4u);
            fill_f32(host_in[t], n, seed * 31u + t, 1.0f);
            if (g_bf16_inputs)
                for (uint64_t i = 0; i < n; i++) host_in[t][i] = bf16_round(host_in[t][i]);
            in[t] = ds4_gpu_tensor_alloc(n * 4u);
            in1[t] = ds4_gpu_tensor_alloc((uint64_t)in_w[t] * 4u);
            ok = ok && host_in[t] && in[t] && in1[t] &&
                ds4_gpu_tensor_write(in[t], 0, host_in[t], n * 4u);
        }
        for (uint32_t t = 0; t < n_out; t++) {
            const uint64_t n = (uint64_t)rows * out_w[t];
            hb[t] = malloc(n * 4u);
            h1[t] = malloc(n * 4u);
            float *nan = malloc(n * 4u);
            for (uint64_t i = 0; i < n; i++) nan[i] = NAN;
            outb[t] = ds4_gpu_tensor_alloc(n * 4u);
            out1[t] = ds4_gpu_tensor_alloc((uint64_t)out_w[t] * 4u);
            ok = ok && hb[t] && h1[t] && outb[t] && out1[t] &&
                ds4_gpu_tensor_write(outb[t], 0, nan, n * 4u);
            free(nan);
        }
        ds4_gpu_set_dspark_verify_mode(g_verify_mode);
        ok = ok && fn(ctx, in, outb, rows) && ds4_gpu_synchronize();
        for (uint32_t t = 0; ok && t < n_out; t++)
            ok = ds4_gpu_tensor_read(outb[t], 0, hb[t], (uint64_t)rows * out_w[t] * 4u);
        for (uint32_t r = 0; ok && r < rows; r++) {
            for (uint32_t t = 0; ok && t < n_in; t++)
                ok = ds4_gpu_tensor_write(in1[t], 0, host_in[t] + (uint64_t)r * in_w[t],
                                          (uint64_t)in_w[t] * 4u);
            for (uint32_t t = 0; ok && t < n_out; t++) {
                float *nan = malloc((uint64_t)out_w[t] * 4u);
                for (uint32_t i = 0; i < out_w[t]; i++) nan[i] = NAN;
                ok = ds4_gpu_tensor_write(out1[t], 0, nan, (uint64_t)out_w[t] * 4u);
                free(nan);
            }
            ds4_gpu_set_dspark_verify_mode(g_verify_mode);
            ok = ok && fn(ctx, in1, out1, 1) && ds4_gpu_synchronize();
            for (uint32_t t = 0; ok && t < n_out; t++)
                ok = ds4_gpu_tensor_read(out1[t], 0, h1[t] + (uint64_t)r * out_w[t],
                                         (uint64_t)out_w[t] * 4u);
        }
        uint64_t ndiff = 0, nnan = 0, total = 0;
        float maxabs = 0.0f;
        for (uint32_t t = 0; ok && t < n_out; t++) {
            const uint64_t n = (uint64_t)rows * out_w[t];
            total += n;
            for (uint64_t i = 0; i < n; i++) {
                if (isnan(hb[t][i]) || isnan(h1[t][i])) { nnan++; continue; }
                if (memcmp(&hb[t][i], &h1[t][i], 4) == 0) continue;
                ndiff++;
                const float d = fabsf(hb[t][i] - h1[t][i]);
                if (d > maxabs) maxabs = d;
            }
        }
        const bool equal = ok && !ndiff && !nnan;
        printf("%-34s %s rows=%u %-5s %s ndiff=%llu/%llu nan=%llu maxabs=%.6g\n", name,
               g_bf16_inputs ? "bf16-in" : "f32-in ", rows,
               exact ? "exact" : "info", !ok ? "ERROR" : equal ? "EQUAL" : "DIFF",
               (unsigned long long)ndiff, (unsigned long long)total,
               (unsigned long long)nnan, (double)maxabs);
        if (!ok || (exact && !equal)) g_fail = 1;
        for (uint32_t t = 0; t < 4; t++) {
            ds4_gpu_tensor_free(in[t]); ds4_gpu_tensor_free(in1[t]);
            ds4_gpu_tensor_free(outb[t]); ds4_gpu_tensor_free(out1[t]);
            free(host_in[t]); free(hb[t]); free(h1[t]);
        }
    }
}

typedef struct {
    const void *map;
    uint64_t size, off, off2, off3;
    uint32_t width, outputs;
    uint32_t kind;
} ctx_t;

static int run_q8(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_q8_projection_rows(out[0], x->map, x->size, x->off, x->width,
                                           x->outputs, rows, in[0]);
}
static int run_q8_bf16(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_q8_projection_rows_bf16(out[0], x->map, x->size, x->off, x->width,
                                                x->outputs, rows, in[0]);
}
static int run_f32(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_f32_projection_rows(out[0], x->map, x->size, x->off, x->width,
                                            x->outputs, rows, in[0]);
}
static int run_f16(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_projection_rows(out[0], x->map, x->size, x->off, x->width,
                                        x->outputs, rows, in[0]);
}
static int run_pair(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_q8_attention_pair_bf16(out[0], out[1], x->map, x->size, x->off,
                                               x->off2, rows, in[0]);
}
static int run_attn_out(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_attention_output_batch(out[0], out[1], x->map, x->size, x->off,
                                               x->off2, in[0], rows);
}
static int run_norm(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_norm(out[0], in[0], x->map, x->size, x->off, x->width, rows, 1e-6f);
}
static int run_hc_sum(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    const ctx_t *x = c;
    return ds4_gpu_dsv41_hc_sum(out[0], in[0], in[1], EMBD, rows, x->width);
}
static int run_hc_expand(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    (void)c;
    return ds4_gpu_dsv41_hc_expand(out[0], in[0], in[1], in[2], EMBD, rows);
}

/* Routed MXFP4 experts, resident table, verify mode. Inputs: x, and a fixed
 * routing per row taken from the context (selected/weights tensors built
 * from the row index stored in in[1]'s first float). */
typedef struct {
    ctx_t base;
    int32_t sel[MAX_ROWS][6];
    float wts[MAX_ROWS][6];
    bool verify_mode;
} moe_ctx;

static int run_moe(void *c, ds4_gpu_tensor *const *in, ds4_gpu_tensor *const *out, uint32_t rows) {
    moe_ctx *m = c;
    /* in[1] holds each row's routing slot id (as float) so rows = 1 reruns
     * pick the same experts as the batched call. */
    float ids[MAX_ROWS];
    int32_t sel[MAX_ROWS * 6];
    float wts[MAX_ROWS * 6];
    if (!ds4_gpu_tensor_read(in[1], 0, ids, (uint64_t)rows * 4u)) return 0;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t k = (uint32_t)(fabsf(ids[r]) * 1000.0f) % MAX_ROWS;
        memcpy(sel + r * 6, m->sel[k], sizeof(m->sel[k]));
        memcpy(wts + r * 6, m->wts[k], sizeof(m->wts[k]));
    }
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * 4u);
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * 4u);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * FF_EXP * 4u);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * FF_EXP * 4u);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * FF_EXP * 4u);
    ds4_gpu_tensor *ex = ds4_gpu_tensor_alloc((uint64_t)rows * 6u * EMBD * 4u);
    const uint64_t grow = (EMBD / 32u) * 17u, drow = (FF_EXP / 32u) * 17u;
    int ok = ts && tw && gate && up && mid && ex &&
        ds4_gpu_tensor_write(ts, 0, sel, (uint64_t)rows * 24u) &&
        ds4_gpu_tensor_write(tw, 0, wts, (uint64_t)rows * 24u);
    ds4_gpu_set_dspark_verify_mode(m->verify_mode);
    if (ok && rows == 1u) {
        ok = ds4_gpu_routed_moe_one_tensor(out[0], gate, up, mid, ex, m->base.map, m->base.size,
            m->base.off, m->base.off2, m->base.off3, MXFP4_TYPE, MXFP4_TYPE,
            grow * FF_EXP, grow, drow * EMBD, drow, EMBD, FF_EXP, EMBD, ts, tw,
            N_EXPERT, 6u, 10.0f, in[0], NULL, 0u, true);
    } else if (ok) {
        bool f16 = false;
        ok = ds4_gpu_routed_moe_batch_tensor(out[0], gate, up, mid, ex, m->base.map, m->base.size,
            m->base.off, m->base.off2, m->base.off3, MXFP4_TYPE, MXFP4_TYPE,
            grow * FF_EXP, grow, drow * EMBD, drow, EMBD, FF_EXP, EMBD, ts, tw,
            N_EXPERT, 6u, 10.0f, in[0], 0u, rows, &f16, true);
    }
    ds4_gpu_set_dspark_verify_mode(g_verify_mode);
    ds4_gpu_tensor_free(ts); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(ex);
    return ok;
}

int main(void) {
    const uint64_t q8_row_5120 = (EMBD / 32u) * 34u;
    const uint64_t mx_gate = (uint64_t)N_EXPERT * FF_EXP * (EMBD / 32u) * 17u;
    const uint64_t mx_down = (uint64_t)N_EXPERT * EMBD * (FF_EXP / 32u) * 17u;
    arena a = {0};
    a.size = UINT64_C(16) << 30;
    FILE *fp = tmpfile();
    if (!fp || ftruncate(fileno(fp), (off_t)a.size) != 0) return 2;
    a.map = mmap(NULL, a.size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(fp), 0);
    if (a.map == MAP_FAILED) return 2;
    uint8_t *base = a.map;
    const uint64_t q_a = take(&a, LORA_Q * q8_row_5120);
    const uint64_t kv = take(&a, HEAD_DIM * q8_row_5120);
    const uint64_t q_b = take(&a, (uint64_t)HEADS * HEAD_DIM * (LORA_Q / 32u) * 34u);
    const uint64_t sh_gate = take(&a, FF_EXP * q8_row_5120);
    const uint64_t sh_down = take(&a, (uint64_t)EMBD * (FF_EXP / 32u) * 34u);
    const uint64_t head = take(&a, UINT64_C(16384) * q8_row_5120); /* vocabulary slice */
    const uint64_t out_a = take(&a, UINT64_C(8192) * 128u * 34u);
    const uint64_t out_b = take(&a, UINT64_C(5120) * 256u * 34u);
    const uint64_t router = take(&a, (uint64_t)N_EXPERT * EMBD * 4u);
    const uint64_t hc_fn = take(&a, (uint64_t)24u * HC * EMBD * 2u);
    const uint64_t norm_w = take(&a, EMBD * 4u);
    const uint64_t g_mx = take(&a, mx_gate), u_mx = take(&a, mx_gate), d_mx = take(&a, mx_down);
    fill_q8(base + q_a, LORA_Q, EMBD, 1);
    fill_q8(base + kv, HEAD_DIM, EMBD, 2);
    fill_q8(base + q_b, (uint64_t)HEADS * HEAD_DIM, LORA_Q, 3);
    fill_q8(base + sh_gate, FF_EXP, EMBD, 4);
    fill_q8(base + sh_down, EMBD, FF_EXP, 5);
    fill_q8(base + head, 16384u, EMBD, 6);
    fill_q8(base + out_a, 8192u, 4096u, 7);
    fill_q8(base + out_b, 5120u, 8192u, 8);
    fill_f32((float *)(base + router), (uint64_t)N_EXPERT * EMBD, 9, 0.02f);
    fill_f16((uint16_t *)(base + hc_fn), (uint64_t)24u * HC * EMBD, 10, 0.02f);
    fill_f32((float *)(base + norm_w), EMBD, 11, 1.0f);
    fill_mxfp4(base + g_mx, mx_gate / 17u, 12);
    fill_mxfp4(base + u_mx, mx_gate / 17u, 13);
    fill_mxfp4(base + d_mx, mx_down / 17u, 14);

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 2; }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    if (!ds4_gpu_set_model_map(a.map, a.used) || !ds4_gpu_set_model_fd(fileno(fp))) {
        fprintf(stderr, "model map setup failed\n");
        return 2;
    }
    printf("gfx1151 fast path: %d\n", ds4_gpu_dspark_gfx1151_fast_path());
    const uint32_t w5120[] = {EMBD}, w1280[] = {LORA_Q}, w2304[] = {FF_EXP};
    ctx_t c = {.map = a.map, .size = a.used};
    for (int bf = 1; bf >= 0; bf--) {
    g_bf16_inputs = bf;
    const bool ex = bf;  /* exactness is claimed for production (BF16) inputs */

    c.off = q_a; c.width = EMBD; c.outputs = LORA_Q;
    const uint32_t o1280[] = {LORA_Q};
    check("q8 rows 5120->1280 (attn q_a)", ex, run_q8, &c, w5120, 1, o1280, 1, 1);
    check("q8 rows bf16 5120->1280", ex, run_q8_bf16, &c, w5120, 1, o1280, 1, 2);
    c.off = q_b; c.width = LORA_Q; c.outputs = HEADS * HEAD_DIM;
    const uint32_t oq[] = {HEADS * HEAD_DIM};
    check("q8 rows bf16 1280->32768 (q_b)", ex, run_q8_bf16, &c, w1280, 1, oq, 1, 3);
    c.off = sh_gate; c.width = EMBD; c.outputs = FF_EXP;
    const uint32_t off_[] = {FF_EXP};
    check("q8 rows bf16 5120->2304 (shared)", ex, run_q8_bf16, &c, w5120, 1, off_, 1, 4);
    c.off = sh_down; c.width = FF_EXP; c.outputs = EMBD;
    const uint32_t oe[] = {EMBD};
    check("q8 rows bf16 2304->5120 (shared dn)", ex, run_q8_bf16, &c, w2304, 1, oe, 1, 5);
    c.off = head; c.width = EMBD; c.outputs = 16384u;
    const uint32_t oh[] = {16384u};
    check("q8 rows 5120->16384 (head slice)", ex, run_q8, &c, w5120, 1, oh, 1, 6);
    c.off = q_a; c.off2 = kv;
    const uint32_t opair[] = {LORA_Q, HEAD_DIM};
    check("q8 attention pair bf16", ex, run_pair, &c, w5120, 1, opair, 2, 7);
    c.off = out_a; c.off2 = out_b;
    const uint32_t wheads[] = {HEADS * HEAD_DIM};
    const uint32_t oout[] = {EMBD, 8192u};
    check("attention output A+B batch", ex, run_attn_out, &c, wheads, 1, oout, 2, 8);
    c.off = router; c.width = EMBD; c.outputs = N_EXPERT;
    const uint32_t orouter[] = {N_EXPERT};
    check("f32 rows 5120->384 (router)", ex, run_f32, &c, w5120, 1, orouter, 1, 9);
    c.off = hc_fn; c.width = HC * EMBD; c.outputs = 24u;
    const uint32_t whc[] = {HC * EMBD}, o24[] = {24u};
    check("f16 rows 20480->24 (hc mix)", ex, run_f16, &c, whc, 1, o24, 1, 10);
    c.off = norm_w; c.width = EMBD;
    check("norm bf16 5120", ex, run_norm, &c, w5120, 1, oe, 1, 11);
    const uint32_t whcs4[] = {HC * EMBD, 4u}, whcs24[] = {HC * EMBD, 24u};
    c.width = 4u;
    check("hc_sum stride 4", ex, run_hc_sum, &c, whcs4, 2, oe, 1, 12);
    c.width = 24u;
    check("hc_sum stride 24", ex, run_hc_sum, &c, whcs24, 2, oe, 1, 13);
    const uint32_t whce[] = {EMBD, HC * EMBD, 24u}, ohce[] = {HC * EMBD};
    check("hc_expand", ex, run_hc_expand, &c, whce, 3, ohce, 1, 14);

    }
    g_bf16_inputs = true;
    moe_ctx m = {.base = {.map = a.map, .size = a.used, .off = g_mx, .off2 = u_mx, .off3 = d_mx}};
    uint32_t s = 99u;
    for (uint32_t k = 0; k < MAX_ROWS; k++)
        for (uint32_t j = 0; j < 6u; j++) {
            s = mix32(s);
            m.sel[k][j] = (int32_t)((s % (N_EXPERT / 6u)) * 6u + j);
            m.wts[k][j] = 0.05f + 0.1f * (float)j;
        }
    const uint32_t wmoe[] = {EMBD, 1u}, oe[] = {EMBD};
    m.verify_mode = true;
    check("mxfp4 routed resident verify-mode", true, run_moe, &m, wmoe, 2, oe, 1, 15);
    m.verify_mode = false;
    check("mxfp4 routed resident prefill-mode", false, run_moe, &m, wmoe, 2, oe, 1, 16);
    m.verify_mode = true;
    g_bf16_inputs = false;
    check("mxfp4 routed resident verify f32-in", false, run_moe, &m, wmoe, 2, oe, 1, 17);

    printf("%s\n", g_fail ? "D0 kernel rows: FAIL" : "D0 kernel rows: PASS");
    return g_fail;
}
