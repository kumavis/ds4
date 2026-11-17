// C1 model-free check: candidate decode kernels against verbatim copies of
// the kernels they would replace.  Bitwise comparison on random inputs and
// GPU time per launch (hipEvents, 200 launches, weights larger than L2 are
// cycled through several copies so each launch reads DRAM).
// Build with the ds4 ROCm flags:
//   hipcc -std=c++20 -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 -o c1_exact c1_exact.cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

/* ---- verbatim helpers (rocm/ds4_rocm_common.cuh, ds4_rocm_q8.cuh, ds4_rocm_v41.cuh) ---- */
__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}
__device__ static float q8_0_scale_scalar(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}
__device__ static float v41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

/* ---- old kernels, verbatim ---- */
__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void v41_hc_ordered_prefetch_kernel(float *out,
        const __half *weights, const float *input) {
    constexpr unsigned width = 20480u, outputs = 24u, chunk = 640u;
    const unsigned lane = threadIdx.x, row = blockIdx.x, token = blockIdx.y;
    const __half *w = weights + (uint64_t)row * width;
    const float *x = input + (uint64_t)token * width;
    __shared__ float partial[32];
    float sum = 0.0f;
    for (unsigned base = lane * chunk; base < (lane + 1u) * chunk; base += 16u) {
        float ww[16], xx[16];
#pragma unroll
        for (unsigned u = 0; u < 16u; u += 2u) {
            const float2 wp = __half22float2(*reinterpret_cast<const __half2 *>(w + base + u));
            const float2 xp = *reinterpret_cast<const float2 *>(x + base + u);
            ww[u] = wp.x; ww[u + 1u] = wp.y;
            xx[u] = xp.x; xx[u + 1u] = xp.y;
        }
#pragma unroll
        for (unsigned u = 0; u < 16u; u++) sum = __fmaf_rn(ww[u], xx[u], sum);
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0u) {
        float total = 0.0f;
        for (unsigned i = 0; i < 32u; i++) total += partial[i];
        out[(uint64_t)token * outputs + row] = total;
    }
}

template<bool ROUND_BF16 = false>
__global__ static void v41_q8_f32_blocks4_kernel(float *out,
        const unsigned char *weights, const float *input,
        uint32_t width, uint32_t outputs, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    const uint32_t blocks = width / 32u;
    float acc = 0.0f;
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        const uint32_t j = (lane & 7u) * 4u;
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = ROUND_BF16 ? v41_bf16(acc) : acc;
}


__global__ static void rms_norm_weight_bf16_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = v41_bf16(xr[i] * scale * w[i]);
    }
}
template <uint32_t PER>
__global__ static void c1_rms_norm_weight_bf16_pre_kernel(float *out, const float *x, const float *w, uint32_t n, float eps) {
    const float *xr = x;
    float *orow = out;
    float v[PER];
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) v[k] = xr[threadIdx.x + k * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) sum += v[k] * v[k];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) {
        const uint32_t i = threadIdx.x + k * 256u;
        orow[i] = v41_bf16(v[k] * scale * w[i]);
    }
}


__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}
__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

/* ---- candidates ---- */
/* C1 (DS4_ROCM_C1_SINKHORN, default 1; 0 = old kernel): the one-row HC
 * split + Sinkhorn on 16 threads (one per comb element) instead of one.
 * Every element sees the same operations in the same order as
 * hc4_split_one (row sums and column sums are recomputed by each thread in
 * the original index order), so the result is bitwise equal
 * (tools/c1/c1_exact.cpp). */
__global__ static void c1_hc_split_sinkhorn16_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const unsigned t = threadIdx.x;
    const unsigned r = t >> 2u, col = t & 3u;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (t < 4u) {
        float z = mix[t] * pre_scale + base[t];
        out[t] = 1.0f / (1.0f + expf(-z)) + epsv;
    } else if (t < 8u) {
        float z = mix[t] * post_scale + base[t];
        out[t] = 2.0f / (1.0f + expf(-z));
    }
    __shared__ float c[16];
    float v[4];
    float m = -INFINITY;
    for (int k = 0; k < 4; k++) {
        v[k] = mix[8 + r * 4 + k] * comb_scale + base[8 + r * 4 + k];
        m = fmaxf(m, v[k]);
    }
    float s = 0.0f;
    for (int k = 0; k < 4; k++) {
        v[k] = expf(v[k] - m);
        s += v[k];
    }
    c[t] = v[col] / s + epsv;
    __syncthreads();
    {
        float cs = epsv;
        for (int rr = 0; rr < 4; rr++) cs += c[rr * 4 + col];
        const float mine = c[t];
        __syncthreads();
        c[t] = mine / cs;
        __syncthreads();
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        float rs = epsv;
        for (int cc = 0; cc < 4; cc++) rs += c[r * 4 + cc];
        float mine = c[t];
        __syncthreads();
        c[t] = mine / rs;
        __syncthreads();
        float cs = epsv;
        for (int rr = 0; rr < 4; rr++) cs += c[rr * 4 + col];
        mine = c[t];
        __syncthreads();
        c[t] = mine / cs;
        __syncthreads();
    }
    out[8 + t] = c[t];
}

// RMS norm: each thread's N/256 values loaded before the in-order sum.
template <uint32_t PER>
__global__ static void c1_rms_norm_plain_pre_kernel(float *out, const float *x, uint32_t n, float eps) {
    const float *xr = x;
    float v[PER];
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) v[k] = xr[threadIdx.x + k * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) sum += v[k] * v[k];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) out[threadIdx.x + k * 256u] = v[k] * scale;
}
// Same, explicit fused multiply-add chain (what fp-contract=fast should emit).
template <uint32_t PER>
__global__ static void c1_rms_norm_plain_fma_kernel(float *out, const float *x, uint32_t n, float eps) {
    float v[PER];
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) v[k] = x[threadIdx.x + k * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) sum = __fmaf_rn(v[k], v[k], sum);
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
    for (uint32_t k = 0; k < PER; k++) out[threadIdx.x + k * 256u] = v[k] * scale;
}

template <unsigned STEP>
__global__ static void c1_hc_ordered_wide_kernel(float *out,
        const __half *weights, const float *input) {
    constexpr unsigned width = 20480u, outputs = 24u, chunk = 640u;
    const unsigned lane = threadIdx.x, row = blockIdx.x, token = blockIdx.y;
    const __half *w = weights + (uint64_t)row * width;
    const float *x = input + (uint64_t)token * width;
    __shared__ float partial[32];
    float sum = 0.0f;
    for (unsigned base = lane * chunk; base < (lane + 1u) * chunk; base += STEP) {
        float ww[STEP], xx[STEP];
#pragma unroll
        for (unsigned u = 0; u < STEP; u += 2u) {
            const float2 wp = __half22float2(*reinterpret_cast<const __half2 *>(w + base + u));
            const float2 xp = *reinterpret_cast<const float2 *>(x + base + u);
            ww[u] = wp.x; ww[u + 1u] = wp.y;
            xx[u] = xp.x; xx[u + 1u] = xp.y;
        }
#pragma unroll
        for (unsigned u = 0; u < STEP; u++) sum = __fmaf_rn(ww[u], xx[u], sum);
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0u) {
        float total = 0.0f;
        for (unsigned i = 0; i < 32u; i++) total += partial[i];
        out[(uint64_t)token * outputs + row] = total;
    }
}

template<bool ROUND_BF16, uint32_t U>
__global__ static void c1_q8_f32_blocks4_pipe_kernel(float *out,
        const unsigned char *weights, const float *input,
        uint32_t width, uint32_t outputs, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    const uint32_t blocks = width / 32u;
    const uint32_t j = (lane & 7u) * 4u;
    const unsigned char *rowp = weights + row * row_bytes;
    float acc = 0.0f;
    uint32_t b = lane / 8u;
    for (; b + 4u * (U - 1u) < blocks; b += 4u * U) {
        float d[U], wq[U][4], xv[U][4];
#pragma unroll
        for (uint32_t u = 0; u < U; u++) {
            const unsigned char *p = rowp + (uint64_t)(b + 4u * u) * 34u;
            d[u] = q8_0_scale_scalar(p);
#pragma unroll
            for (uint32_t k = 0; k < 4u; k++) {
                wq[u][k] = (float)((const int8_t *)(p + 2u))[j + k];
                xv[u][k] = input[(b + 4u * u) * 32u + j + k];
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < U; u++) {
            float value = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < 4u; k++) value += wq[u][k] * xv[u][k];
            acc += d[u] * value;
        }
    }
    for (; b < blocks; b += 4u) {
        const unsigned char *p = rowp + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = ROUND_BF16 ? v41_bf16(acc) : acc;
}

/* ---- harness ---- */
static std::mt19937 rng(12345);
static void fill_f32(std::vector<float> &v, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto &x : v) x = nd(rng);
}
static int cmp(const char *name, const float *a, const float *b, size_t n) {
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4) != 0) diff++;
    printf("  %-44s %s (%zu of %zu differ)\n", name, diff ? "DIFF" : "EQUAL", diff, n);
    return diff == 0;
}
template <typename F>
static float time_us(F launch, int iters) {
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int i = 0; i < 5; i++) launch(i);
    CK(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch(i);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
    return ms * 1000.0f / iters;
}

int main() {
    int all_ok = 1;
    const int iters = 200;
    // ---------- RMS norm plain, n = 20480 ----------
    {
        const uint32_t n = 20480u;
        printf("rms_norm_plain n=%u\n", n);
        std::vector<float> hx(n); float *x, *o0, *o1, *o2;
        CK(hipMalloc(&x, n * 4)); CK(hipMalloc(&o0, n * 4)); CK(hipMalloc(&o1, n * 4)); CK(hipMalloc(&o2, n * 4));
        std::vector<float> r0(n), r1(n), r2(n);
        for (int seed = 0; seed < 8; seed++) {
            fill_f32(hx, seed < 4 ? 1.0f : 30.0f);
            CK(hipMemcpy(x, hx.data(), n * 4, hipMemcpyHostToDevice));
            rms_norm_plain_kernel<<<1, 256>>>(o0, x, n, 1, 1e-6f);
            c1_rms_norm_plain_pre_kernel<80><<<1, 256>>>(o1, x, n, 1e-6f);
            c1_rms_norm_plain_fma_kernel<80><<<1, 256>>>(o2, x, n, 1e-6f);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(r0.data(), o0, n * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(r1.data(), o1, n * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(r2.data(), o2, n * 4, hipMemcpyDeviceToHost));
            char nm[64];
            snprintf(nm, sizeof nm, "seed %d preload", seed); all_ok &= cmp(nm, r0.data(), r1.data(), n);
            snprintf(nm, sizeof nm, "seed %d explicit fma", seed); cmp(nm, r0.data(), r2.data(), n);
        }
        printf("  time old %.2f us, preload %.2f us, fma %.2f us\n",
               time_us([&](int) { rms_norm_plain_kernel<<<1, 256>>>(o0, x, n, 1, 1e-6f); }, iters),
               time_us([&](int) { c1_rms_norm_plain_pre_kernel<80><<<1, 256>>>(o1, x, n, 1e-6f); }, iters),
               time_us([&](int) { c1_rms_norm_plain_fma_kernel<80><<<1, 256>>>(o2, x, n, 1e-6f); }, iters));
    }
    // ---------- weighted BF16 norm, n = 5120 / 1280 / 512, in place ----------
    for (uint32_t n : {5120u, 1280u, 512u}) {
        printf("rms_norm_weight_bf16 n=%u (in place, as ds41_norm(qr, qr))\n", n);
        std::vector<float> hx(n), hw(n), r0(n), r1(n);
        float *a, *b, *w;
        CK(hipMalloc(&a, n * 4)); CK(hipMalloc(&b, n * 4)); CK(hipMalloc(&w, n * 4));
        for (int seed = 0; seed < 8; seed++) {
            fill_f32(hx, seed < 4 ? 1.0f : 20.0f); fill_f32(hw, 1.0f);
            CK(hipMemcpy(a, hx.data(), n * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(b, hx.data(), n * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(w, hw.data(), n * 4, hipMemcpyHostToDevice));
            rms_norm_weight_bf16_kernel<<<1, 256>>>(a, a, w, n, 1, 1e-6f);
            if (n == 5120u) c1_rms_norm_weight_bf16_pre_kernel<20><<<1, 256>>>(b, b, w, n, 1e-6f);
            if (n == 1280u) c1_rms_norm_weight_bf16_pre_kernel<5><<<1, 256>>>(b, b, w, n, 1e-6f);
            if (n == 512u) c1_rms_norm_weight_bf16_pre_kernel<2><<<1, 256>>>(b, b, w, n, 1e-6f);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(r0.data(), a, n * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(r1.data(), b, n * 4, hipMemcpyDeviceToHost));
            char nm[64]; snprintf(nm, sizeof nm, "seed %d preload", seed);
            all_ok &= cmp(nm, r0.data(), r1.data(), n);
        }
        float *c; CK(hipMalloc(&c, n * 4));
        const float t0 = time_us([&](int) { rms_norm_weight_bf16_kernel<<<1, 256>>>(c, a, w, n, 1, 1e-6f); }, iters);
        const float t1 = time_us([&](int) {
            if (n == 5120u) c1_rms_norm_weight_bf16_pre_kernel<20><<<1, 256>>>(c, a, w, n, 1e-6f);
            if (n == 1280u) c1_rms_norm_weight_bf16_pre_kernel<5><<<1, 256>>>(c, a, w, n, 1e-6f);
            if (n == 512u) c1_rms_norm_weight_bf16_pre_kernel<2><<<1, 256>>>(c, a, w, n, 1e-6f); }, iters);
        printf("  time old %.2f us, preload %.2f us\n", t0, t1);
    }
    // ---------- HC split + Sinkhorn, one row ----------
    {
        printf("hc split sinkhorn (one row, %d iters)\n", 20);
        float *mix, *sc, *bs, *o0, *o1;
        CK(hipMalloc(&mix, 24 * 4)); CK(hipMalloc(&sc, 3 * 4)); CK(hipMalloc(&bs, 24 * 4));
        CK(hipMalloc(&o0, 24 * 4)); CK(hipMalloc(&o1, 24 * 4));
        std::vector<float> hm(24), hs(3), hb(24), r0(24), r1(24);
        for (int iters_s : {20, 7, 1}) for (int seed = 0; seed < 64; seed++) {
            fill_f32(hm, seed < 32 ? 1.0f : 8.0f); fill_f32(hs, 1.0f); fill_f32(hb, 1.0f);
            CK(hipMemcpy(mix, hm.data(), 96, hipMemcpyHostToDevice));
            CK(hipMemcpy(sc, hs.data(), 12, hipMemcpyHostToDevice));
            CK(hipMemcpy(bs, hb.data(), 96, hipMemcpyHostToDevice));
            hc_split_sinkhorn_kernel<<<1, 256>>>(o0, mix, sc, bs, 1, iters_s, 1e-6f);
            c1_hc_split_sinkhorn16_kernel<<<1, 16>>>(o1, mix, sc, bs, iters_s, 1e-6f);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(r0.data(), o0, 96, hipMemcpyDeviceToHost));
            CK(hipMemcpy(r1.data(), o1, 96, hipMemcpyDeviceToHost));
            char nm[64]; snprintf(nm, sizeof nm, "iters %d seed %d", iters_s, seed);
            all_ok &= cmp(nm, r0.data(), r1.data(), 24);
        }
        printf("  time old %.2f us, 16 threads %.2f us\n",
               time_us([&](int) { hc_split_sinkhorn_kernel<<<1, 256>>>(o0, mix, sc, bs, 1, 20, 1e-6f); }, iters),
               time_us([&](int) { c1_hc_split_sinkhorn16_kernel<<<1, 16>>>(o1, mix, sc, bs, 20, 1e-6f); }, iters));
    }
    // ---------- HC projection 20480 -> 24, F16 weights ----------
    {
        const uint32_t W = 20480u, O = 24u;
        printf("hc projection %u -> %u\n", W, O);
        const int copies = 64;  // 64 x 983 KB > L2, like 80 distinct layer weights per token
        std::vector<__half> hw((size_t)W * O); std::vector<float> hx(W), ftmp((size_t)W * O);
        __half *w; float *x, *o0, *o1;
        CK(hipMalloc(&w, (size_t)W * O * 2 * copies)); CK(hipMalloc(&x, W * 4));
        CK(hipMalloc(&o0, O * 4)); CK(hipMalloc(&o1, O * 4));
        std::vector<float> r0(O), r1(O);
        for (int c = 0; c < copies; c++) {
            fill_f32(ftmp, 0.05f);
            for (size_t i = 0; i < hw.size(); i++) hw[i] = __float2half(ftmp[i]);
            CK(hipMemcpy(w + (size_t)c * W * O, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        }
        for (int seed = 0; seed < 8; seed++) {
            fill_f32(hx, 1.0f);
            CK(hipMemcpy(x, hx.data(), W * 4, hipMemcpyHostToDevice));
            const __half *wc = w + (size_t)seed * W * O;
            v41_hc_ordered_prefetch_kernel<<<dim3(O, 1), 32>>>(o0, wc, x);
            CK(hipMemcpy(r0.data(), o0, O * 4, hipMemcpyDeviceToHost));
            c1_hc_ordered_wide_kernel<32><<<dim3(O, 1), 32>>>(o1, wc, x);
            CK(hipMemcpy(r1.data(), o1, O * 4, hipMemcpyDeviceToHost));
            char nm[64]; snprintf(nm, sizeof nm, "seed %d step32", seed); all_ok &= cmp(nm, r0.data(), r1.data(), O);
            c1_hc_ordered_wide_kernel<64><<<dim3(O, 1), 32>>>(o1, wc, x);
            CK(hipMemcpy(r1.data(), o1, O * 4, hipMemcpyDeviceToHost));
            snprintf(nm, sizeof nm, "seed %d step64", seed); all_ok &= cmp(nm, r0.data(), r1.data(), O);
            c1_hc_ordered_wide_kernel<128><<<dim3(O, 1), 32>>>(o1, wc, x);
            CK(hipMemcpy(r1.data(), o1, O * 4, hipMemcpyDeviceToHost));
            snprintf(nm, sizeof nm, "seed %d step128", seed); all_ok &= cmp(nm, r0.data(), r1.data(), O);
        }
        auto wcopy = [&](int i) { return w + (size_t)(i % copies) * W * O; };
        printf("  time old %.2f us, step32 %.2f, step64 %.2f, step128 %.2f us\n",
               time_us([&](int i) { v41_hc_ordered_prefetch_kernel<<<dim3(O, 1), 32>>>(o0, wcopy(i), x); }, iters),
               time_us([&](int i) { c1_hc_ordered_wide_kernel<32><<<dim3(O, 1), 32>>>(o1, wcopy(i), x); }, iters),
               time_us([&](int i) { c1_hc_ordered_wide_kernel<64><<<dim3(O, 1), 32>>>(o1, wcopy(i), x); }, iters),
               time_us([&](int i) { c1_hc_ordered_wide_kernel<128><<<dim3(O, 1), 32>>>(o1, wcopy(i), x); }, iters));
    }
    // ---------- Q8 blocks4 matvec ----------
    {
        struct Shape { uint32_t width, outputs; const char *name; } shapes[] = {
            {1280u, 32768u, "q_b 1280->32768"}, {5120u, 2304u, "shared gate/up 5120->2304"},
            {2304u, 5120u, "shared down 2304->5120"}, {5120u, 1280u, "5120->1280"}};
        for (auto &sh : shapes) {
            const uint64_t row_bytes = (uint64_t)(sh.width / 32u) * 34u;
            const uint64_t wbytes = row_bytes * sh.outputs;
            const int copies = (int)((512ull << 20) / wbytes) + 1;
            printf("q8 blocks4 %s (%.1f MB, %d copies)\n", sh.name, wbytes / 1e6, copies);
            std::vector<unsigned char> hw(wbytes); std::vector<float> hx(sh.width);
            std::uniform_int_distribution<int> q(-127, 127);
            std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
            unsigned char *w; float *x, *o0, *o1;
            CK(hipMalloc(&w, wbytes * copies)); CK(hipMalloc(&x, sh.width * 4));
            CK(hipMalloc(&o0, sh.outputs * 4)); CK(hipMalloc(&o1, sh.outputs * 4));
            for (int c = 0; c < copies; c++) {
                for (uint64_t r = 0; r < sh.outputs; r++)
                    for (uint32_t b = 0; b < sh.width / 32u; b++) {
                        unsigned char *p = hw.data() + r * row_bytes + b * 34u;
                        __half h = __float2half(sc(rng)); memcpy(p, &h, 2);
                        for (int k = 0; k < 32; k++) ((int8_t *)(p + 2))[k] = (int8_t)q(rng);
                    }
                CK(hipMemcpy(w + (size_t)c * wbytes, hw.data(), wbytes, hipMemcpyHostToDevice));
            }
            std::vector<float> r0(sh.outputs), r1(sh.outputs);
            const unsigned grid = (sh.outputs + 7u) / 8u;
            for (int seed = 0; seed < 4; seed++) {
                fill_f32(hx, 1.0f);
                CK(hipMemcpy(x, hx.data(), sh.width * 4, hipMemcpyHostToDevice));
                const unsigned char *wc = w + (size_t)(seed % copies) * wbytes;
                for (int round = 0; round < 2; round++) {
                    if (round) v41_q8_f32_blocks4_kernel<true><<<grid, 256>>>(o0, wc, x, sh.width, sh.outputs, row_bytes);
                    else v41_q8_f32_blocks4_kernel<false><<<grid, 256>>>(o0, wc, x, sh.width, sh.outputs, row_bytes);
                    CK(hipMemcpy(r0.data(), o0, sh.outputs * 4, hipMemcpyDeviceToHost));
                    if (round) c1_q8_f32_blocks4_pipe_kernel<true, 4><<<grid, 256>>>(o1, wc, x, sh.width, sh.outputs, row_bytes);
                    else c1_q8_f32_blocks4_pipe_kernel<false, 4><<<grid, 256>>>(o1, wc, x, sh.width, sh.outputs, row_bytes);
                    CK(hipMemcpy(r1.data(), o1, sh.outputs * 4, hipMemcpyDeviceToHost));
                    char nm[64]; snprintf(nm, sizeof nm, "seed %d %s pipe4", seed, round ? "bf16" : "f32");
                    all_ok &= cmp(nm, r0.data(), r1.data(), sh.outputs);
                }
            }
            auto wcopy = [&](int i) { return w + (size_t)(i % copies) * wbytes; };
            const float t0 = time_us([&](int i) { v41_q8_f32_blocks4_kernel<true><<<grid, 256>>>(o0, wcopy(i), x, sh.width, sh.outputs, row_bytes); }, iters);
            const float t1 = time_us([&](int i) { c1_q8_f32_blocks4_pipe_kernel<true, 4><<<grid, 256>>>(o1, wcopy(i), x, sh.width, sh.outputs, row_bytes); }, iters);
            const float t2 = time_us([&](int i) { c1_q8_f32_blocks4_pipe_kernel<true, 8><<<grid, 256>>>(o1, wcopy(i), x, sh.width, sh.outputs, row_bytes); }, iters);
            printf("  time old %.2f us (%.0f GB/s), pipe4 %.2f us (%.0f GB/s), pipe8 %.2f us (%.0f GB/s)\n",
                   t0, wbytes / (t0 * 1e3), t1, wbytes / (t1 * 1e3), t2, wbytes / (t2 * 1e3));
            CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(o0)); CK(hipFree(o1));
        }
    }
    printf("C1 EXACT %s\n", all_ok ? "ALL EQUAL" : "SOME DIFF");
    return 0;
}
