// D20 device ceiling microbenchmark (model-free).
//  1. WMMA throughput from registers only: v_wmma_f32_16x16x16_f16 and
//     v_wmma_i32_16x16x16_iu8 (wave32), 8 independent accumulators per wave,
//     enough waves to fill every CU.  FLOP per instruction = 2*16*16*16.
//  2. hipBLASLt bf16 GEMM (bf16 in, f32 compute, bf16 out) at 4096^3 and at
//     one expert-sized shape (M rows x N=2304 x K=5120) for M = 32, 256, 2048.
// Build: hipcc -O3 --offload-arch=gfx1151 peak.cpp -lhipblaslt -o peak
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %d\n", hipGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "hipBLASLt %d at %d\n", (int)s_, __LINE__); exit(1);} } while (0)

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i8v __attribute__((ext_vector_type(8)));

__global__ void wmma_f16(float *out, int iters) {
    h16 a, b;
    for (int i = 0; i < 16; i++) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
    f8 c[8] = {};
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int k = 0; k < 8; k++) c[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[k]);
    }
    float s = 0;
    for (int k = 0; k < 8; k++) for (int i = 0; i < 8; i++) s += c[k][i];
    if (s == 12345.678f) out[0] = s;
}
__global__ void wmma_iu8(int *out, int iters) {
    i4v a, b;
    for (int i = 0; i < 4; i++) { a[i] = threadIdx.x * 7 + i; b[i] = i * 3 + 1; }
    i8v c[8] = {};
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int k = 0; k < 8; k++) c[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[k], false);
    }
    int s = 0;
    for (int k = 0; k < 8; k++) for (int i = 0; i < 8; i++) s += c[k][i];
    if (s == 12345) out[0] = s;
}

static double run_wmma(bool i8, int cus) {
    const int blocks = cus * 16, threads = 128, iters = 4096;
    void *o;
    CK(hipMalloc(&o, 64));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int rep = 0; rep < 2; rep++) {
        CK(hipEventRecord(e0));
        if (i8) hipLaunchKernelGGL(wmma_iu8, dim3(blocks), dim3(threads), 0, 0, (int *)o, iters);
        else hipLaunchKernelGGL(wmma_f16, dim3(blocks), dim3(threads), 0, 0, (float *)o, iters);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
    }
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    const double waves = (double)blocks * threads / 32.0;
    const double ops = waves * iters * 8.0 * 2.0 * 16 * 16 * 16;
    CK(hipFree(o));
    return ops / (ms * 1e-3) / 1e12;
}

static double run_gemm(hipblasLtHandle_t h, int M, int N, int K, int reps) {
    // C[M,N] = A[M,K] * B[K,N]; column-major hipBLASLt: compute C^T = B^T A^T,
    // i.e. m=N, n=M, k=K with opA = T on a K x N weight stored row-major...
    // We only need timing, so use plain column-major m=N, n=M, k=K, no transposes.
    void *A, *B, *C, *ws;
    const size_t wsb = 64u << 20;
    CK(hipMalloc(&A, (size_t)N * K * 2)); CK(hipMalloc(&B, (size_t)K * M * 2));
    CK(hipMalloc(&C, (size_t)N * M * 2)); CK(hipMalloc(&ws, wsb));
    CK(hipMemset(A, 0, (size_t)N * K * 2)); CK(hipMemset(B, 0, (size_t)K * M * 2));
    hipblasLtMatmulDesc_t d; hipblasLtMatrixLayout_t la, lb, lc;
    CB(hipblasLtMatmulDescCreate(&d, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CB(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, N, K, N));
    CB(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, M, K));
    CB(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, N, M, N));
    hipblasLtMatmulPreference_t pref;
    CB(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t wsz = wsb;
    CB(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
    hipblasLtMatmulHeuristicResult_t res[4];
    int got = 0;
    CB(hipblasLtMatmulAlgoGetHeuristic(h, d, la, lb, lc, lc, pref, 4, res, &got));
    if (got == 0) { printf("  no algorithm for %dx%dx%d\n", M, N, K); return 0; }
    float alpha = 1, beta = 0;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CB(hipblasLtMatmul(h, d, &alpha, A, la, B, lb, &beta, C, lc, C, lc, &res[0].algo, ws, wsb, 0));
    CK(hipEventRecord(e0));
    for (int r = 0; r < reps; r++)
        CB(hipblasLtMatmul(h, d, &alpha, A, la, B, lb, &beta, C, lc, C, lc, &res[0].algo, ws, wsb, 0));
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    hipFree(A); hipFree(B); hipFree(C); hipFree(ws);
    const double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("  bf16 GEMM M=%d N=%d K=%d: %.3f ms, %.2f TFLOP/s (%d heuristic results)\n", M, N, K, ms, tf, got);
    return tf;
}

int main() {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("device %s gcnArch %s CUs %d clock %d kHz warpSize %d\n", p.name, p.gcnArchName,
           p.multiProcessorCount, p.clockRate, p.warpSize);
    printf("WMMA f16->f32 (registers only): %.2f TFLOP/s\n", run_wmma(false, p.multiProcessorCount));
    printf("WMMA iu8->i32 (registers only): %.2f TOP/s\n", run_wmma(true, p.multiProcessorCount));
    hipblasLtHandle_t h;
    CB(hipblasLtCreate(&h));
    run_gemm(h, 4096, 4096, 4096, 10);
    for (int M : {32, 256, 2048}) run_gemm(h, M, 2304, 5120, 50);
    hipblasLtDestroy(h);
    return 0;
}
