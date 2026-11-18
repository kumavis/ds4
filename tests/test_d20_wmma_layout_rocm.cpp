// D20 step 3 prerequisite: the operand / result register layout of
// __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 on gfx1151, established
// empirically against a scalar reference (model-free).
//
// The kernel only moves registers: each lane loads its A and B operand
// vectors (4 x int32 = 16 int8 each) from per-lane arrays prepared on the
// host, runs one WMMA with a C input, and stores its 8 int32 results.  The
// host packs A and B under one layout hypothesis and checks the raw result
// registers against the scalar product D = A*B + C under each result-layout
// hypothesis.  Then the same check runs over full kernel-shaped problems:
// M x N x K with M, N not multiples of 16 (zero-padded partial tiles) and
// K = 2304 and 5120 accumulated in 16-wide steps across 32-element block
// boundaries, comparing every output element.
//
// Hypothesis A/B (operand): lane l holds row (A) or column (B) l % 16,
// elements k = 0..15 little-endian in the 4 dwords; lanes 16..31 hold the
// same as lanes 0..15.
// Result hypotheses: R1 element i of lane l is (row 2*i + l/16, col l%16);
// R2 element i of lane l is (row i + 8*(l/16), col l%16).
//
// Build: hipcc -O3 --offload-arch=gfx1151 test_d20_wmma_layout_rocm.cpp -o test_d20_wmma_layout_rocm
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %d\n", hipGetErrorString(e_), __LINE__); exit(2);} } while (0)
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i8v __attribute__((ext_vector_type(8)));

// One 16x16x16 step per launch, raw registers in and out.
__global__ void wmma_raw(const int32_t *a, const int32_t *b, const int32_t *c, int32_t *d) {
    const int l = threadIdx.x;
    i4v av, bv;
    i8v cv;
    for (int i = 0; i < 4; i++) { av[i] = a[l * 4 + i]; bv[i] = b[l * 4 + i]; }
    for (int i = 0; i < 8; i++) cv[i] = c[l * 8 + i];
    i8v dv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, cv, false);
    for (int i = 0; i < 8; i++) d[l * 8 + i] = dv[i];
}

// Tiled M x N x K product under the hypotheses (operands packed per lane by
// the kernel itself from row-major int8 A[M][K] and column-major B[N][K]),
// result written with the result-layout hypothesis R1.  One wave per 16x16
// output tile; K stepped 16 at a time; out-of-range rows/cols read as 0.
__global__ void wmma_tiled(const int8_t *A, const int8_t *B, int32_t *D, int M, int N, int K) {
    const int l = threadIdx.x;
    const int tm = blockIdx.x * 16, tn = blockIdx.y * 16;
    const int r = l % 16;
    i8v acc = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
        i4v av, bv;
        for (int q = 0; q < 4; q++) {
            uint32_t wa = 0, wb = 0;
            for (int j = 0; j < 4; j++) {
                const int k = k0 + q * 4 + j;
                const int8_t va = (tm + r < M && k < K) ? A[(size_t)(tm + r) * K + k] : 0;
                const int8_t vb = (tn + r < N && k < K) ? B[(size_t)(tn + r) * K + k] : 0;
                wa |= (uint32_t)(uint8_t)va << (8 * j);
                wb |= (uint32_t)(uint8_t)vb << (8 * j);
            }
            av[q] = (int)wa;
            bv[q] = (int)wb;
        }
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, av, true, bv, acc, false);
    }
    for (int i = 0; i < 8; i++) {
        const int m = tm + 2 * i + l / 16, n = tn + l % 16;
        if (m < M && n < N) D[(size_t)m * N + n] = acc[i];
    }
}

int main() {
    std::mt19937 rng(20261118);
    std::uniform_int_distribution<int> d8(-128, 127), dc(-100000, 100000);
    int bad = 0;
    // ---- part 1: one raw step, random A, B, C ----
    for (int trial = 0; trial < 64; trial++) {
        int8_t A[16][16], B[16][16];  // A[m][k], B[k][n]
        int32_t C[16][16];
        for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
            A[i][j] = (int8_t)d8(rng); B[i][j] = (int8_t)d8(rng); C[i][j] = dc(rng);
        }
        if (trial == 0) { A[0][0] = -128; B[0][0] = -128; }  // extremes
        std::vector<int32_t> ha(32 * 4), hb(32 * 4), hc(32 * 8), hd(32 * 8);
        for (int l = 0; l < 32; l++) {
            const int r = l % 16;
            for (int q = 0; q < 4; q++) {
                uint32_t wa = 0, wb = 0;
                for (int j = 0; j < 4; j++) {
                    wa |= (uint32_t)(uint8_t)A[r][q * 4 + j] << (8 * j);
                    wb |= (uint32_t)(uint8_t)B[q * 4 + j][r] << (8 * j);
                }
                ha[l * 4 + q] = (int32_t)wa;
                hb[l * 4 + q] = (int32_t)wb;
            }
            for (int i = 0; i < 8; i++) hc[l * 8 + i] = C[2 * i + l / 16][l % 16];  // C under R1
        }
        int32_t *da, *db, *dcp, *dd;
        CK(hipMalloc(&da, 512)); CK(hipMalloc(&db, 512)); CK(hipMalloc(&dcp, 1024)); CK(hipMalloc(&dd, 1024));
        CK(hipMemcpy(da, ha.data(), 512, hipMemcpyHostToDevice));
        CK(hipMemcpy(db, hb.data(), 512, hipMemcpyHostToDevice));
        CK(hipMemcpy(dcp, hc.data(), 1024, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(wmma_raw, dim3(1), dim3(32), 0, 0, da, db, dcp, dd);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(hd.data(), dd, 1024, hipMemcpyDeviceToHost));
        hipFree(da); hipFree(db); hipFree(dcp); hipFree(dd);
        int miss_r1 = 0, miss_r2 = 0;
        for (int l = 0; l < 32; l++) for (int i = 0; i < 8; i++) {
            const int n = l % 16;
            const int m1 = 2 * i + l / 16, m2 = i + 8 * (l / 16);
            int64_t ref1 = C[m1][n], ref2 = C[m2][n];
            for (int k = 0; k < 16; k++) { ref1 += (int)A[m1][k] * (int)B[k][n]; ref2 += (int)A[m2][k] * (int)B[k][n]; }
            if (hd[l * 8 + i] != (int32_t)ref1) miss_r1++;
            if (hd[l * 8 + i] != (int32_t)ref2) miss_r2++;
        }
        if (trial < 2 || miss_r1) printf("raw step trial %d: result-layout R1 mismatches %d/256, R2 %d/256\n", trial, miss_r1, miss_r2);
        if (miss_r1) bad = 1;
    }
    // ---- part 2: kernel-shaped problems, every element ----
    const int shapes[][3] = {{16, 16, 32}, {37, 21, 2304}, {64, 5, 5120}, {2304, 33, 5120}, {5120, 17, 2304},
                             {100, 1, 2304}, {48, 48, 5120}};
    for (auto &sh : shapes) {
        const int M = sh[0], N = sh[1], K = sh[2];
        std::vector<int8_t> A((size_t)M * K), B((size_t)N * K);
        for (auto &v : A) v = (int8_t)d8(rng);
        for (auto &v : B) v = (int8_t)d8(rng);
        int8_t *dA, *dB; int32_t *dD;
        CK(hipMalloc(&dA, A.size())); CK(hipMalloc(&dB, B.size())); CK(hipMalloc(&dD, (size_t)M * N * 4));
        CK(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
        CK(hipMemset(dD, 0x5a, (size_t)M * N * 4));
        hipLaunchKernelGGL(wmma_tiled, dim3((M + 15) / 16, (N + 15) / 16), dim3(32), 0, 0, dA, dB, dD, M, N, K);
        CK(hipDeviceSynchronize());
        std::vector<int32_t> D((size_t)M * N);
        CK(hipMemcpy(D.data(), dD, D.size() * 4, hipMemcpyDeviceToHost));
        hipFree(dA); hipFree(dB); hipFree(dD);
        long miss = 0;
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            int64_t ref = 0;
            for (int k = 0; k < K; k++) ref += (int)A[(size_t)m * K + k] * (int)B[(size_t)n * K + k];
            if (D[(size_t)m * N + n] != (int32_t)ref) miss++;
        }
        printf("tiled M=%d N=%d K=%d: %ld of %d elements differ\n", M, N, K, miss, M * N);
        if (miss) bad = 1;
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
