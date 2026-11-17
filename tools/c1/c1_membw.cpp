// C1 microbenchmark (model-free): GPU read bandwidth on this unified-memory
// box, plus launch / synchronise / readback costs.  Bounded runtime (< 1 min).
// Build: hipcc -O3 --offload-arch=gfx1151 -o c1_membw c1_membw.cpp
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void read_kernel(const uint4 *__restrict__ p, size_t n, uint32_t *out) {
    uint32_t acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;  // practically never; keeps loads live
}

// Unrolled variant: 4 independent loads in flight per thread.
__global__ void read_kernel_u4(const uint4 *__restrict__ p, size_t n, uint32_t *out) {
    uint32_t acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ b.y ^ c.z ^ d.w ^ a.w ^ b.x ^ c.y ^ d.z;
    }
    for (; i < n; i += stride) { uint4 v = p[i]; acc ^= v.x ^ v.w; }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

// Gather: read `nchunk` chunks of `chunk` uint4 at given offsets (expert-like).
__global__ void gather_kernel(const uint4 *__restrict__ p, const size_t *__restrict__ offs,
                              size_t chunk, uint32_t *out) {
    const uint4 *q = p + offs[blockIdx.y];
    uint32_t acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < chunk; i += stride) {
        uint4 v = q[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

__global__ void empty_kernel() {}
__global__ void tiny_kernel(float *x) { if (threadIdx.x == 0 && blockIdx.x == 0) x[0] += 1.0f; }

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s CUs %d clock %d kHz memclock %d kHz buswidth %d\n",
           prop.name, prop.multiProcessorCount, prop.clockRate, prop.memoryClockRate, prop.memoryBusWidth);
    uint32_t *out; CK(hipMalloc(&out, 64));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));

    const size_t sizes_mib[] = {64, 256, 1024, 4096};
    for (size_t si = 0; si < sizeof(sizes_mib) / sizeof(sizes_mib[0]); si++) {
        size_t bytes = sizes_mib[si] << 20;
        uint4 *buf; CK(hipMalloc(&buf, bytes));
        CK(hipMemset(buf, 1, bytes));
        size_t n = bytes / sizeof(uint4);
        int blocks_list[] = {80, 160, 320, 640, 1280, 2560};
        for (int k = 0; k < 2; k++) {
            double best = 0; int bestb = 0;
            for (int bi = 0; bi < 6; bi++) {
                int blocks = blocks_list[bi];
                for (int rep = 0; rep < 2; rep++) {  // warmup + measured
                    int iters = sizes_mib[si] <= 256 ? 20 : 5;
                    CK(hipEventRecord(e0));
                    for (int it = 0; it < iters; it++) {
                        if (k == 0) read_kernel<<<blocks, 256>>>(buf, n, out);
                        else read_kernel_u4<<<blocks, 256>>>(buf, n, out);
                    }
                    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                    double gbs = (double)bytes * iters / (ms * 1e-3) / 1e9;
                    if (rep == 1 && gbs > best) { best = gbs; bestb = blocks; }
                }
            }
            printf("read %s hipMalloc %5zu MiB: best %.1f GB/s (blocks %d x 256)\n",
                   k ? "u4  " : "plain", sizes_mib[si], best, bestb);
        }
        if (sizes_mib[si] == 4096) {
            // Expert-like gather: 6 random chunks of 18.8 MB, x 43 layers.
            const size_t chunk_b = 18800640ull;  // 3 x 6266880
            const size_t chunk = chunk_b / sizeof(uint4);
            const size_t nslots = bytes / chunk_b;
            std::vector<size_t> h(6 * 43);
            srand(1);
            for (auto &o : h) o = (size_t)(rand() % nslots) * chunk;
            size_t *d; CK(hipMalloc(&d, h.size() * sizeof(size_t)));
            CK(hipMemcpy(d, h.data(), h.size() * sizeof(size_t), hipMemcpyHostToDevice));
            for (int gx : {16, 40, 80, 160}) {
                for (int rep = 0; rep < 2; rep++) {
                    CK(hipEventRecord(e0));
                    for (int L = 0; L < 43; L++)
                        gather_kernel<<<dim3(gx, 6), 256>>>(buf, d + 6 * L, chunk, out);
                    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                    if (rep) printf("gather 43 layers x 6 x 18.8MB, grid %dx6: %.2f ms = %.1f GB/s\n",
                                    gx, ms, (double)chunk_b * 6 * 43 / (ms * 1e-3) / 1e9);
                }
            }
            CK(hipFree(d));
        }
        CK(hipFree(buf));
    }
    // Pinned host memory (hipHostMalloc), 1 GiB, read by the GPU.
    {
        size_t bytes = 1024ull << 20; uint4 *buf;
        CK(hipHostMalloc(&buf, bytes, hipHostMallocDefault));
        memset(buf, 1, bytes);
        size_t n = bytes / sizeof(uint4);
        double best = 0;
        for (int blocks : {320, 640, 1280}) for (int rep = 0; rep < 2; rep++) {
            CK(hipEventRecord(e0));
            for (int it = 0; it < 5; it++) read_kernel_u4<<<blocks, 256>>>(buf, n, out);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            double gbs = (double)bytes * 5 / (ms * 1e-3) / 1e9;
            if (rep && gbs > best) best = gbs;
        }
        printf("read u4 hipHostMalloc 1024 MiB: best %.1f GB/s\n", best);
        CK(hipHostFree(buf));
    }
    // Launch / sync costs.
    {
        float *x; CK(hipMalloc(&x, 4)); CK(hipMemset(x, 0, 4));
        for (int w = 0; w < 100; w++) empty_kernel<<<1, 64>>>();
        CK(hipDeviceSynchronize());
        const int N = 20000;
        double t0 = now_us();
        CK(hipEventRecord(e0));
        for (int i = 0; i < N; i++) empty_kernel<<<1, 64>>>();
        CK(hipEventRecord(e1));
        double t1 = now_us();
        CK(hipEventSynchronize(e1));
        double t2 = now_us();
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("empty kernel: host enqueue %.2f us/launch, GPU %.2f us/kernel back-to-back, wall %.2f us/kernel\n",
               (t1 - t0) / N, ms * 1000.0 / N, (t2 - t0) / N);
        t0 = now_us();
        CK(hipEventRecord(e0));
        for (int i = 0; i < N; i++) tiny_kernel<<<64, 256>>>(x);
        CK(hipEventRecord(e1));
        t1 = now_us();
        CK(hipEventSynchronize(e1));
        t2 = now_us();
        CK(hipEventElapsedTime(&ms, e0, e1));
        printf("tiny kernel 64x256: host enqueue %.2f us/launch, GPU %.2f us/kernel, wall %.2f us/kernel\n",
               (t1 - t0) / N, ms * 1000.0 / N, (t2 - t0) / N);
        const int M = 2000;
        t0 = now_us();
        for (int i = 0; i < M; i++) { empty_kernel<<<1, 64>>>(); CK(hipDeviceSynchronize()); }
        t1 = now_us();
        printf("launch + hipDeviceSynchronize round trip: %.2f us\n", (t1 - t0) / M);
        t0 = now_us();
        for (int i = 0; i < M; i++) CK(hipDeviceSynchronize());
        t1 = now_us();
        printf("hipDeviceSynchronize on idle device: %.2f us\n", (t1 - t0) / M);
        hipEvent_t ev; CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        t0 = now_us();
        for (int i = 0; i < M; i++) { empty_kernel<<<1, 64>>>(); CK(hipEventRecord(ev)); CK(hipEventSynchronize(ev)); }
        t1 = now_us();
        printf("launch + event record + event sync round trip: %.2f us\n", (t1 - t0) / M);
        hipEvent_t evt[64];
        for (int i = 0; i < 64; i++) CK(hipEventCreate(&evt[i]));
        t0 = now_us();
        for (int i = 0; i < M; i++) { tiny_kernel<<<64, 256>>>(x); CK(hipEventRecord(evt[i & 63])); }
        t1 = now_us();
        CK(hipDeviceSynchronize());
        printf("tiny launch + timing event record (host enqueue): %.2f us per pair\n", (t1 - t0) / M);
        int32_t *dsel; CK(hipMalloc(&dsel, 24)); int32_t hsel[6];
        t0 = now_us();
        for (int i = 0; i < M; i++) CK(hipMemcpy(hsel, dsel, 24, hipMemcpyDeviceToHost));
        t1 = now_us();
        printf("hipMemcpy D2H 24 B (sync): %.2f us\n", (t1 - t0) / M);
        void *dl; CK(hipMalloc(&dl, 129280 * 4)); std::vector<float> hl(129280);
        t0 = now_us();
        for (int i = 0; i < 200; i++) CK(hipMemcpy(hl.data(), dl, 129280 * 4, hipMemcpyDeviceToHost));
        t1 = now_us();
        printf("hipMemcpy D2H 517 KB logits to pageable (sync): %.2f us\n", (t1 - t0) / 200);
        float *pinned; CK(hipHostMalloc(&pinned, 129280 * 4, hipHostMallocDefault));
        t0 = now_us();
        for (int i = 0; i < 200; i++) CK(hipMemcpy(pinned, dl, 129280 * 4, hipMemcpyDeviceToHost));
        t1 = now_us();
        printf("hipMemcpy D2H 517 KB logits to pinned (sync): %.2f us\n", (t1 - t0) / 200);
        t0 = now_us();
        for (int i = 0; i < 200; i++) {
            float best = -1e30f; int bi = 0;
            for (int j = 0; j < 129280; j++) if (hl[j] > best) { best = hl[j]; bi = j; }
            hsel[0] = bi;
        }
        t1 = now_us();
        printf("CPU argmax over 129280 floats: %.2f us\n", (t1 - t0) / 200);
    }
    printf("C1 MEMBW DONE\n");
    return 0;
}
