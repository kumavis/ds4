/* D20 step 3: the int8 WMMA MXFP4 kernels against the tile8 kernels,
 * bitwise, model-free, for one expert at many pair counts (partial 16-pair
 * tiles included), gate/up (mid, gate and up outputs) and down; also the
 * GPU time of both per launch, from which the crossover (rows per expert at
 * which the WMMA kernel is faster) is read.  PASS iff no bit differs. */
#include <stdio.h>
#include <stdint.h>
#include "ds4_gpu.h"
int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    const uint32_t counts[] = {1, 2, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 48, 64, 100, 128, 257, 512, 1000, 2040};
    int bad = 0;
    for (uint32_t kind = 0; kind < 2; kind++) {
        printf("%s\n| pairs | differing / compared | tile8 ms | wmma ms | wmma / tile8 |\n|---:|---:|---:|---:|---:|\n",
               kind ? "down (K 2304, 5120 rows)" : "gate/up (K 5120, 2304 rows)");
        for (uint32_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++)
            for (uint32_t seed = 1; seed <= 2; seed++) {
                uint64_t d = 0, t = 0;
                float a = 0, b = 0;
                if (!ds4_gpu_test_d20_wmma(kind, counts[ci], seed * 104729u + counts[ci], 5, &d, &t, &a, &b)) {
                    printf("| %u | run failed | | | |\n", counts[ci]);
                    bad = 1;
                    continue;
                }
                if (seed == 1 || d) printf("| %u | %llu / %llu | %.3f | %.3f | %.2f |\n", counts[ci],
                                           (unsigned long long)d, (unsigned long long)t, a, b, a > 0 ? b / a : 0.0);
                if (d) bad = 1;
            }
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
