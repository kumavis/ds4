/* D20 step 1: bitwise comparison of the original MXFP4 tile8 down kernel
 * (unstaged at K = 2304) and its 9-block staged copy, model-free, over
 * row counts 1..300 (partial tiles included), several seeds, row_groups 1
 * and 2.  Prints the number of differing output elements; PASS iff 0. */
#include <stdio.h>
#include <stdint.h>
#include "ds4_gpu.h"
int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    const uint32_t counts[] = {1, 2, 3, 5, 6, 7, 8, 9, 13, 15, 16, 17, 24, 31, 32, 33, 64, 100, 257, 300};
    uint64_t diff_all = 0, total_all = 0;
    int bad = 0;
    for (uint32_t rg = 1; rg <= 2; rg++)
        for (uint32_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++)
            for (uint32_t seed = 1; seed <= 3; seed++) {
                uint64_t d = 0, t = 0;
                if (!ds4_gpu_test_d20_down_stage9(counts[ci], seed * 7919u + counts[ci], rg, &d, &t)) {
                    printf("count %u seed %u rg %u: run failed\n", counts[ci], seed, rg);
                    bad = 1;
                    continue;
                }
                if (d) printf("count %u seed %u rg %u: %llu of %llu differ\n", counts[ci], seed, rg,
                              (unsigned long long)d, (unsigned long long)t);
                diff_all += d;
                total_all += t;
            }
    printf("d20 down stage9: %llu of %llu output elements differ\n",
           (unsigned long long)diff_all, (unsigned long long)total_all);
    printf("%s\n", bad || diff_all ? "FAIL" : "PASS");
    return bad || diff_all ? 1 : 0;
}
