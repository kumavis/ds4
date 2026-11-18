/* D22: V4.1 ROCm SSD-streaming admission arithmetic, shared by ds4.c
 * (ds41_memory_admit_for_host) and the model-free boundary test
 * tests/test_d22_admit.c.  Pure functions of the numbers passed in. */
#ifndef DS4_ADMIT_H
#define DS4_ADMIT_H
#include <stdint.h>

/* OS reserve the admission keeps: max(8 GiB, host / 16). */
static inline uint64_t ds4_admit_host_reserve(uint64_t host) {
    const uint64_t minimum = UINT64_C(8) << 30;
    const uint64_t reserve = host / 16u;
    return reserve > minimum ? reserve : minimum;
}

/* Budget: host MemAvailable (at engine open) minus the OS reserve, capped by
 * the GPU working set.  fixed: everything but the dynamic cache.  Returns 0
 * if not even min_experts dynamic slots fit; otherwise 1 and *max_dynamic =
 * the largest dynamic slot count admitted (a request above it is refused, or
 * fitted down when the caller allows fitting). */
static inline int ds4_admit_dynamic_limit(uint64_t host, uint64_t recommended,
                                          uint64_t fixed, uint64_t expert_bytes,
                                          uint32_t min_experts, uint64_t *budget_out,
                                          uint64_t *max_dynamic) {
    const uint64_t reserve = ds4_admit_host_reserve(host);
    uint64_t budget = host > reserve ? host - reserve : 0;
    if (budget > recommended) budget = recommended;
    if (budget_out) *budget_out = budget;
    if (fixed >= budget || (expert_bytes && (budget - fixed) / expert_bytes < min_experts)) return 0;
    if (max_dynamic) *max_dynamic = expert_bytes ? (budget - fixed) / expert_bytes : UINT64_MAX;
    return 1;
}
#endif
