/* D22: model-free boundary test of the V4.1 ROCm SSD admission arithmetic
 * (ds4_admit.h, used by ds41_memory_admit_for_host).  The host MemAvailable
 * is injected, so the result does not depend on the machine.  For plain and
 * DSpark plans at several MemAvailable values: the largest admitted dynamic
 * cache N is admitted and N+1 is refused; 6000 MXFP4 slots are refused.
 * Build: cc -O2 -I. -o tests/test_d22_admit tests/test_d22_admit.c */
#include <stdio.h>
#include "ds4_admit.h"

static const double GiB = 1073741824.0;
static int fails;

static void check(const char *name, int cond) {
    printf("%s %s\n", cond ? "ok  " : "FAIL", name);
    if (!cond) fails++;
}

static int admitted(uint64_t host, uint64_t rec, uint64_t fixed, uint64_t expert, uint64_t n) {
    uint64_t max = 0;
    if (!ds4_admit_dynamic_limit(host, rec, fixed, expert, 6, NULL, &max)) return 0;
    return n <= max; /* ds41_memory_admit_for_host refuses n > max without fitting */
}

int main(void) {
    /* MXFP4 plan pieces as printed by ds4 (ctx 4096, one session). */
    const uint64_t expert = 18800640u;                      /* 17.93 MiB */
    const uint64_t non_routed = (uint64_t)(9.37 * GiB), graph = (uint64_t)(1.53 * GiB);
    const uint64_t staging = 768u * expert, runtime = UINT64_C(2) << 30;
    const uint64_t support = (uint64_t)(8.04 * GiB);
    const uint64_t rec = UINT64_C(120) << 30;               /* GTT working set */
    const double hosts[] = {100.0, 112.42, 122.0, 122.5};
    for (int dsp = 0; dsp < 2; dsp++) {
        const uint64_t fixed = non_routed + graph + staging + runtime + (dsp ? support : 0);
        for (unsigned i = 0; i < sizeof(hosts) / sizeof(hosts[0]); i++) {
            const uint64_t host = (uint64_t)(hosts[i] * GiB);
            uint64_t max = 0, budget = 0;
            char name[160];
            const int ok = ds4_admit_dynamic_limit(host, rec, fixed, expert, 6, &budget, &max);
            snprintf(name, sizeof(name), "%s host %.2f GiB: limit N=%llu (budget %.2f GiB, fixed %.2f GiB)",
                     dsp ? "dspark" : "plain ", hosts[i], (unsigned long long)max,
                     budget / GiB, fixed / GiB);
            check(name, ok);
            snprintf(name, sizeof(name), "%s host %.2f GiB: admit N", dsp ? "dspark" : "plain ", hosts[i]);
            check(name, admitted(host, rec, fixed, expert, max));
            snprintf(name, sizeof(name), "%s host %.2f GiB: refuse N+1", dsp ? "dspark" : "plain ", hosts[i]);
            check(name, !admitted(host, rec, fixed, expert, max + 1));
            snprintf(name, sizeof(name), "%s host %.2f GiB: refuse 6000", dsp ? "dspark" : "plain ", hosts[i]);
            check(name, !admitted(host, rec, fixed, expert, 6000));
        }
        /* Too little memory for even six dynamic slots: refused outright. */
        check(dsp ? "dspark tiny host refused" : "plain  tiny host refused",
              !ds4_admit_dynamic_limit((uint64_t)(20.0 * GiB), rec, fixed, expert, 6, NULL, NULL));
    }
    printf("%s (%d failing)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
