/* ROCm indexer top-k for rows that no fixed-k path takes (top_k not 1, 512,
 * 1024 or 2048; V4.1 decode asks for top_k = n_comp while n_comp < 512).
 *
 * Those rows used to run through a single-lane insertion sort and now run
 * through the bitonic sort kernels (n_comp <= 1024: 1024 lanes, n_comp <=
 * 2048: the 2048-entry sort).  The attention gathers the selected rows in list
 * order, so the lists must be identical, order included, not just the same
 * set.  Every row is checked three ways:
 *
 *   - bitonic path against a CPU oracle (stable sort by score desc, index
 *     asc, -0 == +0) for every length 1..2048 and several top_k each;
 *   - the GPU insertion sort against the same oracle and against the bitonic
 *     output, for every length 1..1100 with top_k = n (the decode case), and
 *     on strides with top_k = n - 1, smaller top_k and lengths up to 2048.
 *     The insertion sort is O(n * top_k) on one lane, so full coverage would
 *     take minutes.
 *
 * Score patterns: random, heavy ties, all equal, ascending, descending,
 * signed zeros, a -inf/+inf mix, all -inf.  NaN is not covered: the two sorts
 * order it differently and upstream never produces it.
 *
 * DS4_TEST_TOPK_LEGACY_FULL=1 compares the insertion sort on every length and
 * top_k too (slow).
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_PATTERN 8u
#define MAX_N 2048u
#define LEGACY_DENSE_N 1100u

static uint32_t rng_state = 0x2545f491u;

static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static float pattern_value(uint32_t pattern, uint32_t i, uint32_t n) {
    switch (pattern) {
    case 0: return (float)((int32_t)(rng() >> 8) - (1 << 23)) / 4096.0f;
    case 1: return (float)(rng() % 4u);
    case 2: return 1.5f;
    case 3: return (float)i;
    case 4: return (float)(n - i);
    case 5: {
        const uint32_t r = rng() % 5u;
        return r == 0u ? 0.0f : r == 1u ? -0.0f : r == 2u ? 1.0f : r == 3u ? -1.0f : 0.0f;
    }
    case 6: {
        const uint32_t r = rng() % 8u;
        return r < 3u ? -INFINITY : r == 3u ? INFINITY : (float)(rng() % 16u) - 8.0f;
    }
    default: return -INFINITY;
    }
}

typedef struct {
    float v;
    uint32_t i;
} entry;

/* Score descending, index ascending; == treats -0 and +0 alike. */
static int entry_cmp(const void *pa, const void *pb) {
    const entry *a = (const entry *)pa, *b = (const entry *)pb;
    if (a->v > b->v) return -1;
    if (a->v < b->v) return 1;
    return a->i < b->i ? -1 : a->i > b->i ? 1 : 0;
}

static void oracle(uint32_t *out, const float *scores, uint32_t n, uint32_t top_k) {
    entry *e = (entry *)malloc((size_t)MAX_N * sizeof(*e));
    for (uint32_t row = 0; row < N_PATTERN; row++) {
        for (uint32_t i = 0; i < n; i++) e[i] = (entry){scores[(size_t)row * n + i], i};
        qsort(e, n, sizeof(*e), entry_cmp);
        for (uint32_t k = 0; k < top_k; k++) out[(size_t)row * top_k + k] = e[k].i;
    }
    free(e);
}

static ds4_gpu_tensor *g_scores, *g_selected;
static float *g_host_scores;
static uint32_t *g_want, *g_new, *g_old;

static bool run_gpu(uint32_t *out, uint32_t n, uint32_t top_k, int legacy) {
    ds4_gpu_test_set_indexer_topk_legacy(legacy);
    memset(out, 0xa5, (size_t)N_PATTERN * top_k * sizeof(*out));
    const bool ok = ds4_gpu_indexer_topk_tensor(g_selected, g_scores, n, N_PATTERN, top_k) &&
                    ds4_gpu_tensor_read(g_selected, 0, out,
                                        (uint64_t)N_PATTERN * top_k * sizeof(*out));
    ds4_gpu_test_set_indexer_topk_legacy(-1);
    return ok;
}

static bool same(const char *what, const uint32_t *got, const uint32_t *want,
                 uint32_t n, uint32_t top_k) {
    for (uint32_t row = 0; row < N_PATTERN; row++) {
        for (uint32_t k = 0; k < top_k; k++) {
            const size_t at = (size_t)row * top_k + k;
            if (got[at] != want[at]) {
                fprintf(stderr,
                        "indexer topk ROCm: %s differs at n=%u top_k=%u pattern=%u slot=%u: "
                        "got %u want %u\n", what, n, top_k, row, k, got[at], want[at]);
                return false;
            }
        }
    }
    return true;
}

int main(void) {
    const char *full_env = getenv("DS4_TEST_TOPK_LEGACY_FULL");
    const bool legacy_full = full_env && full_env[0] && strcmp(full_env, "0") != 0;
    const size_t cells = (size_t)N_PATTERN * MAX_N;
    uint64_t cases = 0, legacy_cases = 0;
    int ok = 0;

    g_host_scores = (float *)malloc(cells * sizeof(float));
    g_want = (uint32_t *)malloc(cells * sizeof(uint32_t));
    g_new = (uint32_t *)malloc(cells * sizeof(uint32_t));
    g_old = (uint32_t *)malloc(cells * sizeof(uint32_t));
    if (!g_host_scores || !g_want || !g_new || !g_old) {
        fprintf(stderr, "indexer topk ROCm: host allocation failed\n");
        return 1;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "indexer topk ROCm: ds4_gpu_init failed\n");
        return 1;
    }
    g_scores = ds4_gpu_tensor_alloc(cells * sizeof(float));
    g_selected = ds4_gpu_tensor_alloc(cells * sizeof(uint32_t));
    if (!g_scores || !g_selected) {
        fprintf(stderr, "indexer topk ROCm: tensor allocation failed\n");
        goto done;
    }

    const time_t t0 = time(NULL);
    for (uint32_t n = 1; n <= MAX_N; n++) {
        for (uint32_t p = 0; p < N_PATTERN; p++) {
            for (uint32_t i = 0; i < n; i++) g_host_scores[(size_t)p * n + i] = pattern_value(p, i, n);
        }
        if (!ds4_gpu_tensor_write(g_scores, 0, g_host_scores, (uint64_t)N_PATTERN * n * sizeof(float))) {
            fprintf(stderr, "indexer topk ROCm: score upload failed at n=%u\n", n);
            goto done;
        }
        const uint32_t ks[] = {n, n - 1u, n / 2u + 1u, n / 3u, 7u, 2u, 1u};
        for (size_t ki = 0; ki < sizeof(ks) / sizeof(*ks); ki++) {
            const uint32_t top_k = ks[ki];
            if (top_k == 0u || top_k > n) continue;
            bool dup = false;
            for (size_t kj = 0; kj < ki; kj++) dup |= ks[kj] == top_k;
            if (dup) continue;

            oracle(g_want, g_host_scores, n, top_k);
            if (!run_gpu(g_new, n, top_k, 0)) {
                fprintf(stderr, "indexer topk ROCm: launch failed n=%u top_k=%u\n", n, top_k);
                goto done;
            }
            if (!same("bitonic vs oracle", g_new, g_want, n, top_k)) goto done;
            cases++;

            /* ki 0 is top_k = n (decode), ki 1 is top_k = n - 1. */
            const bool legacy = legacy_full ||
                (n <= LEGACY_DENSE_N &&
                 (ki == 0u || (ki == 1u && n % 4u == 0u) || n % 16u == 0u)) ||
                (ki < 2u && (n % 128u <= 1u || n >= MAX_N - 1u));
            if (!legacy) continue;
            if (!run_gpu(g_old, n, top_k, 1)) {
                fprintf(stderr, "indexer topk ROCm: legacy launch failed n=%u top_k=%u\n", n, top_k);
                goto done;
            }
            if (!same("insertion sort vs oracle", g_old, g_want, n, top_k)) goto done;
            if (!same("bitonic vs insertion sort", g_new, g_old, n, top_k)) goto done;
            legacy_cases++;
        }
    }
    ok = 1;
    printf("indexer topk ROCm: %llu (n, top_k) cases x %u patterns match the oracle, "
           "%llu also match the insertion sort (%lds): PASS\n",
           (unsigned long long)cases, N_PATTERN, (unsigned long long)legacy_cases,
           (long)(time(NULL) - t0));

done:
    ds4_gpu_tensor_free(g_selected);
    ds4_gpu_tensor_free(g_scores);
    ds4_gpu_cleanup();
    free(g_old);
    free(g_new);
    free(g_want);
    free(g_host_scores);
    return ok ? 0 : 1;
}
