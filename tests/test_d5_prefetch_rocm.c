/* Model-free check of the D5 speculative shelf (rocm/ds4_rocm_runtime.cuh).
 *
 * Writes a scratch "model" file of random bytes in the current directory
 * (O_DIRECT needs a real filesystem, not tmpfs), registers it as the model fd
 * and drives the single-token selected expert load (seed_selected = load +
 * finish) for a pseudo-random sequence of layers and expert sets, with a
 * small cache so that evictions happen.  Before each load it queues
 * speculative reads (ds4_gpu_d5_prefetch) for a mix of right and wrong
 * candidates of that layer, sometimes several layers ahead.
 *
 * Checks after every load:
 *   - every selected expert is resident and its slot holds exactly the file's
 *     bytes (gate | up | down), whether it was a hit, a read or adopted from
 *     the shelf (adoption swaps the shelf buffer into the slot pool; half way
 *     the cache is released, which must hand the shelf its own buffers back);
 *   - the resident set in LRU order equals a reference LRU that knows
 *     nothing about prefetch (hit -> touch; miss -> evict the least recently
 *     used entry that is not one of the current layer's experts, insert;
 *     then touch all selected in slot order).  So prefetch changes neither
 *     admission nor eviction nor recency.
 * And at the end: some experts were adopted (ready and in flight) and some
 * dropped.  Run with DS4_ROCM_D5_PREFETCH=1 (the Makefile target sets it). */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_gpu.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int ds4_gpu_test_d5_resident_bytes(const ds4_gpu_stream_expert_table *t, int32_t expert,
                                   void *out_host);
uint32_t ds4_gpu_test_d5_lru(uint32_t *layers, int32_t *experts, uint32_t max);
void ds4_gpu_test_d5_stats(uint64_t *issued, uint64_t *adopted_ready, uint64_t *adopted_inflight,
                           uint64_t *dropped, uint64_t *failed);

#define N_LAYER 3u
#define N_EXP 10u
#define N_SEL 3u
#define BUDGET 8u
#define GATE_BYTES (1048576u + 8192u + 4096u)  /* 2 chunks, unaligned tail */
#define DOWN_BYTES (786432u + 4096u)
#define STEPS 400u

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_rand(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

/* reference LRU: keys layer*N_EXP+expert, order[0] oldest */
static uint32_t ref[64];
static uint32_t ref_n;

static void ref_touch(uint32_t key) {
    uint32_t i = 0;
    while (i < ref_n && ref[i] != key) i++;
    if (i == ref_n) return;
    memmove(&ref[i], &ref[i + 1], (ref_n - i - 1u) * sizeof(ref[0]));
    ref[ref_n - 1u] = key;
}

static void ref_access(uint32_t layer, const int32_t *ids, uint32_t n) {
    for (uint32_t s = 0; s < n; s++) {
        const uint32_t key = layer * N_EXP + (uint32_t)ids[s];
        int found = 0;
        for (uint32_t i = 0; i < ref_n; i++) found |= ref[i] == key;
        if (found) {
            ref_touch(key);
            continue;
        }
        if (ref_n >= BUDGET) {
            for (uint32_t i = 0; i < ref_n; i++) {
                int cur = 0;
                for (uint32_t j = 0; j < n; j++) cur |= ref[i] == layer * N_EXP + (uint32_t)ids[j];
                if (cur) continue;
                memmove(&ref[i], &ref[i + 1], (ref_n - i - 1u) * sizeof(ref[0]));
                ref_n--;
                break;
            }
        }
        ref[ref_n++] = key;
    }
    for (uint32_t s = 0; s < n; s++) ref_touch(layer * N_EXP + (uint32_t)ids[s]);
}

int main(void) {
    setenv("DS4_ROCM_D5_PREFETCH", "1", 0);
    setenv("DS4_ROCM_D5_SHELF", "4", 0);
    setenv("DS4_ROCM_D5_INFLIGHT", "2", 0);
    const uint64_t layer_bytes = 2ull * N_EXP * GATE_BYTES + (uint64_t)N_EXP * DOWN_BYTES;
    const uint64_t file_bytes = 65536u + N_LAYER * layer_bytes + 65536u;
    char path[] = "./d5-test-model-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    unlink(path);
    {
        uint64_t *buf = malloc(1u << 20);
        for (uint64_t off = 0; off < file_bytes; off += 1u << 20) {
            const uint64_t n = file_bytes - off < (1u << 20) ? file_bytes - off : (1u << 20);
            for (uint64_t i = 0; i < (n + 7u) / 8u; i++) buf[i] = next_rand();
            if (pwrite(fd, buf, n, (off_t)off) != (ssize_t)n) {
                perror("pwrite");
                return 1;
            }
        }
        free(buf);
        fsync(fd);
    }
    const char *map = mmap(NULL, file_bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }
    ds4_gpu_set_model_fd(fd);
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_stream_expert_table tab[N_LAYER];
    for (uint32_t l = 0; l < N_LAYER; l++) {
        const uint64_t base = 65536u + l * layer_bytes;
        tab[l].model_map = map;
        tab[l].model_size = file_bytes;
        tab[l].layer = l;
        tab[l].n_total_expert = N_EXP;
        tab[l].gate_offset = base;
        tab[l].up_offset = base + (uint64_t)N_EXP * GATE_BYTES;
        tab[l].down_offset = base + 2ull * N_EXP * GATE_BYTES;
        tab[l].gate_expert_bytes = GATE_BYTES;
        tab[l].down_expert_bytes = DOWN_BYTES;
    }
    const size_t slot = 2u * GATE_BYTES + DOWN_BYTES;
    char *got = malloc(slot);
    int bad = 0;
    uint32_t layer = 0;
    for (uint32_t step = 0; step < STEPS && !bad; step++) {
        int32_t ids[N_SEL];
        for (uint32_t s = 0; s < N_SEL; s++) {
            int dup;
            do {
                ids[s] = (int32_t)(next_rand() % N_EXP);
                dup = 0;
                for (uint32_t j = 0; j < s; j++) dup |= ids[j] == ids[s];
            } while (dup);
        }
        /* speculative reads for this layer: some of the real ids, some wrong
         * ones; sometimes also for the layer after (dropped or kept). */
        if (step > 0) {
            int32_t cand[4];
            const uint32_t nc = 1u + (uint32_t)(next_rand() % 4u);
            for (uint32_t c = 0; c < nc; c++) {
                cand[c] = (next_rand() & 1u) ? ids[next_rand() % N_SEL] :
                                               (int32_t)(next_rand() % N_EXP);
            }
            (void)ds4_gpu_d5_prefetch(&tab[layer], cand, nc, 1u + (uint32_t)(next_rand() % 2u));
            if (next_rand() % 4u == 0) {
                const int32_t c2 = (int32_t)(next_rand() % N_EXP);
                (void)ds4_gpu_d5_prefetch(&tab[(layer + 1u) % N_LAYER], &c2, 1u, 1u);
            }
            if (next_rand() % 3u == 0) usleep((useconds_t)(next_rand() % 3000u));
        }
        if (step == STEPS / 2u) {
            /* the cache frees its slabs: the shelf must take its own
             * buffers back (adoption swaps buffers with the slot pool) */
            ds4_gpu_stream_expert_cache_release_resident();
            ref_n = 0;
        }
        if (!ds4_gpu_stream_expert_cache_seed_selected(&tab[layer], ids, N_SEL)) {
            fprintf(stderr, "step %u: selected load failed\n", step);
            bad = 1;
            break;
        }
        ref_access(layer, ids, N_SEL);
        for (uint32_t s = 0; s < N_SEL; s++) {
            const uint64_t e = (uint64_t)ids[s];
            if (!ds4_gpu_test_d5_resident_bytes(&tab[layer], ids[s], got)) {
                fprintf(stderr, "step %u: layer %u expert %d not resident\n", step, layer, ids[s]);
                bad = 1;
                break;
            }
            if (memcmp(got, map + tab[layer].gate_offset + e * GATE_BYTES, GATE_BYTES) ||
                memcmp(got + GATE_BYTES, map + tab[layer].up_offset + e * GATE_BYTES, GATE_BYTES) ||
                memcmp(got + 2u * GATE_BYTES, map + tab[layer].down_offset + e * DOWN_BYTES,
                       DOWN_BYTES)) {
                fprintf(stderr, "step %u: layer %u expert %d bytes differ\n", step, layer, ids[s]);
                bad = 1;
                break;
            }
        }
        uint32_t ll[64];
        int32_t ee[64];
        const uint32_t n = ds4_gpu_test_d5_lru(ll, ee, 64);
        if (n != ref_n) {
            fprintf(stderr, "step %u: %u resident, reference %u\n", step, n, ref_n);
            bad = 1;
        }
        for (uint32_t i = 0; !bad && i < n; i++) {
            if (ll[i] * N_EXP + (uint32_t)ee[i] != ref[i]) {
                fprintf(stderr, "step %u: LRU position %u differs\n", step, i);
                bad = 1;
            }
        }
        layer = (layer + 1u) % N_LAYER;
    }
    uint64_t issued = 0, ready = 0, inflight = 0, dropped = 0, failed = 0;
    ds4_gpu_test_d5_stats(&issued, &ready, &inflight, &dropped, &failed);
    printf("d5 prefetch test: steps=%u issued=%llu adopted_ready=%llu adopted_inflight=%llu "
           "dropped=%llu failed=%llu\n", STEPS, (unsigned long long)issued,
           (unsigned long long)ready, (unsigned long long)inflight,
           (unsigned long long)dropped, (unsigned long long)failed);
    if (!bad && (ready + inflight == 0 || dropped == 0 || failed != 0)) {
        fprintf(stderr, "expected adoptions and drops and no failures\n");
        bad = 1;
    }
    free(got);
    ds4_gpu_set_model_fd(-1);
    close(fd);
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
