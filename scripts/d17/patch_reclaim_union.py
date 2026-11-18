"""D17 part B item 4: reclaim only the staging slots a union load needs
(DS4_ROCM_STREAM_STAGING_RECLAIM_UNION=1, off by default; effective only with
DS4_ROCM_V41_UNION_SWEEP).

Runtime (rocm/ds4_rocm_runtime.cuh), next to D13's code, D13 functions unchanged:
  cuda_stream_staging_union_begin()   enter partial mode at a union sweep start:
        the staging stays lent; refused (caller does the full reclaim) when the
        D5 shelf is active or a load is pending
  cuda_stream_staging_reclaim_slots(parity, mask)   before a union layer load:
        evict each resident entry whose buffer is one of the union's slots in
        that parity half (cuda_stream_resident_evict_at, the per-slot eviction
        D13's full reclaim uses), take those slots off the free list for the
        rest of the sweep, then check that none of them is still held by an
        entry or on the free list (violation counter, must stay 0)
  cuda_stream_staging_union_end()     at the sweep end: return the held slots
        to the free list, mark both staging halves inactive (as a lend does)
  cuda_stream_v41_prepare_layer       accepts the lent state only in partial mode
Wrappers: ds4_gpu.h, rocm/ds4_rocm_current_api_compat.cuh.
ds4.c: ds41_stream_sweep_start_union, ds41_union_after_route, the union
sweep's end block in ds41_graph_prefill_sweep.
"""
root = "/data/ds4-d17b/"

def edit(path, pairs):
    s = open(root + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:90])
        s = s.replace(a, b)
    open(root + path, "w").write(s)

# ------------------------------------------------------------- runtime
edit("rocm/ds4_rocm_runtime.cuh", [
("""/* The engine joins its full-layer loader before entering this boundary. */
static int cuda_stream_expert_cache_quiesce(void) {""",
 """/* D17: partial reclaim for union-loading sweeps (see D17.md).  The staging
 * stays lent; before each union layer load only that layer's routed expert
 * positions in its parity half are taken back.  Runs on the graph thread at
 * the same quiesced points as D13's reclaim (no loader, no pending selected
 * load).  Uses only D13's per-slot eviction helper and the free list. */
static int g_stream_staging_partial;
static std::vector<char *> g_stream_staging_held;
static uint64_t g_stream_staging_partial_evicted, g_stream_staging_partial_held,
    g_stream_staging_partial_violations, g_stream_staging_partial_lent_resident0;

static int cuda_stream_staging_union_begin(void) {
    if (!g_stream_staging_lent || g_stream_staging_partial) return 0;
    if (g_d5_shelf_state == 1) return 0; /* D5 shelf may hold staging buffers: full reclaim */
    if (g_stream_selected_pending.active || g_stream_batch_selected_pending.active) return 0;
    if (!cuda_stream_resident_reclaim_wait("D17 partial staging reclaim")) return 0;
    uint64_t lent_resident = 0;
    for (size_t i = 0; i < g_stream_resident_experts.size(); i++) {
        const cuda_stream_resident_expert &e = g_stream_resident_experts[i];
        if (e.pooled && cuda_stream_staging_contains(e.base)) lent_resident++;
    }
    g_stream_staging_held.clear();
    g_stream_staging_partial_evicted = 0;
    g_stream_staging_partial_held = 0;
    g_stream_staging_partial_violations = 0;
    g_stream_staging_partial_lent_resident0 = lent_resident;
    g_stream_staging_partial = 1;
    return 1;
}

static int cuda_stream_staging_reclaim_slots(uint32_t parity, const uint64_t *mask) {
    if (!g_stream_staging_partial || parity > 1u || !mask) return 0;
    const cuda_stream_layer_expert_cache &c = g_stream_layer_expert_cache[parity];
    const uint64_t slot_bytes = g_stream_expert_slot_bytes;
    if (!c.base || !slot_bytes || c.n_total_expert != 384u) return 0;
    auto wanted = [&](const char *p) -> bool {
        if (p < c.base || p >= c.base + c.capacity) return false;
        const uint64_t off = (uint64_t)(p - c.base);
        if (off % slot_bytes) return false;
        const uint64_t e = off / slot_bytes;
        return e < 384u && ((mask[e / 64u] >> (e % 64u)) & 1u);
    };
    /* Evict entries living in a needed slot (walk down as D13 does). */
    for (size_t i = g_stream_resident_experts.size(); i-- > 0;) {
        const cuda_stream_resident_expert &e = g_stream_resident_experts[i];
        if (!e.pooled || !wanted(e.base)) continue;
        if (!cuda_stream_resident_evict_at(i)) return 0;
        g_stream_staging_partial_evicted++;
    }
    /* Hold the needed slots back from the free list until the sweep ends. */
    size_t kept = 0;
    for (size_t j = 0; j < g_stream_expert_free_slots.size(); j++) {
        char *slot = g_stream_expert_free_slots[j];
        if (wanted(slot)) {
            g_stream_staging_held.push_back(slot);
            g_stream_staging_partial_held++;
        } else {
            g_stream_expert_free_slots[kept++] = slot;
        }
    }
    g_stream_expert_free_slots.resize(kept);
    /* Check: no needed slot may still be an entry's buffer or allocatable. */
    for (size_t i = 0; i < g_stream_resident_experts.size(); i++)
        if (wanted(g_stream_resident_experts[i].base)) g_stream_staging_partial_violations++;
    for (size_t j = 0; j < g_stream_expert_free_slots.size(); j++)
        if (wanted(g_stream_expert_free_slots[j])) g_stream_staging_partial_violations++;
    return g_stream_staging_partial_violations == 0;
}

static int cuda_stream_staging_union_end(void) {
    if (!g_stream_staging_partial) return 1;
    /* Held slots are free again; a slot held for two layers is listed once
     * per hold, so put each address back only once. */
    std::sort(g_stream_staging_held.begin(), g_stream_staging_held.end());
    g_stream_staging_held.erase(std::unique(g_stream_staging_held.begin(), g_stream_staging_held.end()),
                                g_stream_staging_held.end());
    for (char *slot : g_stream_staging_held) g_stream_expert_free_slots.push_back(slot);
    g_stream_staging_held.clear();
    for (uint32_t i = 0; i < 2u; i++) g_stream_layer_expert_cache[i].active = 0;
    g_stream_staging_partial = 0;
    return 1;
}

/* The engine joins its full-layer loader before entering this boundary. */
static int cuda_stream_expert_cache_quiesce(void) {"""),
# prepare accepts the lent staging only in partial mode
("""                                         const uint64_t *want = NULL) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    uint64_t total = 0;
    if (!g_ssd_streaming_mode || g_stream_staging_lent ||""",
 """                                         const uint64_t *want = NULL) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    uint64_t total = 0;
    if (!g_ssd_streaming_mode || (g_stream_staging_lent && !(want && g_stream_staging_partial)) ||"""),
])

# ------------------------------------------------------------- compat wrappers
edit("rocm/ds4_rocm_current_api_compat.cuh", [
("""extern "C" int ds4_gpu_dsv41_stream_prepare_layer_masked(""",
 """extern "C" int ds4_gpu_stream_expert_cache_staging_union_begin(void) {
    return cuda_stream_staging_union_begin();
}

extern "C" int ds4_gpu_stream_expert_cache_staging_reclaim_slots(uint32_t parity, const uint64_t *mask) {
    return cuda_stream_staging_reclaim_slots(parity, mask);
}

extern "C" int ds4_gpu_stream_expert_cache_staging_union_end(uint64_t *stats) {
    if (stats) {
        stats[0] = g_stream_staging_partial_lent_resident0;
        stats[1] = g_stream_staging_partial_evicted;
        stats[2] = g_stream_staging_partial_held;
        stats[3] = g_stream_staging_partial_violations;
    }
    return cuda_stream_staging_union_end();
}

extern "C" int ds4_gpu_dsv41_stream_prepare_layer_masked("""),
])

edit("ds4_gpu.h", [
("""int ds4_gpu_dsv41_stream_prepare_layer_masked(""",
 """/* D17 partial staging reclaim for union sweeps (ROCm): begin (1 = partial
 * mode; 0 = refused, do the full reclaim), take back one layer's union slots,
 * end (stats: lent entries at begin, evicted, held, violations). */
int ds4_gpu_stream_expert_cache_staging_union_begin(void);
int ds4_gpu_stream_expert_cache_staging_reclaim_slots(uint32_t parity, const uint64_t *mask);
int ds4_gpu_stream_expert_cache_staging_union_end(uint64_t *stats);
int ds4_gpu_dsv41_stream_prepare_layer_masked("""),
])

# ------------------------------------------------------------- ds4.c
edit("ds4.c", [
("""    uint64_t sweeps, layers, union_experts, read_experts;
    double load_ms, sync_ms, sel_ms, prep_ms, read_ms;
} g_ds41_union;""",
 """    uint64_t sweeps, layers, union_experts, read_experts;
    double load_ms, sync_ms, sel_ms, prep_ms, read_ms;
    bool partial;   /* D17: staging still lent, union slots reclaimed per layer */
} g_ds41_union;

static bool ds41_union_partial_reclaim_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_ROCM_STREAM_STAGING_RECLAIM_UNION");
        v = e && e[0] && strcmp(e, "0") != 0;
    }
    return v == 1;
}"""),
("""static bool ds41_stream_sweep_start_union(void) {
    if (!ds4_gpu_stream_expert_cache_quiesce()) return false;
    if (ds4_gpu_stream_expert_cache_staging_lent() &&""",
 """static bool ds41_stream_sweep_start_union(void) {
    if (!ds4_gpu_stream_expert_cache_quiesce()) return false;
    g_ds41_union.partial = false;
    if (ds41_union_partial_reclaim_enabled() && ds4_gpu_stream_expert_cache_staging_lent() &&
        ds4_gpu_stream_expert_cache_staging_union_begin()) {
        g_ds41_union.partial = true;
        return true;
    }
    if (ds4_gpu_stream_expert_cache_staging_lent() &&"""),
("""    const double t_sel = now_sec();
    if (!ds41_stream_table(g_ds41_union.m, l, il, &table) ||""",
 """    const double t_sel = now_sec();
    if (g_ds41_union.partial && !ds4_gpu_stream_expert_cache_staging_reclaim_slots(il & 1u, want)) {
        fprintf(stderr, "ds4: V4.1 union sweep: partial staging reclaim failed at layer %u\\n", il);
        return false;
    }
    if (!ds41_stream_table(g_ds41_union.m, l, il, &table) ||"""),
("""    if (union_sweep) {
        g_ds41_union.active = false;
        g_ds41_union.sweeps++;""",
 """    if (union_sweep) {
        uint64_t pst[4] = {0, 0, 0, 0};
        if (g_ds41_union.partial) {
            if (!ds4_gpu_stream_expert_cache_staging_union_end(pst)) ok = false;
            fprintf(stderr, "ds4: V4.1 union sweep partial reclaim lent_resident=%llu evicted=%llu "
                    "held=%llu violations=%llu\\n", (unsigned long long)pst[0],
                    (unsigned long long)pst[1], (unsigned long long)pst[2], (unsigned long long)pst[3]);
            if (pst[3]) ok = false;
            g_ds41_union.partial = false;
        }
        g_ds41_union.active = false;
        g_ds41_union.sweeps++;"""),
])
print("patched")
