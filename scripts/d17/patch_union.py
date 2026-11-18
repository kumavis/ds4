"""D17 part B: union-loading layer sweep on ROCm (DS4_ROCM_V41_UNION_SWEEP=<max rows>, off by default).

For a prefill sweep of at most <max rows> rows (and at most 2048, one
sub-chunk), each layer's staging receives only the experts the chunk's rows
route to: the layer's whole-layer read is not started ahead; after the
router (ds41_route_batch, unchanged) the selected IDs are read back, the
union is formed, residents in it are copied from the cache and the rest is
read, then the routed MoE runs on the staging as before. Routing and the
experts' bytes are unchanged, so the output should be identical to the
whole-layer path. Whole-layer loading remains the path for every other sweep.

Files: rocm/ds4_rocm_runtime.cuh (cuda_stream_v41_prepare_layer gets a
wanted-expert mask; cuda_stream_v41_missing_jobs and cuda_stream_v41_plan_matches
honour the skip mask), rocm/ds4_rocm_current_api_compat.cuh and ds4_gpu.h
(ds4_gpu_dsv41_stream_prepare_layer_masked), ds4.c (ds41_union_sweep_rows,
ds41_union_after_route, ds41_stream_sweep_start_union, ds41_moe_batch,
ds41_graph_prefill_sweep, ds41_stream_layer_enter use).
"""
import re
root = "/data/ds4-d17b/"

def edit(path, pairs):
    s = open(root + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:90])
        s = s.replace(a, b)
    open(root + path, "w").write(s)

# ---------------------------------------------------------------- ds4_gpu.h
edit("ds4_gpu.h", [
("""typedef struct ds4_gpu_dsv41_stream_layer_plan {
    ds4_gpu_stream_expert_table table;
    uint64_t generation;
    uint64_t hits[6];
} ds4_gpu_dsv41_stream_layer_plan;
int ds4_gpu_dsv41_stream_prepare_layer(
        const ds4_gpu_stream_expert_table *table,
        ds4_gpu_dsv41_stream_layer_plan *plan);
""", """typedef struct ds4_gpu_dsv41_stream_layer_plan {
    ds4_gpu_stream_expert_table table;
    uint64_t generation;
    uint64_t hits[6];
    uint64_t skip[6];   /* D17: experts left out of a union-loaded layer */
} ds4_gpu_dsv41_stream_layer_plan;
int ds4_gpu_dsv41_stream_prepare_layer(
        const ds4_gpu_stream_expert_table *table,
        ds4_gpu_dsv41_stream_layer_plan *plan);
/* D17: like ds4_gpu_dsv41_stream_prepare_layer, but only the experts set in
 * want[6] (a 384-bit mask) are copied or read; the others' staging bytes are
 * left as they are and must not be used (the routed MoE uses only routed
 * experts). want == NULL is the whole layer. */
int ds4_gpu_dsv41_stream_prepare_layer_masked(
        const ds4_gpu_stream_expert_table *table,
        const uint64_t *want,
        ds4_gpu_dsv41_stream_layer_plan *plan);
"""),
])

# ------------------------------------------------- rocm compat wrapper
edit("rocm/ds4_rocm_current_api_compat.cuh", [
("""extern "C" int ds4_gpu_dsv41_stream_prepare_layer(
        const ds4_gpu_stream_expert_table *table,
        ds4_gpu_dsv41_stream_layer_plan *plan) {
    return cuda_stream_v41_prepare_layer(table, plan);
}
""", """extern "C" int ds4_gpu_dsv41_stream_prepare_layer(
        const ds4_gpu_stream_expert_table *table,
        ds4_gpu_dsv41_stream_layer_plan *plan) {
    return cuda_stream_v41_prepare_layer(table, plan);
}

extern "C" int ds4_gpu_dsv41_stream_prepare_layer_masked(
        const ds4_gpu_stream_expert_table *table,
        const uint64_t *want,
        ds4_gpu_dsv41_stream_layer_plan *plan) {
    return cuda_stream_v41_prepare_layer(table, plan, want);
}
"""),
])

# ------------------------------------------------- rocm runtime
edit("rocm/ds4_rocm_runtime.cuh", [
("""static int cuda_stream_v41_prepare_layer(const ds4_gpu_stream_expert_table *t,
                                         ds4_gpu_dsv41_stream_layer_plan *out) {""",
 """static int cuda_stream_v41_prepare_layer(const ds4_gpu_stream_expert_table *t,
                                         ds4_gpu_dsv41_stream_layer_plan *out,
                                         const uint64_t *want = NULL) {"""),
("""    for (uint32_t e = 0; ok && e < 384u; e++) {
        const int idx = cuda_stream_resident_find(t->model_map, t->layer, (int32_t)e,""",
 """    uint32_t skipped = 0;
    for (uint32_t e = 0; ok && e < 384u; e++) {
        if (want && !((want[e / 64u] >> (e % 64u)) & 1u)) {
            /* D17 union load: neither copied nor read. */
            plan.skip[e / 64u] |= UINT64_C(1) << (e % 64u);
            skipped++;
            continue;
        }
        const int idx = cuda_stream_resident_find(t->model_map, t->layer, (int32_t)e,"""),
("""        g_stream_v41_reuse_stats.missing_bytes += total - (uint64_t)hits * (total / 384u);""",
 """        g_stream_v41_reuse_stats.missing_bytes += total - (uint64_t)(hits + skipped) * (total / 384u);"""),
("""        memcmp(s.reuse_plan.hits, p->hits, sizeof(p->hits)) == 0;""",
 """        memcmp(s.reuse_plan.hits, p->hits, sizeof(p->hits)) == 0 &&
        memcmp(s.reuse_plan.skip, p->skip, sizeof(p->skip)) == 0;"""),
("""            if ((p->hits[e / 64u] >> (e % 64u)) & 1u) { e++; continue; }
            const uint32_t first = e++;
            while (e < 384u && !((p->hits[e / 64u] >> (e % 64u)) & 1u)) e++;""",
 """            if (((p->hits[e / 64u] | p->skip[e / 64u]) >> (e % 64u)) & 1u) { e++; continue; }
            const uint32_t first = e++;
            while (e < 384u && !(((p->hits[e / 64u] | p->skip[e / 64u]) >> (e % 64u)) & 1u)) e++;"""),
])

# ------------------------------------------------- ds4.c
edit("ds4.c", [
# sweep start variant and the after-route hook, placed before ds41_stream_layer_enter
("""static bool ds41_moe_batch(ds41_gpu_graph *g, const ds4_model *m,
                           const ds4_layer_weights *l, uint32_t il, uint32_t count,
                           bool shared_owner) {""",
 """#ifdef DS4_ROCM_BUILD
/* D17: union-loading sweep (DS4_ROCM_V41_UNION_SWEEP=<max rows>, off by
 * default). A sweep of at most that many rows (and one sub-chunk) does not read
 * whole layers ahead; after each layer's router the staging receives only the
 * routed union (residents copied, the rest read). Same routing, same expert
 * bytes, same kernels: the output is unchanged by design. */
static uint32_t ds41_union_sweep_rows(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_ROCM_V41_UNION_SWEEP");
        v = e && e[0] ? atoi(e) : 0;
        if (v < 0) v = 0;
        if (v > 2048) v = 2048;
    }
    return (uint32_t)v;
}

static struct {
    bool active;
    const ds4_model *m;
    uint64_t sweeps, layers, union_experts, read_experts;
    double load_ms;
} g_ds41_union;

static bool ds41_stream_sweep_start_union(void) {
    if (!ds4_gpu_stream_expert_cache_quiesce()) return false;
    if (ds4_gpu_stream_expert_cache_staging_lent() &&
        !ds4_gpu_stream_expert_cache_staging_reclaim()) {
        fprintf(stderr, "ds4: V4.1 could not take the lent staging back for a layer sweep\\n");
        return false;
    }
    return true;
}

/* Called by ds41_moe_batch right after ds41_route_batch. Outside a union
 * sweep it does nothing. */
static bool ds41_union_after_route(ds41_gpu_graph *g, const ds4_layer_weights *l,
                                   uint32_t il, uint32_t count) {
    if (!g_ds41_union.active || !g->streaming) return true;
    const double t0 = now_sec();
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) return false;
    static int32_t sel[2048u * 6u];
    if (!count || count > 2048u || DS4_N_EXPERT_USED != 6u ||
        !ds4_gpu_tensor_read(g->batch.selected, 0, sel, (uint64_t)count * 6u * sizeof(int32_t)))
        return false;
    uint64_t want[6] = {0};
    for (uint32_t i = 0; i < count * 6u; i++) {
        const int32_t id = sel[i];
        if (id < 0 || id >= 384) return false;
        want[id / 64] |= UINT64_C(1) << (id % 64);
    }
    ds4_gpu_stream_expert_table table;
    ds4_gpu_dsv41_stream_layer_plan plan;
    if (!ds41_stream_table(g_ds41_union.m, l, il, &table) ||
        !ds4_gpu_dsv41_stream_prepare_layer_masked(&table, want, &plan)) return false;
    if (!ds4_gpu_dsv41_stream_load_layer(&plan)) return false;
    uint32_t n_union = 0, n_hit = 0;
    for (int k = 0; k < 6; k++) {
        n_union += (uint32_t)__builtin_popcountll(want[k]);
        n_hit += (uint32_t)__builtin_popcountll(plan.hits[k]);
    }
    g_ds41_union.layers++;
    g_ds41_union.union_experts += n_union;
    g_ds41_union.read_experts += n_union - n_hit;
    g_ds41_union.load_ms += (now_sec() - t0) * 1000.0;
    return ds4_gpu_begin_commands() != 0;
}
#endif

static bool ds41_moe_batch(ds41_gpu_graph *g, const ds4_model *m,
                           const ds4_layer_weights *l, uint32_t il, uint32_t count,
                           bool shared_owner) {"""),
# sweep end: clear the flag, one log line per union sweep
("""#ifdef DS4_ROCM_BUILD
    if (g->streaming && !ds41_stream_sweep_finish(&prepare)) ok = false;
    if (g->streaming && !ok) (void)ds4_gpu_synchronize();""",
 """#ifdef DS4_ROCM_BUILD
    if (g->streaming && !ds41_stream_sweep_finish(&prepare)) ok = false;
    if (union_sweep) {
        g_ds41_union.active = false;
        g_ds41_union.sweeps++;
        fprintf(stderr, "ds4: V4.1 union sweep rows=%u layers=%llu union_experts=%llu "
                "read_experts=%llu load_ms=%.1f ok=%d\\n", total_count,
                (unsigned long long)(g_ds41_union.layers - union_layers0),
                (unsigned long long)(g_ds41_union.union_experts - union_x0),
                (unsigned long long)(g_ds41_union.read_experts - union_r0),
                g_ds41_union.load_ms - union_ms0, ok ? 1 : 0);
    }
    if (g->streaming && !ok) (void)ds4_gpu_synchronize();"""),
# hook in ds41_moe_batch
("""    return ds41_matmul_batch(b->route_logits, m, l->ffn_gate_inp, b->norm, count, false) &&
        ds41_route_batch(g, m, l, count) &&
        ((shared_owner && !balance_shared && g->tp_rank != (il & 1u)) ||""",
 """    return ds41_matmul_batch(b->route_logits, m, l->ffn_gate_inp, b->norm, count, false) &&
        ds41_route_batch(g, m, l, count) &&
#ifdef DS4_ROCM_BUILD
        ds41_union_after_route(g, l, il, count) &&
#endif
        ((shared_owner && !balance_shared && g->tp_rank != (il & 1u)) ||"""),
# sweep start and layer enter in ds41_graph_prefill_sweep
("""#ifdef DS4_ROCM_BUILD
    ds41_stream_layer_load prepare = {0};
    bool ok = !g->streaming || ds41_stream_sweep_start(&prepare, m, w);
#else""",
 """#ifdef DS4_ROCM_BUILD
    ds41_stream_layer_load prepare = {0};
    /* D17 union sweep: small single-sub-chunk sweeps only; never the encoder
     * or decoder-suffix paths, and never with tensor parallelism. */
    const uint32_t union_rows = ds41_union_sweep_rows();
    const bool union_sweep = g->streaming && union_rows && total_count <= union_rows &&
        !encoder_only && g->tp_world != 2u && ds41_encoder_chunk_cap(g, total_count) >= total_count;
    g_ds41_union.active = union_sweep;
    g_ds41_union.m = m;
    const uint64_t union_layers0 = g_ds41_union.layers, union_x0 = g_ds41_union.union_experts,
        union_r0 = g_ds41_union.read_experts;
    const double union_ms0 = g_ds41_union.load_ms;
    bool ok = !g->streaming ||
        (union_sweep ? ds41_stream_sweep_start_union() : ds41_stream_sweep_start(&prepare, m, w));
#else"""),
("""            ok = ds41_stream_layer_enter(&prepare, m, w, il, encoder_only ? 20u : DS4_N_LAYER);
#elif defined(__APPLE__)""",
 """            if (!union_sweep)
                ok = ds41_stream_layer_enter(&prepare, m, w, il, encoder_only ? 20u : DS4_N_LAYER);
#elif defined(__APPLE__)"""),
])
print("patched")
