"""D17 part B: split the union load time (sync / readback / prepare / read)."""
p = "/data/ds4-d17b/ds4.c"
s = open(p).read()

def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)

rep("""    uint64_t sweeps, layers, union_experts, read_experts;
    double load_ms;
} g_ds41_union;""", """    uint64_t sweeps, layers, union_experts, read_experts;
    double load_ms, sync_ms, sel_ms, prep_ms, read_ms;
} g_ds41_union;""")
rep("""    const double t0 = now_sec();
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) return false;
    static int32_t sel[2048u * 6u];""", """    const double t0 = now_sec();
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) return false;
    const double t_sync = now_sec();   /* attention and router GPU work done */
    static int32_t sel[2048u * 6u];""")
rep("""    ds4_gpu_stream_expert_table table;
    ds4_gpu_dsv41_stream_layer_plan plan;
    if (!ds41_stream_table(g_ds41_union.m, l, il, &table) ||
        !ds4_gpu_dsv41_stream_prepare_layer_masked(&table, want, &plan)) return false;
    if (!ds4_gpu_dsv41_stream_load_layer(&plan)) return false;""", """    ds4_gpu_stream_expert_table table;
    ds4_gpu_dsv41_stream_layer_plan plan;
    const double t_sel = now_sec();
    if (!ds41_stream_table(g_ds41_union.m, l, il, &table) ||
        !ds4_gpu_dsv41_stream_prepare_layer_masked(&table, want, &plan)) return false;
    const double t_prep = now_sec();
    if (!ds4_gpu_dsv41_stream_load_layer(&plan)) return false;
    const double t_read = now_sec();
    g_ds41_union.sync_ms += (t_sync - t0) * 1000.0;
    g_ds41_union.sel_ms += (t_sel - t_sync) * 1000.0;
    g_ds41_union.prep_ms += (t_prep - t_sel) * 1000.0;
    g_ds41_union.read_ms += (t_read - t_prep) * 1000.0;""")
rep("""    const double union_ms0 = g_ds41_union.load_ms;""", """    const double union_ms0 = g_ds41_union.load_ms, union_sync0 = g_ds41_union.sync_ms,
        union_sel0 = g_ds41_union.sel_ms, union_prep0 = g_ds41_union.prep_ms,
        union_read0 = g_ds41_union.read_ms;""")
rep("""                "read_experts=%llu load_ms=%.1f ok=%d\\n", total_count,""", """                "read_experts=%llu load_ms=%.1f sync_ms=%.1f readback_ms=%.1f prepare_ms=%.1f read_ms=%.1f ok=%d\\n", total_count,""")
rep("""                g_ds41_union.load_ms - union_ms0, ok ? 1 : 0);""", """                g_ds41_union.load_ms - union_ms0, g_ds41_union.sync_ms - union_sync0,
                g_ds41_union.sel_ms - union_sel0, g_ds41_union.prep_ms - union_prep0,
                g_ds41_union.read_ms - union_read0, ok ? 1 : 0);""")
open(p, "w").write(s)
print("patched")
