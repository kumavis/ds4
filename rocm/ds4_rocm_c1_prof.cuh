/* C1 decode compute profiler, runtime side.  See ds4_c1_prof.h. */
#include "../ds4_c1_prof.h"

#define C1_MAX_MARKS 2048
#define C1_MAX_LAYERS 64

static const char *const g_c1_names[C1_N_LABELS] = {
    "start", "embed", "layer_pre", "hc_norm", "hc_proj", "hc_sinkhorn", "hc_pre", "attn_qa_kv", "attn_proj", "attn_rope_kv",
    "compressor", "indexer_topk", "attn_core", "attn_out", "hc_post",
    "router", "shared_expert", "sel_wait_host", "moe_prep_xq", "moe_gu_resident",
    "moe_read_wait", "moe_gu_missing", "moe_down", "moe_tail", "moe_all_resident", "moe_finish",
    "hc_expand", "layer_drain", "quiesce", "head", "logits_read", "step_tail",
};

static int g_c1_mode_v = -1;    /* 0 off, 1 coarse, 2 events, 3 sync */
static int g_c1_rt_active;
static int g_c1_skip = 4;
static int g_c1_layers_table;
static int g_c1_n;
static hipEvent_t g_c1_ev[C1_MAX_MARKS];
static int g_c1_ev_ready;
static double g_c1_host[C1_MAX_MARKS];
static uint64_t g_c1_wait[C1_MAX_MARKS];
static int16_t g_c1_lab[C1_MAX_MARKS];
static int16_t g_c1_lay[C1_MAX_MARKS];
static uint64_t g_c1_tokens_seen;
static uint64_t g_c1_tokens;
static double g_c1_span_ms;           /* graph step span, host */
static double g_c1_gpu_span_ms;       /* first..last event */
static double g_c1_outer_ms;          /* perf token minus graph step span */
static double g_c1_last_span_ms;
static int g_c1_last_counted;
static double g_c1_gpu[C1_N_LABELS], g_c1_hst[C1_N_LABELS], g_c1_wt[C1_N_LABELS];
static uint64_t g_c1_cnt[C1_N_LABELS];
static double g_c1_lgpu[C1_MAX_LAYERS][C1_N_LABELS], g_c1_lhst[C1_MAX_LAYERS][C1_N_LABELS];
static double g_c1_lwt[C1_MAX_LAYERS][C1_N_LABELS];
static int g_c1_report_registered;

static double c1_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

extern "C" int ds4_gpu_c1_mode(void) {
    if (g_c1_mode_v < 0) {
        const char *e = getenv("DS4_ROCM_C1_PROF");
        g_c1_mode_v = 0;
        if (e && e[0] && strcmp(e, "0") != 0) {
            if (!strcmp(e, "coarse") || !strcmp(e, "1")) g_c1_mode_v = 1;
            else if (!strcmp(e, "events")) g_c1_mode_v = 2;
            else if (!strcmp(e, "sync")) g_c1_mode_v = 3;
            else {
                fprintf(stderr, "ds4: C1: unknown DS4_ROCM_C1_PROF=%s (coarse, events, sync); off\n", e);
            }
        }
        const char *s = getenv("DS4_ROCM_C1_PROF_TOKENS");
        if (s && s[0]) g_c1_skip = atoi(s);
        const char *l = getenv("DS4_ROCM_C1_PROF_LAYERS");
        g_c1_layers_table = l && l[0] && strcmp(l, "0") != 0;
        if (g_c1_mode_v) fprintf(stderr, "ds4: C1 profiler mode %s, skipping %d tokens\n",
                                 g_c1_mode_v == 1 ? "coarse" : g_c1_mode_v == 2 ? "events" : "sync",
                                 g_c1_skip);
    }
    return g_c1_mode_v;
}

static int g_c1_cur_layer = -1;

static void c1_record(int label, int layer) {
    if (g_c1_n >= C1_MAX_MARKS) return;
    if (layer == -2) layer = g_c1_cur_layer;
    else if (layer >= 0) g_c1_cur_layer = layer;
    const int i = g_c1_n++;
    if (g_c1_mode_v == 3) (void)hipDeviceSynchronize();
    if (g_c1_mode_v >= 2) (void)hipEventRecord(g_c1_ev[i], 0);
    g_c1_host[i] = c1_now_ms();
    g_c1_wait[i] = g_stream_read_profile_wait_us;
    g_c1_lab[i] = (int16_t)label;
    g_c1_lay[i] = (int16_t)layer;
}

extern "C" void ds4_gpu_c1_token_begin(void) {
    if (ds4_gpu_c1_mode() == 0) return;
    if (!g_c1_report_registered) {
        g_c1_report_registered = 1;
        atexit(ds4_gpu_c1_report);
    }
    if (g_c1_mode_v >= 2 && !g_c1_ev_ready) {
        for (int i = 0; i < C1_MAX_MARKS; i++) (void)hipEventCreate(&g_c1_ev[i]);
        g_c1_ev_ready = 1;
    }
    g_c1_n = 0;
    g_c1_rt_active = 1;
    c1_record(C1_START, -1);
}

extern "C" void ds4_gpu_c1_mark(int label, int layer) {
    if (!g_c1_rt_active) return;
    c1_record(label, layer);
}

extern "C" void ds4_gpu_c1_mark_fine(int label, int layer) {
    if (!g_c1_rt_active || g_c1_mode_v == 1) return;
    c1_record(label, layer);
}

/* Runtime-side fine mark (inside routed launches): current layer. */
#define C1R(label) do { if (g_c1_rt_active && g_c1_mode_v != 1) c1_record((label), -2); } while (0)

extern "C" void ds4_gpu_c1_mark_routed_end(int layer) {
    if (!g_c1_rt_active || g_c1_mode_v == 1) return;
    const int prev = g_c1_n > 0 ? g_c1_lab[g_c1_n - 1] : -1;
    c1_record(prev == C1_MOE_DOWN ? C1_MOE_TAIL : C1_MOE_ALLRES, layer);
}

extern "C" void ds4_gpu_c1_token_end(void) {
    if (!g_c1_rt_active) return;
    c1_record(C1_STEP_TAIL, -1);
    g_c1_rt_active = 0;
    g_c1_last_counted = 0;
    const int n = g_c1_n;
    if (g_c1_mode_v >= 2) (void)hipEventSynchronize(g_c1_ev[n - 1]);
    /* Only decode steps count: they end with the logits readback.  Prefill
     * runs the same single-token step without logits on this path. */
    int has_logits = 0;
    for (int i = 0; i < n; i++) if (g_c1_lab[i] == C1_LOGITS_READ) has_logits = 1;
    if (!has_logits) return;
    g_c1_tokens_seen++;
    if (g_c1_tokens_seen <= (uint64_t)g_c1_skip || n >= C1_MAX_MARKS) return;
    g_c1_tokens++;
    g_c1_last_counted = 1;
    g_c1_last_span_ms = g_c1_host[n - 1] - g_c1_host[0];
    g_c1_span_ms += g_c1_last_span_ms;
    for (int i = 1; i < n; i++) {
        const int lab = g_c1_lab[i];
        const int lay = g_c1_lay[i];
        const double h = g_c1_host[i] - g_c1_host[i - 1];
        const double w = (double)(g_c1_wait[i] - g_c1_wait[i - 1]) / 1000.0;
        float gms = 0.0f;
        if (g_c1_mode_v >= 2 &&
            hipEventElapsedTime(&gms, g_c1_ev[i - 1], g_c1_ev[i]) != hipSuccess) {
            (void)hipGetLastError();
            gms = 0.0f;
        }
        g_c1_gpu[lab] += gms;
        g_c1_hst[lab] += h;
        g_c1_wt[lab] += w;
        g_c1_cnt[lab]++;
        if (lay >= 0 && lay < C1_MAX_LAYERS) {
            g_c1_lgpu[lay][lab] += gms;
            g_c1_lhst[lay][lab] += h;
            g_c1_lwt[lay][lab] += w;
        }
    }
    if (g_c1_mode_v >= 2) {
        float span = 0.0f;
        if (hipEventElapsedTime(&span, g_c1_ev[0], g_c1_ev[n - 1]) == hipSuccess)
            g_c1_gpu_span_ms += span;
        else (void)hipGetLastError();
    }
}

extern "C" void ds4_gpu_c1_note_token_ms(double total_ms) {
    if (!g_c1_mode_v || !g_c1_last_counted) return;
    g_c1_outer_ms += total_ms - g_c1_last_span_ms;
    g_c1_last_counted = 0;
}

extern "C" void ds4_gpu_c1_report(void) {
    if (!g_c1_mode_v || !g_c1_tokens) return;
    const double t = (double)g_c1_tokens;
    const char *mode = g_c1_mode_v == 1 ? "coarse" : g_c1_mode_v == 2 ? "events" : "sync";
    fprintf(stderr, "C1 report mode=%s tokens=%llu (first %d skipped)\n", mode,
            (unsigned long long)g_c1_tokens, g_c1_skip);
    fprintf(stderr, "C1 step_span_ms_per_tok %.3f gpu_span_ms_per_tok %.3f outside_step_ms_per_tok %.3f\n",
            g_c1_span_ms / t, g_c1_gpu_span_ms / t, g_c1_outer_ms / t);
    fprintf(stderr, "C1 %-16s %10s %10s %10s %8s\n", "segment", "gpu_ms/tok", "host_ms/tok",
            "rdwait_ms", "n/tok");
    double sg = 0, sh = 0, sw = 0;
    for (int l = 1; l < C1_N_LABELS; l++) {
        if (!g_c1_cnt[l]) continue;
        fprintf(stderr, "C1 %-16s %10.3f %10.3f %10.3f %8.2f\n", g_c1_names[l],
                g_c1_gpu[l] / t, g_c1_hst[l] / t, g_c1_wt[l] / t, (double)g_c1_cnt[l] / t);
        sg += g_c1_gpu[l]; sh += g_c1_hst[l]; sw += g_c1_wt[l];
    }
    fprintf(stderr, "C1 %-16s %10.3f %10.3f %10.3f\n", "TOTAL", sg / t, sh / t, sw / t);
    if (g_c1_layers_table) {
        fprintf(stderr, "C1L layer");
        for (int l = 1; l < C1_N_LABELS; l++) if (g_c1_cnt[l]) fprintf(stderr, " %s", g_c1_names[l]);
        fprintf(stderr, "   (gpu ms/tok; then host ms/tok; then read wait ms/tok)\n");
        for (int kind = 0; kind < 3; kind++) {
            for (int y = 0; y < C1_MAX_LAYERS; y++) {
                double any = 0;
                for (int l = 1; l < C1_N_LABELS; l++) any += g_c1_lhst[y][l];
                if (any == 0) continue;
                fprintf(stderr, "C1L%c %2d", kind == 0 ? 'g' : kind == 1 ? 'h' : 'w', y);
                for (int l = 1; l < C1_N_LABELS; l++) {
                    if (!g_c1_cnt[l]) continue;
                    const double v = kind == 0 ? g_c1_lgpu[y][l] : kind == 1 ? g_c1_lhst[y][l] : g_c1_lwt[y][l];
                    fprintf(stderr, " %.4f", v / t);
                }
                fprintf(stderr, "\n");
            }
        }
    }
    g_c1_tokens = 0;
}
