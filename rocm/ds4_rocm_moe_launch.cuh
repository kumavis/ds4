/* ---- D20: routed-MoE launch profiler (DS4_ROCM_D20_PROF=1, off by default) ----
 * Measurement only.  Inside every routed_moe_launch call with at least 2 rows,
 * a timing hipEvent is recorded after each kernel launch / memset of this file
 * (at its existing error check) and the host clock is read at the same point.
 * At the call's end the events are synchronized; each launch is charged the
 * GPU time from the previous mark to its own (so idle GPU time before it, e.g.
 * a host wait, is charged to it too) and the host time since the previous
 * mark.  Totals per row bucket and launch label print at exit.  Off: every
 * check is cuda_ok plus one branch. */
#include <chrono>
enum { D20_MAX_MARKS = 160, D20_MAX_LABELS = 96, D20_BUCKETS = 6 };
static int g_d20_prof = -1;
static int g_d20_active;
static cudaEvent_t g_d20_ev[D20_MAX_MARKS];
static const char *g_d20_lab[D20_MAX_MARKS];
static double g_d20_host[D20_MAX_MARKS];
static int g_d20_n;
static const char *g_d20_names[D20_MAX_LABELS];
static double g_d20_gpu_ms[D20_BUCKETS][D20_MAX_LABELS], g_d20_host_ms[D20_BUCKETS][D20_MAX_LABELS];
static uint64_t g_d20_cnt[D20_BUCKETS][D20_MAX_LABELS];
static double g_d20_call_gpu[D20_BUCKETS], g_d20_call_host[D20_BUCKETS], g_d20_call_sync[D20_BUCKETS];
static uint64_t g_d20_calls[D20_BUCKETS], g_d20_rows[D20_BUCKETS], g_d20_overflow;
static const char *g_d20_bucket_name[D20_BUCKETS] = {"2-6", "7-255", "256-511", "512-1023", "1024-2047", "2048+"};

static double d20_now_ms(void) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int d20_bucket(uint32_t n) {
    return n <= 6u ? 0 : n < 256u ? 1 : n < 512u ? 2 : n < 1024u ? 3 : n < 2048u ? 4 : 5;
}
static void d20_report(void) {
    for (int b = 0; b < D20_BUCKETS; b++) {
        if (!g_d20_calls[b]) continue;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "D20 moe prof rows %s: calls=%llu mean_rows=%.0f gpu=%.1f ms/call host_to_end=%.1f ms/call sync_wait=%.1f ms/call\n",
                g_d20_bucket_name[b], (unsigned long long)g_d20_calls[b],
                (double)g_d20_rows[b] / (double)g_d20_calls[b], g_d20_call_gpu[b] / g_d20_calls[b],
                g_d20_call_host[b] / g_d20_calls[b], g_d20_call_sync[b] / g_d20_calls[b]);
        for (int l = 0; l < D20_MAX_LABELS && g_d20_names[l]; l++) {
            if (!g_d20_cnt[b][l]) continue;
            fprintf(stderr, DS4_GPU_LOG_PREFIX "D20 moe prof rows %s:   %-48s n=%llu gpu=%.2f ms/call host=%.2f ms/call\n",
                    g_d20_bucket_name[b], g_d20_names[l], (unsigned long long)g_d20_cnt[b][l],
                    g_d20_gpu_ms[b][l] / g_d20_calls[b], g_d20_host_ms[b][l] / g_d20_calls[b]);
        }
    }
    if (g_d20_overflow) fprintf(stderr, DS4_GPU_LOG_PREFIX "D20 moe prof: %llu calls exceeded %d marks\n",
                                (unsigned long long)g_d20_overflow, D20_MAX_MARKS);
}
static int d20_on(void) {
    if (g_d20_prof < 0) {
        const char *e = getenv("DS4_ROCM_D20_PROF");
        g_d20_prof = e && e[0] && strcmp(e, "0") ? 1 : 0;
        if (g_d20_prof) {
            for (int i = 0; i < D20_MAX_MARKS; i++)
                if (cudaEventCreate(&g_d20_ev[i]) != cudaSuccess) { g_d20_prof = 0; break; }
            if (g_d20_prof) atexit(d20_report);
        }
    }
    return g_d20_prof;
}
static void d20_mark(const char *label) {
    if (!g_d20_active) return;
    if (g_d20_n >= D20_MAX_MARKS) { g_d20_active = 2; return; }
    (void)cudaEventRecord(g_d20_ev[g_d20_n], 0);
    g_d20_lab[g_d20_n] = label;
    g_d20_host[g_d20_n] = d20_now_ms();
    g_d20_n++;
}
static int d20_ok(cudaError_t err, const char *what) {
    const int ok = cuda_ok(err, what);
    if (g_d20_active == 1) d20_mark(what);
    return ok;
}
struct D20ProfScope {
    uint32_t rows;
    bool on;
    explicit D20ProfScope(uint32_t n) : rows(n), on(false) {
        if (n < 2u || g_d20_active || !d20_on()) return;
        on = true;
        g_d20_active = 1;
        g_d20_n = 0;
        d20_mark("(call start)");
    }
    ~D20ProfScope() {
        if (!on) return;
        const bool overflow = g_d20_active == 2;
        g_d20_active = 1;
        if (!overflow) d20_mark("(call end)");
        g_d20_active = 0;
        if (overflow) { g_d20_overflow++; return; }
        const double t_end = d20_now_ms();
        (void)cudaEventSynchronize(g_d20_ev[g_d20_n - 1]);
        const int b = d20_bucket(rows);
        g_d20_calls[b]++;
        g_d20_rows[b] += rows;
        g_d20_call_host[b] += t_end - g_d20_host[0];
        g_d20_call_sync[b] += d20_now_ms() - t_end;
        float total = 0.0f;
        (void)cudaEventElapsedTime(&total, g_d20_ev[0], g_d20_ev[g_d20_n - 1]);
        g_d20_call_gpu[b] += total;
        for (int i = 1; i < g_d20_n; i++) {
            float ms = 0.0f;
            (void)cudaEventElapsedTime(&ms, g_d20_ev[i - 1], g_d20_ev[i]);
            int l = 0;
            for (; l < D20_MAX_LABELS && g_d20_names[l]; l++)
                if (!strcmp(g_d20_names[l], g_d20_lab[i])) break;
            if (l == D20_MAX_LABELS) continue;
            if (!g_d20_names[l]) g_d20_names[l] = g_d20_lab[i];
            g_d20_gpu_ms[b][l] += ms;
            g_d20_host_ms[b][l] += g_d20_host[i] - g_d20_host[i - 1];
            g_d20_cnt[b][l]++;
        }
    }
};


static int routed_moe_u64_add_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int routed_moe_align256_checked(uint64_t v, uint64_t *out) {
    if (!out || v > UINT64_MAX - 255ull) return 0;
    *out = (v + 255ull) & ~255ull;
    return 1;
}

/* Exact worst-case number of nonempty expert tiles. Give every live expert
 * its first tile, then charge complete tile_m groups beyond those first
 * assignments. The former ceil(routes/tile_m)+expert_count bound launched
 * hundreds of empty blocks for a 12--36 route speculative verifier. */
static uint32_t routed_moe_expert_tile_capacity(
        uint32_t pair_count,
        uint32_t tile_m,
        uint32_t bucket_count) {
    const uint32_t first = pair_count < bucket_count ? pair_count : bucket_count;
    return first + (pair_count - first) / tile_m;
}

enum {
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START = 0,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
    DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT
};

typedef struct {
    uint64_t calls;
    uint64_t split_calls;
    uint64_t q8_gateup_calls;
    uint64_t q8_down_calls;
    double finish_missing_ms;
    double gate_resident_ms;
    double gate_missing_ms;
    double gate_full_ms;
    double mid_quant_ms;
    double down_ms;
} ds4_rocm_moe_decode_profile_stats;

typedef struct {
    int gate_resident;
    int gate_missing;
    int gate_full;
    int mid_quant;
    int down;
} ds4_rocm_moe_decode_profile_record;

static ds4_rocm_moe_decode_profile_stats g_moe_decode_profile_stats;
static cudaEvent_t g_moe_decode_profile_events[DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT];
static int g_moe_decode_profile_enabled = -1;
static int g_moe_decode_profile_registered = 0;
static int g_moe_decode_profile_events_ready = 0;

static void routed_moe_decode_profile_print(void) {
    const ds4_rocm_moe_decode_profile_stats *p =
        &g_moe_decode_profile_stats;
    if (p->calls == 0u) return;
    const double calls = (double)p->calls;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "Q2 decode MoE profile calls=%llu "
            "split=%llu q8_gateup=%llu q8_down=%llu "
            "finish_missing=%.3f ms gate_resident=%.3f ms "
            "gate_missing=%.3f ms gate_full=%.3f ms "
            "mid_quant=%.3f ms down=%.3f ms avg_sum=%.3f ms\n",
            (unsigned long long)p->calls,
            (unsigned long long)p->split_calls,
            (unsigned long long)p->q8_gateup_calls,
            (unsigned long long)p->q8_down_calls,
            p->finish_missing_ms,
            p->gate_resident_ms,
            p->gate_missing_ms,
            p->gate_full_ms,
            p->mid_quant_ms,
            p->down_ms,
            (p->finish_missing_ms + p->gate_resident_ms +
             p->gate_missing_ms + p->gate_full_ms + p->mid_quant_ms +
             p->down_ms) / calls);
}

static int routed_moe_decode_profile_enabled(void) {
    if (g_moe_decode_profile_enabled < 0) {
        const char *env = getenv("DS4_ROCM_MOE_DECODE_PROFILE");
        g_moe_decode_profile_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        if (g_moe_decode_profile_enabled && !g_moe_decode_profile_registered) {
            atexit(routed_moe_decode_profile_print);
            g_moe_decode_profile_registered = 1;
        }
    }
    return g_moe_decode_profile_enabled;
}

static int routed_moe_decode_profile_ensure_events(void) {
    if (g_moe_decode_profile_events_ready) return 1;
    for (uint32_t i = 0; i < DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT; i++) {
        cudaError_t err = cudaEventCreate(&g_moe_decode_profile_events[i]);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "Q2 decode MoE profile event create failed: %s\n",
                    cudaGetErrorString(err));
            for (uint32_t j = 0; j < i; j++) {
                (void)cudaEventDestroy(g_moe_decode_profile_events[j]);
                g_moe_decode_profile_events[j] = NULL;
            }
            return 0;
        }
    }
    g_moe_decode_profile_events_ready = 1;
    return 1;
}

static int routed_moe_decode_profile_record_event(uint32_t ev, const char *what) {
    if (ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    return cuda_ok(cudaEventRecord(g_moe_decode_profile_events[ev], 0), what);
}

static int routed_moe_decode_profile_add_event_ms(uint32_t start_ev,
                                                  uint32_t end_ev,
                                                  double *accum,
                                                  const char *what) {
    if (!accum ||
        start_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        end_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    if (!cuda_ok(cudaEventSynchronize(g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    float ms = 0.0f;
    if (!cuda_ok(cudaEventElapsedTime(&ms,
                                      g_moe_decode_profile_events[start_ev],
                                      g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    *accum += (double)ms;
    return 1;
}

static int routed_moe_decode_profile_collect(
        const ds4_rocm_moe_decode_profile_record *rec) {
    if (!rec) return 1;
    if (rec->gate_resident &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                &g_moe_decode_profile_stats.gate_resident_ms,
                "Q2 decode MoE profile resident gate/up")) {
        return 0;
    }
    if (rec->gate_missing &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                &g_moe_decode_profile_stats.gate_missing_ms,
                "Q2 decode MoE profile missing gate/up")) {
        return 0;
    }
    if (rec->gate_full &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                &g_moe_decode_profile_stats.gate_full_ms,
                "Q2 decode MoE profile gate/up")) {
        return 0;
    }
    if (rec->mid_quant &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                &g_moe_decode_profile_stats.mid_quant_ms,
                "Q2 decode MoE profile mid quant")) {
        return 0;
    }
    if (rec->down &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                &g_moe_decode_profile_stats.down_ms,
                "Q2 decode MoE profile down")) {
        return 0;
    }
    return 1;
}

/* Mixed IQ2_XXS-gate/Q2_K-down models already compute routed mid activations
 * as float.  Reuse the newer Q2_K expert-batch/WMMA down kernels instead of
 * re-quantizing mid to Q8_K and taking the older qwarp down path.  This keeps
 * the CyberNeurova all-Q2 path untouched while giving the standard IQ2 mix the
 * same fast Q2 down projection used by q2k_path. */
static int routed_moe_q2_float_down_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const ds4_gpu_tensor *mid,
        const half *mid_h_hot,
        int hot_mid_f16,
        const char *down_w,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *sorted_pairs,
        uint32_t *hot_experts_dev,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t hot_capacity = DS4_ROCM_MAX_N_EXPERT) {
    if (!out || !down || !mid || !down_w || !counts || !offsets || !sorted_pairs ||
        n_tokens == 0u || n_total_expert == 0u || n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        (expert_mid_dim % CUDA_QK_K) != 0u || expert_mid_dim == 0u || out_dim == 0u ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }

    uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (n_tokens >= 8u &&
        !cuda_ok(cudaMemcpy(h_counts, counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "routed_moe iq2/q2 float-down counts copy")) {
        return 0;
    }

    const uint32_t down_tile = n_tokens <= 8u ? 2u : 4u;
    const uint32_t down_rpb = 32u;
    const uint32_t down_threads = down_rpb * 32u;
    const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
    const int use_f16_down = (out_dim & 1u) == 0u;
    half *down_h = use_f16_down ? (half *)down->ptr : NULL;

    uint32_t hot_count = 0u;
    uint32_t hot_max = 0u;
    const uint32_t hot_threshold = 8u;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const int use_wmma_hot = n_tokens >= hot_threshold && hot_experts_dev &&
        !g_quality_mode &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;
#else
    const int use_wmma_hot = 0;
#endif
    uint32_t h_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (use_wmma_hot) {
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t c = h_counts[e];
            if (c >= hot_threshold) {
                h_hot[hot_count++] = e;
                if (c > hot_max) hot_max = c;
            }
        }
    }

    const uint32_t original_hot_count = hot_count;
    bool halo_compact = use_wmma_hot && g_halo_prefill_model && halo_prefill_scope(n_tokens) &&
        !g_glm_model && !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
        n_total_expert==256u && n_expert==6u && expert_mid_dim==2048u && out_dim==4096u &&
        down_expert_bytes==2752512u && down_row_bytes==672u && hot_mid_f16 && mid_h_hot &&
        use_f16_down && (((uintptr_t)mid_h_hot&15u)==0) && (((uintptr_t)down_w&3u)==0);
    bool v41_compact = use_wmma_hot &&
        g_deepseek41_model && !g_glm_model && !g_quality_mode && !g_ssd_streaming_mode &&
        ds4_rocm_is_gfx1151() && n_tokens==2048u &&
        n_total_expert==384u && n_expert==6u && expert_mid_dim==2304u && out_dim==5120u &&
        down_expert_bytes==3870720u && down_row_bytes==756u && hot_mid_f16 && mid_h_hot &&
        use_f16_down && (((uintptr_t)mid_h_hot&15u)==0) && (((uintptr_t)down_w&3u)==0);
    uint32_t h_tiles[1024], tile_count=0, prepared_count=0;
    if(v41_compact) {
        for(uint32_t e=0;e<384u;e++)if(h_counts[e]>=hot_threshold) {
            for(uint32_t j=0;j<h_counts[e];j+=64u) {
                if(tile_count==1024u){v41_compact=false;break;}
                h_tiles[tile_count++]=e|((j/64u)<<9u);
            }
            if(!v41_compact)break;
        }
        if(tile_count>hot_capacity)v41_compact=false;
    }
    if(halo_compact) {
        for(uint32_t e=0;e<256u;e++)if(h_counts[e]>=hot_threshold) {
            if(h_counts[e]>=512u)++prepared_count;
            else for(uint32_t j=0;j<h_counts[e];j+=64u) {
                if(tile_count==640u){halo_compact=false;break;}
                h_tiles[tile_count++]=e|((j/64u)<<8u);
            }
            if(!halo_compact)break;
        }
        if(tile_count>hot_capacity || (prepared_count && !ds4_rocm_halo_q2_prepare()))halo_compact=false;
        if(halo_compact) {
            hot_count=0;hot_max=0;
            for(uint32_t e=0;e<256u;e++)if(h_counts[e]>=hot_threshold && h_counts[e]<512u) {
                h_hot[hot_count++]=e;if(h_counts[e]>hot_max)hot_max=h_counts[e];
            }
        }
    }
    const uint32_t scalar_max = original_hot_count != 0u ? hot_threshold : 0u;
    const bool compact_active = n_tokens <= 8u && hot_experts_dev != NULL;
    uint32_t *active_count = compact_active ? hot_experts_dev : NULL;
    uint32_t *active_experts = compact_active ? hot_experts_dev + 1u : NULL;
    if (compact_active) {
        moe_build_active_experts_kernel<<<1u, 1u>>>(
                active_count, active_experts, counts, n_total_expert);
        if (!d20_ok(cudaGetLastError(),
                     "routed_moe iq2/q2 active expert list launch")) {
            return 0;
        }
    }
    const dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb,
                         compact_active ? n_tokens * n_expert : n_total_expert,
                         1u);
    if (use_f16_down) {
        if (down_tile == 2u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<2,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else if (down_tile == 4u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else if (down_tile == 8u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        }
    } else if (down_tile == 2u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<2><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else if (down_tile == 4u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else if (down_tile == 8u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else {
        moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    }
    if (!d20_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down scalar launch")) return 0;
    if (halo_compact && prepared_count && !ds4_rocm_halo_q2_prepared(down_h,down_w,
            mid_h_hot,offsets,sorted_pairs,h_counts,n_tokens)) return 0;
    if (hot_count != 0u &&
        !cuda_ok(cudaMemcpy(hot_experts_dev, (halo_compact || v41_compact) ? h_tiles : h_hot,
                           ((halo_compact || v41_compact) ? tile_count : hot_count) * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "routed_moe iq2/q2 float-down hot copy")) {
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_wmma_hot && hot_count != 0u) {
        if (v41_compact && tile_count) {
            const dim3 grid(160u,1u,tile_count);
#define V41_Q2_ARGS nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,hot_experts_dev,tile_count,2304u,5120u,3870720u,756u,6u,0u
            // Only V4.1's staged nine-bit-descriptor kernel is covered here.
            // Four half slabs need 4096 bytes; 32 packed rows need another 2688.
            // The float epilogue reuses the first slab after the reader barrier.
            const char *lds_env = getenv("DS4_ROCM_V41_Q2_LDS_COMPACT");
            const unsigned lds_bytes = ds4_rocm_is_gfx1151() &&
                (!lds_env || !strcmp(lds_env, "1")) ? 6784u : 9856u;
            halo_q2::q2_wstage64_n2_kernel<4,16,16,16,true,true,false,true,9><<<grid,128,lds_bytes>>>(V41_Q2_ARGS);
            const char *lds_trace = getenv("DS4_ROCM_V41_Q2_LDS_TRACE");
            if (lds_trace && !strcmp(lds_trace, "1"))
                fprintf(stderr,"V41_Q2_LDS rows=%u tiles=%u bytes=%u\n",n_tokens,tile_count,lds_bytes);
#undef V41_Q2_ARGS
            if(!d20_ok(cudaGetLastError(), "V4.1 Q2 prefill launch"))return 0;
            static bool logged=false;
            if(!logged){logged=true;fprintf(stderr,"ds4: ROCm V4.1 Q2 reused panels rows=%u tiles=%u\n",n_tokens,tile_count);}
        } else if (halo_compact && tile_count) {
            const dim3 grid(128u,1u,tile_count);
            if(n_tokens==2048u) {
                halo_q2::q2_wstage64_n2_kernel<4,16,16,16,true,true,false,true><<<grid,128,9856>>>(
                    nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,
                    hot_experts_dev,tile_count,2048u,4096u,2752512u,672u,6u,0u);
            } else {
                halo_q2::q2_direct_x_n2_kernel<4,16,16,16,true,true,false,true><<<grid,128,9856>>>(
                    nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,
                    hot_experts_dev,tile_count,2048u,4096u,2752512u,672u,6u,0u);
            }
            if(!d20_ok(cudaGetLastError(), "Halo Q2 direct-X compact launch"))return 0;
        } else
        {

        constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
        const int no_n2 = 0;
        const uint32_t wmma_mtiles = 4u;
        if (!no_n2) {
            if (wmma_mtiles == 4u) {
                constexpr uint32_t mt = 4u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt = 16u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else {
                constexpr uint32_t mt = 8u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            }
        } else if (wmma_mtiles == 16u) {
            constexpr uint32_t mt = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else if (wmma_mtiles == 4u) {
            constexpr uint32_t mt = 4u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else {
            constexpr uint32_t mt = 8u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        }
        }
        if (!d20_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down wmma launch")) return 0;
    }
#endif

    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (use_f16_down && (out_dim & 1u) == 0u) {
        const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
        moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else if (use_f16_down) {
        moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else {
        moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    }
    return d20_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down sum launch");
}

typedef struct {
    int q4k_path;
    int iq2_path;
    int iq2_iq2_path;
    int q2k_path;
    int mxfp4_path;
    uint64_t gate_bytes;
    uint64_t down_bytes;
} routed_moe_launch_plan;

static int routed_moe_build_plan(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        routed_moe_launch_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0u ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        !cuda_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    plan->q4k_path = (gate_type == 12u && down_type == 12u);
    plan->iq2_path = (gate_type == 16u && down_type == 10u);
    plan->iq2_iq2_path = (gate_type == 16u && down_type == 16u);
    plan->q2k_path = (gate_type == 10u && down_type == 10u);
    plan->mxfp4_path = (gate_type == 39u && down_type == 39u);
    if (!plan->q4k_path && !plan->iq2_path &&
        !plan->iq2_iq2_path && !plan->q2k_path &&
        !plan->mxfp4_path) return 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !cuda_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return 0;
    }
    return 1;
}

/* D1: under SSD streaming, a few MXFP4 rows (DSpark verification, at most
 * one seed plus five drafts) take the batch union/pointer-table path
 * (cuda_stream_batch_selected_prepare) and run the one-token decode kernels
 * on the resident slots in place.  Before this the launcher had no MXFP4
 * kernel that could address union slots for several rows and failed with
 * "missing compact selected experts".  DS4_ROCM_DSPARK_SSD=0 restores that
 * refusal (and the engine-level refusal in ds4.c). */
#define DS4_ROCM_MXFP4_STREAM_MAX_ROWS 6u
static int routed_moe_mxfp4_batch_stream_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_ROCM_DSPARK_SSD");
        cached = !(env && env[0] == '0');
    }
    return cached;
}

/* Down projection for the MXFP4 stream rows: 0 = one launch of the
 * <Batch=true, Indirect=true> instantiation over (out_dim/8, rows); 1 = one
 * launch per row of decode's own <false, true> instantiation on per-row views.
 * Both read the same source; tests/test_mxfp4_stream_rows_rocm.c checks each
 * bitwise against the one-token decode launches.
 * DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW=0|1 overrides the default. */
#ifndef DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW_DEFAULT
#define DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW_DEFAULT 0
#endif
static int routed_moe_mxfp4_stream_down_per_row(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW");
        cached = env && env[0] ? (env[0] != '0') :
            DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW_DEFAULT;
    }
    return cached;
}

/* Routed MXFP4 experts for n_tokens rows whose experts are addressed through
 * a slot pointer table (selected holds table indices).  xq must already hold
 * the Q8_K rows of x.  Every (row, expert) pair runs the arithmetic of the
 * one-token decode path (routed_moe_launch, n_tokens == 1, zero-copy):
 *   gate/up  moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true>, one block row
 *            per pair; a pair's lanes depend only on its own row and expert;
 *   mid      q8_K_quantize_kernel per pair block;
 *   down     moe_down_mxfp4_sum6_qwarp32_kernel over the row's six slots in
 *            route order (decode's fold), see the down mode above.
 * Launch only; the caller owns the stream-cache ordering (upload wait before,
 * reuse event after). */
static int routed_moe_mxfp4_ptrs_rows_tail_launch(
        float *out,
        const float *mid,
        const char *const *down_ptrs,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        int down_per_row);

static int routed_moe_mxfp4_ptrs_rows_launch(
        float *out,
        float *gate,
        float *up,
        float *mid,
        const char *const *gate_ptrs,
        const char *const *up_ptrs,
        const char *const *down_ptrs,
        const cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        float clamp,
        int down_per_row) {
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t pair_count = n_tokens * n_expert;
    dim3 gate_grid((expert_mid_dim + 7u) / 8u, pair_count, 1);
    moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true><<<gate_grid, 256>>>(
            gate,
            up,
            mid,
            (const char *)gate_ptrs,
            (const char *)up_ptrs,
            xq,
            selected,
            weights,
            0,
            gate_row_bytes,
            xq_blocks,
            expert_mid_dim,
            n_expert,
            0u,
            clamp);
    if (!d20_ok(cudaGetLastError(), "routed_moe streaming rows mxfp4 gate/up launch")) return 0;
    return routed_moe_mxfp4_ptrs_rows_tail_launch(out, mid, down_ptrs, midq, selected,
                                                  down_row_bytes, expert_mid_dim, out_dim,
                                                  n_expert, n_tokens, down_per_row);
}

/* D2: gate/up of the listed (row, expert) pairs only, with the pair-list
 * twin of the kernel above (bitwise the same per pair).  n may be 0. */
static int routed_moe_mxfp4_gate_up_pairs_launch(
        float *gate,
        float *up,
        float *mid,
        const char *const *gate_ptrs,
        const char *const *up_ptrs,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_gate_up,
        float clamp,
        const uint8_t *pairs,
        uint32_t n) {
    if (n == 0u) return 1;
    moe_mxfp4_pair_list list;
    memset(&list, 0, sizeof(list));
    if (n > sizeof(list.pair)) return 0;
    for (uint32_t i = 0; i < n; i++) list.pair[i] = pairs[i];
    dim3 grid((expert_mid_dim + 7u) / 8u, n, 1);
    moe_gate_up_mid_decode_mxfp4_qwarp32_pairs_kernel<<<grid, 256>>>(
            gate, up, mid,
            (const char *)gate_ptrs,
            (const char *)up_ptrs,
            xq, selected, weights,
            gate_row_bytes,
            expert_in_dim / CUDA_QK_K,
            expert_mid_dim,
            n_expert,
            write_gate_up,
            clamp,
            list);
    return d20_ok(cudaGetLastError(), "routed_moe mxfp4 pair-list gate/up launch");
}

/* D2 (single-token decode overlap): the routed MXFP4 experts of one token
 * whose missing route slots are still being read.  xq holds Q8_K(x); stream
 * 0 already waits for the pointer-table upload.  Order of work:
 *   1. gate/up of the resident slots (pair-list kernel), enqueued at once;
 *   2. for each batch of missing slots whose gate and up jobs finished (read
 *      and uploaded, host-synchronised by the read worker), their gate/up;
 *   3. finish the pending load: every job, down included, uploaded;
 *   4. mid quantize over all six pairs and the fused down
 *      moe_down_mxfp4_sum6_qwarp32_kernel<false, true> in route order: the
 *      exact launches of the sequential zero-copy path.
 * Each pair's mid is bitwise what the full gate/up launch writes, and steps 4
 * are unchanged, so the output equals the sequential path bit for bit.  The
 * caller records the reuse event afterwards, also on failure. */
static int routed_moe_mxfp4_decode_overlap_launch(
        float *out,
        float *gate,
        float *up,
        float *mid,
        const char *const *gate_ptrs,
        const char *const *up_ptrs,
        const char *const *down_ptrs,
        const cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t write_gate_up,
        float clamp,
        uint32_t resident_mask,
        uint32_t missing_mask) {
    uint8_t list[DS4_ROCM_N_EXPERT_USED];
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_expert; i++) {
        if (resident_mask & (1u << i)) list[n++] = (uint8_t)i;
    }
    int ok = routed_moe_mxfp4_gate_up_pairs_launch(
            gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
            gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
            write_gate_up, clamp, list, n);
    C1R(C1_MOE_GU_RES);
    uint32_t remaining = missing_mask;
    while (ok && remaining) {
        const uint32_t ready = cuda_stream_selected_overlap_wait(remaining) & remaining;
        C1R(C1_MOE_READ_WAIT);
        if (!ready) break;
        n = 0;
        for (uint32_t i = 0; i < n_expert; i++) {
            if (ready & (1u << i)) list[n++] = (uint8_t)i;
        }
        ok = routed_moe_mxfp4_gate_up_pairs_launch(
                gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
                gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
                write_gate_up, clamp, list, n);
        C1R(C1_MOE_GU_MISS);
        remaining &= ~ready;
    }
    /* All reads and uploads (down too) complete before any later launch. */
    if (!cuda_stream_selected_finish_pending_missing(0)) return 0;
    C1R(C1_MOE_READ_WAIT);
    if (!ok) return 0;
    if (remaining) {
        n = 0;
        for (uint32_t i = 0; i < n_expert; i++) {
            if (remaining & (1u << i)) list[n++] = (uint8_t)i;
        }
        if (!routed_moe_mxfp4_gate_up_pairs_launch(
                gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
                gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
                write_gate_up, clamp, list, n)) return 0;
        C1R(C1_MOE_GU_MISS);
    }
    const int tail_ok = routed_moe_mxfp4_ptrs_rows_tail_launch(out, mid, down_ptrs, midq, selected,
                                                  down_row_bytes, expert_mid_dim, out_dim,
                                                  n_expert, 1u, 1);
    C1R(C1_MOE_DOWN);
    return tail_ok;
}

/* D2 (verify overlap): routed_moe_mxfp4_ptrs_rows_launch for a pending union
 * load whose missing experts are still being read: gate/up of every pair
 * whose union slot is resident first, then the pairs of each missing union
 * slot as its gate and up land, then (all jobs finished) the unchanged tail.
 * Bitwise equal to the D1 launch on the same inputs.  The caller records the
 * batch reuse event afterwards, also on failure. */
static int routed_moe_mxfp4_rows_overlap_launch(
        float *out,
        float *gate,
        float *up,
        float *mid,
        const char *const *gate_ptrs,
        const char *const *up_ptrs,
        const char *const *down_ptrs,
        const cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        float clamp,
        int down_per_row) {
    const uint32_t pair_count = n_tokens * n_expert;
    const cuda_stream_batch_selected_pending &p = g_stream_batch_selected_pending;
    if (!p.active || !p.host_pairs_valid || p.n_pairs != pair_count ||
        pair_count > DS4_ROCM_D2_MAX_PAIRS) {
        (void)cuda_stream_batch_selected_finish_pending_missing();
        return 0;
    }
    /* Copy the host maps: finishing the load clears the pending state. */
    uint8_t pair_unique[DS4_ROCM_D2_MAX_PAIRS];
    uint8_t unique_missing[DS4_ROCM_D2_MAX_PAIRS];
    const uint32_t n_unique = p.n_unique;
    memcpy(pair_unique, p.pair_unique, sizeof(pair_unique));
    memcpy(unique_missing, p.unique_missing, sizeof(unique_missing));
    uint8_t list[DS4_ROCM_D2_MAX_PAIRS];
    uint32_t n = 0;
    uint64_t remaining = 0;
    for (uint32_t u = 0; u < n_unique; u++) {
        if (unique_missing[u]) remaining |= 1ull << u;
    }
    for (uint32_t i = 0; i < pair_count; i++) {
        if (!unique_missing[pair_unique[i]]) list[n++] = (uint8_t)i;
    }
    int ok = routed_moe_mxfp4_gate_up_pairs_launch(
            gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
            gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
            0u, clamp, list, n);
    while (ok && remaining) {
        const uint64_t ready = cuda_stream_batch_selected_overlap_wait(remaining) & remaining;
        if (!ready) break;
        n = 0;
        for (uint32_t i = 0; i < pair_count; i++) {
            if (ready & (1ull << pair_unique[i])) list[n++] = (uint8_t)i;
        }
        ok = routed_moe_mxfp4_gate_up_pairs_launch(
                gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
                gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
                0u, clamp, list, n);
        remaining &= ~ready;
    }
    /* All reads and uploads (down too) complete before any later launch. */
    if (!cuda_stream_batch_selected_finish_pending_missing()) return 0;
    if (!ok) return 0;
    if (remaining) {
        n = 0;
        for (uint32_t i = 0; i < pair_count; i++) {
            if (remaining & (1ull << pair_unique[i])) list[n++] = (uint8_t)i;
        }
        if (!routed_moe_mxfp4_gate_up_pairs_launch(
                gate, up, mid, gate_ptrs, up_ptrs, xq, selected, weights,
                gate_row_bytes, expert_in_dim, expert_mid_dim, n_expert,
                0u, clamp, list, n)) return 0;
    }
    return routed_moe_mxfp4_ptrs_rows_tail_launch(out, mid, down_ptrs, midq, selected,
                                                  down_row_bytes, expert_mid_dim, out_dim,
                                                  n_expert, n_tokens, down_per_row);
}

/* Mid quantize and down for routed_moe_mxfp4_ptrs_rows_launch (all pairs'
 * mid written).  With n_tokens == 1 and down_per_row these are exactly the
 * one-token zero-copy decode launches. */
static int routed_moe_mxfp4_ptrs_rows_tail_launch(
        float *out,
        const float *mid,
        const char *const *down_ptrs,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        int down_per_row) {
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint32_t pair_count = n_tokens * n_expert;
    dim3 midq_grid(midq_blocks, pair_count, 1);
    q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, mid, expert_mid_dim, pair_count);
    if (!d20_ok(cudaGetLastError(), "routed_moe streaming rows mxfp4 mid quantize launch")) return 0;
    if (down_per_row) {
        dim3 row_grid((out_dim + 7u) / 8u, 1, 1);
        for (uint32_t t = 0; t < n_tokens; t++) {
            moe_down_mxfp4_sum6_qwarp32_kernel<false, true><<<row_grid, 256>>>(
                    out + (uint64_t)t * out_dim,
                    (const char *)down_ptrs,
                    midq + (uint64_t)t * n_expert * midq_blocks,
                    selected + (uint64_t)t * n_expert,
                    0,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
        }
    } else {
        dim3 batch_grid((out_dim + 7u) / 8u, n_tokens, 1);
        moe_down_mxfp4_sum6_qwarp32_kernel<true, true><<<batch_grid, 256>>>(
                out,
                (const char *)down_ptrs,
                midq,
                selected,
                0,
                down_row_bytes,
                midq_blocks,
                out_dim,
                n_expert);
    }
    return d20_ok(cudaGetLastError(), "routed_moe streaming rows mxfp4 down launch");
}

static int routed_moe_full_table_is_cached(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes) {
    return cuda_model_range_is_cached(model_map, gate_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, up_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, down_offset, down_bytes);
}

/* D20 step 1 switch (see moe_down_mxfp4_expert_tile8_row32_stage9_kernel). */
static int d20_down_stage9_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_ROCM_D20_DOWN_STAGE9");
        v = e && e[0] && strcmp(e, "0") ? 1 : 0;
    }
    return v;
}

/* D20 step 3 switches and the WMMA tile-list scratch. */
static int d20_wmma_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_ROCM_D20_WMMA");
        v = e && e[0] && strcmp(e, "0") ? 1 : 0;
    }
    return v;
}
static uint32_t d20_wmma_min_rows(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_ROCM_D20_WMMA_MIN");
        long x = e && e[0] ? strtol(e, NULL, 10) : 16;
        if (x < 1 || x > 100000) x = 16;
        v = (int)x;
    }
    return (uint32_t)v;
}
static int d20_wmma_scratch(uint32_t capacity, uint32_t **total, uint32_t **experts, uint32_t **starts) {
    static uint32_t *buf = NULL;
    static uint32_t cap = 0;
    if (capacity > cap) {
        if (buf) (void)cudaFree(buf);
        buf = NULL;
        cap = 0;
        if (!cuda_ok(cudaMalloc((void **)&buf, (size_t)(1u + 2u * capacity) * sizeof(uint32_t)),
                     "d20 wmma tile scratch")) return 0;
        cap = capacity;
    }
    *total = buf;
    *experts = buf + 1;
    *starts = buf + 1 + capacity;
    return 1;
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool force_resident) {
    D20ProfScope d20_scope(n_tokens);
    routed_moe_launch_plan plan;
    if (!routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                               gate_offset, up_offset, down_offset, gate_type, down_type,
                               gate_expert_bytes, down_expert_bytes, expert_in_dim,
                               expert_mid_dim, out_dim, selected, weights, n_total_expert, n_expert, x,
                               n_tokens, &plan)) {
        return 0;
    }
    const int q4k_path = plan.q4k_path;
    const int iq2_path = plan.iq2_path;
    const int iq2_iq2_path = plan.iq2_iq2_path;
    const int iq2_gate_path = iq2_path || iq2_iq2_path;
    const int q2k_path = plan.q2k_path;
    const int mxfp4_path = plan.mxfp4_path;
    const uint64_t gate_bytes = plan.gate_bytes;
    const uint64_t down_bytes = plan.down_bytes;
    uint64_t pair_count64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_expert, &pair_count64) ||
        pair_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t pair_count = (uint32_t)pair_count64;
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const char **gate_slot_ptrs = NULL;
    const char **up_slot_ptrs = NULL;
    const char **down_slot_ptrs = NULL;
    const char **resident_gate_slot_ptrs = NULL;
    const char **resident_up_slot_ptrs = NULL;
    const char **missing_gate_slot_ptrs = NULL;
    const char **missing_up_slot_ptrs = NULL;
    const uint8_t *stream_batch_pair_missing = NULL;
    uint32_t stream_resident_mask = 0;
    uint32_t stream_missing_mask = 0;
    uint32_t stream_batch_unique = 0;
    uint32_t stream_batch_resident_count = 0;
    uint32_t stream_batch_missing_count = 0;
    const int stream_full_layer =
        (n_tokens > 1u || force_resident) &&
        cuda_stream_layer_expert_cache_apply(model_map,
                                             layer_index,
                                             n_total_expert,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             &gate_w,
                                             &up_w,
                                             &down_w);
    const int full_table_cached =
        !stream_full_layer &&
        routed_moe_full_table_is_cached(model_map,
                                        gate_offset,
                                        up_offset,
                                        down_offset,
                                        gate_bytes,
                                        down_bytes);
    const int batch_stream_split_selected =
        !stream_full_layer &&
        !full_table_cached &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_apply_split(model_map,
                                               layer_index,
                                               n_total_expert,
                                               n_expert,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               &selected_exec,
                                               &resident_gate_slot_ptrs,
                                               &resident_up_slot_ptrs,
                                               &missing_gate_slot_ptrs,
                                               &missing_up_slot_ptrs,
                                               &down_slot_ptrs,
                                               &stream_batch_pair_missing,
                                               &stream_batch_resident_count,
                                               &stream_batch_missing_count,
                                               &stream_batch_unique);
    /* D2: MXFP4 verify rows whose union load was started (asynchronous
     * router readback, reads in flight) by
     * ds4_gpu_stream_expert_cache_begin_selected_batch take the overlap
     * launch; otherwise D1's synchronous prepare below. */
    const int batch_stream_overlap =
        !stream_full_layer &&
        !full_table_cached &&
        !batch_stream_split_selected &&
        n_tokens > 1u &&
        mxfp4_path && g_ssd_streaming_mode &&
        n_tokens <= DS4_ROCM_MXFP4_STREAM_MAX_ROWS &&
        routed_moe_mxfp4_batch_stream_enabled() &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_apply_overlap(model_map,
                                                 layer_index,
                                                 n_total_expert,
                                                 n_expert,
                                                 n_tokens,
                                                 gate_offset,
                                                 up_offset,
                                                 down_offset,
                                                 gate_expert_bytes,
                                                 down_expert_bytes,
                                                 &selected_exec,
                                                 &gate_slot_ptrs,
                                                 &up_slot_ptrs,
                                                 &down_slot_ptrs,
                                                 &stream_batch_unique);
    const int batch_stream_selected =
        batch_stream_overlap || (
        !stream_full_layer &&
        !full_table_cached &&
        !batch_stream_split_selected &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path ||
         (mxfp4_path && g_ssd_streaming_mode &&
          n_tokens <= DS4_ROCM_MXFP4_STREAM_MAX_ROWS &&
          routed_moe_mxfp4_batch_stream_enabled())) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_prepare(model_map,
                                           model_size,
                                           layer_index,
                                           selected,
                                           n_tokens,
                                           n_total_expert,
                                           n_expert,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           &selected_exec,
                                           &gate_slot_ptrs,
                                           &up_slot_ptrs,
                                           &down_slot_ptrs,
                                           &stream_batch_unique));
    /* The one-token resident/missing split can expose partially updated
     * selected-expert state to the default stream. Keep the asynchronous
     * read overlap, then use the deterministic compact table below. */
    int split_selected = 0;
    /* Zero-copy single-token decode (IQ2_XXS/Q2_K and MXFP4 experts): the
     * expert kernels read the resident cache slots through the per-slot
     * pointer tables instead of a compacted copy.  Only the kernel paths
     * below that have a pointer-table twin are eligible; everything else,
     * and DS4_ROCM_MOE_COMPACT_SELECTED=1, keeps the compact copy.  See
     * cuda_stream_selected_apply_ptrs for the ordering guarantees.  When it
     * is taken, compact_selected is also set: it still means "single-token
     * selected-expert path", which owns the upload wait and the reuse event
     * recorded after the routed kernels. */
    const int zero_copy_eligible =
        !stream_full_layer &&
        n_tokens == 1u &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        (iq2_path || mxfp4_path) &&
        !q4k_path &&
        down->bytes >= (uint64_t)(expert_in_dim / CUDA_QK_K) * sizeof(cuda_block_q8_K) &&
        gate->bytes >= (uint64_t)n_expert * (expert_mid_dim / CUDA_QK_K) * sizeof(cuda_block_q8_K);
    int zero_copy_selected = 0;
    /* D2: MXFP4 with misses still being read: resident-first overlap. */
    int zero_copy_overlap = 0;
    uint32_t overlap_resident_mask = 0;
    uint32_t overlap_missing_mask = 0;
    if (zero_copy_eligible && mxfp4_path) {
        zero_copy_overlap =
            cuda_stream_selected_apply_ptrs_overlap(model_map,
                                                    layer_index,
                                                    n_total_expert,
                                                    n_expert,
                                                    gate_expert_bytes,
                                                    down_expert_bytes,
                                                    &selected_exec,
                                                    &gate_w,
                                                    &up_w,
                                                    &down_w,
                                                    &gate_slot_ptrs,
                                                    &up_slot_ptrs,
                                                    &down_slot_ptrs,
                                                    &overlap_resident_mask,
                                                    &overlap_missing_mask);
        zero_copy_selected = zero_copy_overlap;
    }
    if (zero_copy_eligible && !zero_copy_overlap) {
        zero_copy_selected =
            cuda_stream_selected_apply_ptrs(model_map,
                                            layer_index,
                                            n_total_expert,
                                            n_expert,
                                            gate_expert_bytes,
                                            down_expert_bytes,
                                            &selected_exec,
                                            &gate_w,
                                            &up_w,
                                            &down_w,
                                            &gate_slot_ptrs,
                                            &up_slot_ptrs,
                                            &down_slot_ptrs);
        if (zero_copy_selected < 0) return 0;
        static int logged_zero_copy = 0;
        if (zero_copy_selected && !logged_zero_copy) {
            logged_zero_copy = 1;
            fprintf(stderr, "ds4: ROCm decode MoE reads selected experts in place (zero-copy)\n");
        }
    }
    const int compact_selected =
        split_selected ||
        zero_copy_selected ||
        (!stream_full_layer &&
        n_tokens == 1u &&
        cuda_stream_selected_apply(model_map,
                                   layer_index,
                                   n_total_expert,
                                   n_expert,
                                   gate_expert_bytes,
                                   down_expert_bytes,
                                   &selected_exec,
                                   &gate_w,
                                   &up_w,
                                   &down_w));
    if (!compact_selected && !batch_stream_selected && !batch_stream_split_selected) {
        if (g_ssd_streaming_mode &&
            n_total_expert > n_expert &&
            !stream_full_layer &&
            !full_table_cached) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "SSD streaming routed MoE missing compact selected experts "
                    "(layer=%u tokens=%u total_experts=%u selected=%u); full expert table is not mapped\n",
                    layer_index,
                    n_tokens,
                    n_total_expert,
                    n_expert);
            return 0;
        }
        if (!stream_full_layer) {
            gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
            up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
            down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
        }
    }
    if (batch_stream_selected || batch_stream_split_selected) {
        if (!down_slot_ptrs ||
            stream_batch_unique == 0) {
            return 0;
        }
        if (batch_stream_selected && (!gate_slot_ptrs || !up_slot_ptrs)) return 0;
        if (batch_stream_split_selected &&
            (!resident_gate_slot_ptrs || !resident_up_slot_ptrs ||
             !missing_gate_slot_ptrs || !missing_up_slot_ptrs ||
             !stream_batch_pair_missing)) {
            return 0;
        }
        if (!cuda_stream_batch_selected_wait_upload_ready()) return 0;
    } else if (!gate_w || !up_w || !down_w) {
        return 0;
    }
    if (compact_selected && !cuda_stream_selected_wait_upload_ready()) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    uint64_t xq_count = 0;
    uint64_t midq_count = 0;
    uint64_t xq_bytes = 0;
    uint64_t midq_bytes = 0;
    if (!cuda_u64_mul_checked(n_tokens, xq_blocks, &xq_count) ||
        !cuda_u64_mul_checked(pair_count64, midq_blocks, &midq_count) ||
        !cuda_u64_mul_checked(xq_count, sizeof(cuda_block_q8_K), &xq_bytes) ||
        !cuda_u64_mul_checked(midq_count, sizeof(cuda_block_q8_K), &midq_bytes)) {
        return 0;
    }
    if (mxfp4_path && batch_stream_selected &&
        (down->bytes < xq_bytes || gate->bytes < midq_bytes)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "SSD streaming MXFP4 rows: Q8_K scratch too small "
                        "(layer=%u tokens=%u)\n", layer_index, n_tokens);
        if (batch_stream_overlap) (void)cuda_stream_batch_selected_finish_pending_missing();
        (void)cuda_stream_batch_selected_mark_inflight();
        return 0;
    }
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        /* Correctness rollback for the optimized resident IQ2 prefill path. */
        const uint32_t disable_resident_iq2_sorted =
            iq2_gate_path && getenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED") != NULL;
        const uint32_t use_mxfp4_tiny_batch =
            mxfp4_path && n_tokens <= 4u;
        const uint32_t use_sorted_pairs =
            n_tokens > 1u &&
            !use_mxfp4_tiny_batch &&
            (!q4k_path || n_tokens >= 32u) &&
            !disable_resident_iq2_sorted;
        const uint32_t use_expert_tiles = use_sorted_pairs;
        const uint32_t use_glm_q4_row8 =
            g_glm_model && q4k_path && !g_ssd_streaming_mode &&
            ds4_rocm_is_gfx1151() && n_tokens >= 32u &&
            n_total_expert == 288u && n_expert == 8u &&
            expert_in_dim == 4096u && expert_mid_dim == 2048u &&
            out_dim == 4096u &&
            use_sorted_pairs;
        const uint32_t expert_tile_m =
            use_glm_q4_row8 ? 4u : (n_tokens <= 8u ? 4u : 8u);
        const uint32_t write_gate_up = 0u;
        const bool use_v41_wave_gate = g_deepseek41_model && iq2_gate_path &&
            n_tokens == 1u && n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && ds4_rocm_is_gfx1151();
        const uint32_t use_p2_sorted = 0u;
        const uint32_t use_atomic_down =
            !mxfp4_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_gate_row2048 =
            !q4k_path && !mxfp4_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_down_tile16 =
            !q4k_path && !mxfp4_path && use_atomic_down && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            !mxfp4_path && n_tokens == 1u && xq_blocks <= 16u;
        const uint32_t gate_row_span = 1024u;
        const uint32_t down_row_span = 1024u;
        const uint32_t use_down_row2048 =
            !q4k_path && !mxfp4_path && use_atomic_down && use_down_tile16;
        const bool use_mxfp4_draft_down = g_deepseek41_model && !g_quality_mode &&
            ds4_rocm_is_gfx1151() &&
            mxfp4_path && n_tokens >= 5u && n_tokens <= 6u &&
            n_total_expert == 128u && n_expert == 3u &&
            expert_mid_dim == 2304u && out_dim == 5120u &&
            down_row_bytes == 1224u && down_expert_bytes == 6266880u &&
            !batch_stream_selected && !batch_stream_split_selected &&
            !split_selected && !compact_selected && down_w &&
            (stream_full_layer || full_table_cached) &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_V41_MXFP4_DRAFT_DOWN_FUSED");
        const uint32_t use_direct_down_sum6 =
            (n_tokens == 1u || use_mxfp4_tiny_batch) &&
            n_expert <= DS4_ROCM_N_EXPERT_USED;
        const uint32_t use_mxfp4_ldsB =
            mxfp4_path && use_expert_tiles && n_tokens >= 128u &&
            16u * gate_row_bytes <= 60u * 1024u &&
            (expert_mid_dim % 8u) == 0u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_LDSB") != NULL;
        const uint32_t use_mxfp4_tile32 =
            mxfp4_path && use_expert_tiles && n_tokens >= 32u &&
            (expert_mid_dim % 32u) == 0u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_TILE32") != NULL;
        /* Occupancy variants (bit-exact; see kernel comments): tile4 cuts
         * the staged-activation LDS to ~29 KiB (4 blocks/WGP), row64 runs
         * 512 threads per block (2 blocks/WGP = 32 resident warps).  Both
         * target the tile8 kernel's latency-bound profile.  Default off. */
        const uint32_t use_mxfp4_tile4 =
            mxfp4_path && use_expert_tiles && !use_mxfp4_tile32 && !use_mxfp4_ldsB &&
            n_tokens >= 4u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_TILE4") != NULL;
        const uint32_t use_mxfp4_row64 =
            mxfp4_path && use_expert_tiles && !use_mxfp4_tile32 && !use_mxfp4_ldsB &&
            !use_mxfp4_tile4 && n_tokens >= 8u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_ROW64") != NULL;
        /* V4.1 Q2 uses canonical IQ2 gate/up rows and Q2_K down rows.
         * Keep other expert layouts on their existing admissions. */
        const uint32_t v41_mmq_topology =
            g_deepseek41_model && n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && n_tokens <= 2048u &&
            gate_row_bytes == 1320u && gate_expert_bytes == 3041280u &&
            down_row_bytes == 756u && down_expert_bytes == 3870720u &&
            ds4_rocm_is_gfx1151();
        /* GLM-5.3 Flash has 288 experts, beyond the generic admission cap.
         * Admit only its resident IQ2 gate/up and Q2 down topology on gfx1151;
         * the existing checks below still exclude quality and small batches. */
        const uint32_t glm53_iq2_mmq_topology =
            g_glm_model && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
            n_total_expert == DS4_ROCM_GLM53_N_EXPERT &&
            n_expert == 8u && expert_in_dim == 4096u &&
            expert_mid_dim == 2048u && out_dim == 4096u;
        const uint32_t use_rocm_mmq_gateup =
            ok && iq2_path && n_tokens >= 128u && !g_quality_mode &&
            (n_total_expert <= 256u || v41_mmq_topology || glm53_iq2_mmq_topology) &&
            !batch_stream_selected && !batch_stream_split_selected &&
            !split_selected && !compact_selected && gate_w && up_w &&
            (stream_full_layer || full_table_cached) &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_MMQ_IQ2");
        uint32_t down_row_groups = 1u;
        {
            const char *rge = getenv("DS4_ROCM_MXFP4_DOWN_RGROUP");
            if (rge && rge[0]) {
                const long rv = strtol(rge, NULL, 10);
                if (rv >= 1 && rv <= 8) down_row_groups = (uint32_t)rv;
            }
        }
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t *tile128_total = NULL;
        uint32_t *tile128_experts = NULL;
        uint32_t *tile128_starts = NULL;
        uint32_t *tile32_total = NULL;
        uint32_t *tile32_experts = NULL;
        uint32_t *tile32_starts = NULL;
        uint32_t *tile4_total = NULL;
        uint32_t *tile4_experts = NULL;
        uint32_t *tile4_starts = NULL;
        uint32_t *iq2_gate_hot_dev = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        uint32_t tile128_capacity = 0;
        uint32_t tile32_capacity = 0;
        uint32_t tile4_capacity = 0;
        if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
            fprintf(stderr,
                    "ds4: moe path n=%u mxfp4=%d stream_full=%d full_cached=%d "
                    "batch_stream=%d split=%d compact=%d sorted=%u tiles=%u\n",
                    n_tokens, mxfp4_path, stream_full_layer, full_table_cached,
                    batch_stream_selected, batch_stream_split_selected,
                    compact_selected, use_sorted_pairs, use_expert_tiles);
        }
        /* D20 step 3 admission: the default tile8 MXFP4 path at V4.1 expert
         * shapes, no occupancy variant selected. */
        const bool d20_wmma_active = mxfp4_path && use_expert_tiles && d20_wmma_enabled() &&
            !use_mxfp4_ldsB && !use_mxfp4_tile32 && !use_mxfp4_tile4 && !use_mxfp4_row64 &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u && out_dim == 5120u &&
            xq_blocks == 20u && midq_blocks == 9u;
        const uint32_t d20_wmma_min = d20_wmma_min_rows();
        uint32_t d20_wmma_capacity = 0u;
        uint32_t *d20_wmma_total = NULL, *d20_wmma_experts = NULL, *d20_wmma_starts = NULL;
        if (d20_wmma_active) {
            d20_wmma_capacity = pair_count / 16u + 385u;
            if (!d20_wmma_scratch(d20_wmma_capacity, &d20_wmma_total, &d20_wmma_experts,
                                  &d20_wmma_starts)) return 0;
        }
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        if (!use_rocm_mmq_gateup) {
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok = d20_ok(cudaGetLastError(), "routed_moe x quantize launch");
        }
        if (n_tokens == 1u) C1R(C1_MOE_PREP);
        if (zero_copy_overlap) {
            /* D2: resident-first single-token MXFP4 experts (see
             * routed_moe_mxfp4_decode_overlap_launch); the pending load is
             * always finished before this returns. */
            if (ok) {
                ok = routed_moe_mxfp4_decode_overlap_launch(
                        (float *)out->ptr,
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        down_slot_ptrs,
                        xq,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        down_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        out_dim,
                        n_expert,
                        write_gate_up,
                        clamp,
                        overlap_resident_mask,
                        overlap_missing_mask);
            } else {
                (void)cuda_stream_selected_finish_pending_missing(0);
            }
            /* Ordering guarantee 3: reuse event after the last slot reader. */
            const int inflight_ok = cuda_stream_selected_mark_inflight();
            return ok && inflight_ok;
        }
        if (!ok && (batch_stream_selected || batch_stream_split_selected)) {
            /* Ordering: the reuse fence follows any work that may read the
             * union slots, also when a launch failed. */
            if (batch_stream_overlap) (void)cuda_stream_batch_selected_finish_pending_missing();
            (void)cuda_stream_batch_selected_mark_inflight();
            return 0;
        }
        if (ok && batch_stream_overlap) {
            /* D2: verify rows, resident union slots first (bitwise D1). */
            ok = routed_moe_mxfp4_rows_overlap_launch(
                    (float *)out->ptr,
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_slot_ptrs,
                    up_slot_ptrs,
                    down_slot_ptrs,
                    xq,
                    midq,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_row_bytes,
                    down_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    out_dim,
                    n_expert,
                    n_tokens,
                    clamp,
                    routed_moe_mxfp4_stream_down_per_row());
            const int inflight_ok = cuda_stream_batch_selected_mark_inflight();
            return ok && inflight_ok;
        }
        if (ok && batch_stream_selected && mxfp4_path) {
            /* D1: DSpark verify rows on streamed MXFP4 experts, read in place
             * from the resident slots through the union pointer table. */
            ok = routed_moe_mxfp4_ptrs_rows_launch(
                    (float *)out->ptr,
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_slot_ptrs,
                    up_slot_ptrs,
                    down_slot_ptrs,
                    xq,
                    midq,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_row_bytes,
                    down_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    out_dim,
                    n_expert,
                    n_tokens,
                    clamp,
                    routed_moe_mxfp4_stream_down_per_row());
            const int inflight_ok = cuda_stream_batch_selected_mark_inflight();
            return ok && inflight_ok;
        }
        if (ok && (batch_stream_selected || batch_stream_split_selected)) {
            dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
            if (batch_stream_split_selected) {
                if (stream_batch_resident_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            resident_gate_slot_ptrs,
                            resident_up_slot_ptrs,
                            stream_batch_pair_missing,
                            0u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe streaming batch resident gate/up launch");
                }
                if (!ok) {
                    (void)cuda_stream_batch_selected_finish_pending_missing();
                } else {
                    ok = cuda_stream_batch_selected_finish_pending_missing();
                }
                if (ok && stream_batch_missing_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            missing_gate_slot_ptrs,
                            missing_up_slot_ptrs,
                            stream_batch_pair_missing,
                            1u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe streaming batch missing gate/up launch");
                }
            } else {
                moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                ok = d20_ok(cudaGetLastError(),
                             "routed_moe streaming batch gate/up launch");
            }
            if (ok && !iq2_path) {
                dim3 midq_grid(midq_blocks, pair_count, 1);
                q8_K_quantize_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        expert_mid_dim,
                        pair_count);
                ok = d20_ok(cudaGetLastError(), "routed_moe streaming batch mid quantize launch");
            }
            if (ok) {
                dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
                if (iq2_iq2_path) {
                    moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe streaming batch iq2 down launch");
                } else if (iq2_path) {
                    const ds4_rocm_runtime_config *runtime_cfg =
                        cuda_runtime_config();
                    uint32_t rows_per_block = runtime_cfg->moe_decode_down_rpb;
                    if (rows_per_block == 0u) rows_per_block = 1u;
                    const uint32_t threads = rows_per_block * 32u;
                    dim3 float_grid(
                        (out_dim + rows_per_block - 1u) / rows_per_block,
                        n_tokens,
                        1);
                    moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<
                            float_grid, threads>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_row_bytes,
                        n_expert);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe streaming batch fp32 q2 down launch");
                } else {
                    moe_down_sum6_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe streaming batch down launch");
                }
            }
            {
                /* Record the reuse fence even after a failed launch. */
                const int inflight_ok = cuda_stream_batch_selected_mark_inflight();
                ok = ok && inflight_ok;
            }
            return ok;
        }
        if (ok && use_sorted_pairs) {
            const uint32_t bucket_count = n_total_expert;
            const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = routed_moe_expert_tile_capacity(
                pair_count, expert_tile_m, bucket_count);
            tile16_capacity = use_down_tile16 ? routed_moe_expert_tile_capacity(
                pair_count, 16u, bucket_count) : 0u;
            tile128_capacity = use_mxfp4_ldsB ? routed_moe_expert_tile_capacity(
                pair_count, 128u, bucket_count) : 0u;
            tile32_capacity = use_mxfp4_tile32 ? routed_moe_expert_tile_capacity(
                pair_count, 32u, bucket_count) : 0u;
            tile4_capacity = use_mxfp4_tile4 ? routed_moe_expert_tile_capacity(
                pair_count, 4u, bucket_count) : 0u;
            const uint64_t tile_offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t iq2_gate_hot_off = tile16_starts_off + tile16_starts_bytes;
            const uint64_t iq2_gate_hot_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t tile128_offsets_off = iq2_gate_hot_off + iq2_gate_hot_bytes;
            const uint64_t tile128_offsets_bytes = use_mxfp4_ldsB ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile128_total_off = tile128_offsets_off + tile128_offsets_bytes;
            const uint64_t tile128_total_bytes = use_mxfp4_ldsB ? sizeof(uint32_t) : 0u;
            const uint64_t tile128_experts_off = tile128_total_off + tile128_total_bytes;
            const uint64_t tile128_experts_bytes = (uint64_t)tile128_capacity * sizeof(uint32_t);
            const uint64_t tile128_starts_off = tile128_experts_off + tile128_experts_bytes;
            const uint64_t tile128_starts_bytes = (uint64_t)tile128_capacity * sizeof(uint32_t);
            const uint64_t tile32_offsets_off = tile128_starts_off + tile128_starts_bytes;
            const uint64_t tile32_offsets_bytes = use_mxfp4_tile32 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile32_total_off = tile32_offsets_off + tile32_offsets_bytes;
            const uint64_t tile32_total_bytes = use_mxfp4_tile32 ? sizeof(uint32_t) : 0u;
            const uint64_t tile32_experts_off = tile32_total_off + tile32_total_bytes;
            const uint64_t tile32_experts_bytes = (uint64_t)tile32_capacity * sizeof(uint32_t);
            const uint64_t tile32_starts_off = tile32_experts_off + tile32_experts_bytes;
            const uint64_t tile32_starts_bytes = (uint64_t)tile32_capacity * sizeof(uint32_t);
            const uint64_t tile4_offsets_off = tile32_starts_off + tile32_starts_bytes;
            const uint64_t tile4_offsets_bytes = use_mxfp4_tile4 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile4_total_off = tile4_offsets_off + tile4_offsets_bytes;
            const uint64_t tile4_total_bytes = use_mxfp4_tile4 ? sizeof(uint32_t) : 0u;
            const uint64_t tile4_experts_off = tile4_total_off + tile4_total_bytes;
            const uint64_t tile4_experts_bytes = (uint64_t)tile4_capacity * sizeof(uint32_t);
            const uint64_t tile4_starts_off = tile4_experts_off + tile4_experts_bytes;
            const uint64_t tile4_starts_bytes = (uint64_t)tile4_capacity * sizeof(uint32_t);
            const uint64_t scratch_bytes = tile4_starts_off + tile4_starts_bytes;
            /* GLM keeps gate/down activations in cuda_tmp_alloc; sorted buckets
             * must survive the later mid quantization into that same buffer. */
            uint8_t *scratch = (uint8_t *)cuda_moe_sorted_tmp_alloc(scratch_bytes);
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                iq2_gate_hot_dev = (uint32_t *)(scratch + iq2_gate_hot_off);
                uint32_t *tile128_offsets = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_offsets_off) : NULL;
                tile128_total = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_total_off) : NULL;
                tile128_experts = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_experts_off) : NULL;
                tile128_starts = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_starts_off) : NULL;
                uint32_t *tile32_offsets = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_offsets_off) : NULL;
                tile32_total = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_total_off) : NULL;
                tile32_experts = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_experts_off) : NULL;
                tile32_starts = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_starts_off) : NULL;
                uint32_t *tile4_offsets = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_offsets_off) : NULL;
                tile4_total = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_total_off) : NULL;
                tile4_experts = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_experts_off) : NULL;
                tile4_starts = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_starts_off) : NULL;
                const uint32_t use_narrow_sorted_builder =
                    g_dspark_verify_mode && iq2_gate_path && use_expert_tiles &&
                    pair_count <= 48u && !use_down_tile16 &&
                    !use_mxfp4_ldsB && !use_mxfp4_tile32 && !use_mxfp4_tile4;
                if (use_narrow_sorted_builder) {
                    moe_build_narrow_sorted_tiles_kernel<<<1u, 256u>>>(
                        counts, offsets, sorted_pairs, tile_offsets, tile_total,
                        tile_experts, tile_starts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count, expert_tile_m, bucket_count);
                    ok = d20_ok(cudaGetLastError(),
                                 "routed_moe narrow sorted tiles launch");
                } else {
                    ok = cuda_ok(cudaMemset(counts, 0, counts_bytes),
                                 "routed_moe sorted counts clear");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                        sorted_pairs,
                        offsets,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles && !use_narrow_sorted_builder) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_mxfp4_ldsB) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile128_offsets, tile128_total, counts, 128u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile128 offsets launch");
                }
                if (ok && use_mxfp4_ldsB) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile128_experts, tile128_starts, tile128_offsets, counts, 128u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile128 build launch");
                }
                if (ok && use_mxfp4_tile32) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile32_offsets, tile32_total, counts, 32u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile32 offsets launch");
                }
                if (ok && use_mxfp4_tile32) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile32_experts, tile32_starts, tile32_offsets, counts, 32u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile32 build launch");
                }
                if (ok && use_mxfp4_tile4) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile4_offsets, tile4_total, counts, 4u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile4 offsets launch");
                }
                if (ok && use_mxfp4_tile4) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile4_experts, tile4_starts, tile4_offsets, counts, 4u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile4 build launch");
                }
                if (ok && use_expert_tiles && !use_narrow_sorted_builder) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile16_experts, tile16_starts, tile16_offsets, counts, 16u, bucket_count);
                    ok = d20_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        uint32_t iq2_gate_hot_count = 0u;
        uint32_t iq2_gate_hot_max = 0u;
        const uint32_t iq2_gate_hot_threshold = 8u;
        const uint32_t iq2_down_hot_threshold = 8u;
        uint32_t h_iq2_gate_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t use_iq2_gate_wmma =
            ok && !use_rocm_mmq_gateup && iq2_gate_path &&
            n_tokens >= iq2_gate_hot_threshold && n_expert == 6u && !write_gate_up &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts && iq2_gate_hot_dev && use_expert_tiles &&
            (expert_in_dim % 16u) == 0u && (expert_mid_dim % 16u) == 0u &&
            !g_quality_mode;
        if (use_iq2_gate_wmma) {
            uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
            if (!cuda_ok(cudaMemcpy(h_counts, sorted_counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe iq2 gate wmma counts copy")) {
                ok = 0;
            } else {
                for (uint32_t e = 0; e < n_total_expert; e++) {
                    const uint32_t c = h_counts[e];
                    if (c >= iq2_gate_hot_threshold) {
                        h_iq2_gate_hot[iq2_gate_hot_count++] = e;
                        if (c > iq2_gate_hot_max) iq2_gate_hot_max = c;
                    }
                }
                if (iq2_gate_hot_count != 0u &&
                    !cuda_ok(cudaMemcpy(iq2_gate_hot_dev, h_iq2_gate_hot,
                                        iq2_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                             "routed_moe iq2 gate hot copy")) {
                    ok = 0;
                }
            }
        }
        const uint32_t iq2_gate_scalar_max = iq2_gate_hot_count != 0u ? iq2_gate_hot_threshold : 0u;
        const int use_iq2_hot_f16_mid =
            ((use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
              iq2_gate_hot_threshold == iq2_down_hot_threshold) ||
             use_rocm_mmq_gateup) &&
            (out_dim & 1u) == 0u && !g_quality_mode;
        half *iq2_hot_mid_h = use_iq2_hot_f16_mid ? (half *)gate->ptr : NULL;
        half *mmq_epilogue_mid_h = NULL;
        if (use_rocm_mmq_gateup && use_iq2_hot_f16_mid) {
            uint64_t down_h_bytes = 0;
            uint64_t mid_h_bytes = 0;
            if (cuda_u64_mul3_checked(pair_count64, out_dim, sizeof(half),
                                      &down_h_bytes) &&
                cuda_u64_mul3_checked(pair_count64, expert_mid_dim, sizeof(half),
                                      &mid_h_bytes) &&
                down_h_bytes <= down->bytes &&
                mid_h_bytes <= down->bytes - down_h_bytes) {
                mmq_epilogue_mid_h =
                    (half *)((char *)down->ptr + down_h_bytes);
                iq2_hot_mid_h = mmq_epilogue_mid_h;
            }
        }
        const int use_iq2_x_f16 = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            up->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(half);
        half *iq2_x_h = use_iq2_x_f16 ? (half *)up->ptr : NULL;
        if (ok && use_iq2_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(iq2_x_h, (const float *)x->ptr, xh_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe iq2 gate x f16 launch");
        }
        int mmq_gateup_done = 0;
        if (ok && g_halo_prefill_model && halo_prefill_scope(n_tokens) && use_rocm_mmq_gateup && !g_glm_model &&
            !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
            (n_tokens == 2048u || n_tokens == 4096u) && expert_in_dim == 4096u &&
            expert_mid_dim == 2048u && out_dim == 4096u && n_total_expert == 256u &&
            n_expert == 6u && gate_type == 16u && down_type == 10u &&
            gate_expert_bytes == 2162688u && gate_row_bytes == 1056u &&
            down_expert_bytes == 2752512u && down_row_bytes == 672u &&
            use_expert_tiles && sorted_pairs && sorted_offsets && sorted_counts &&
            use_iq2_hot_f16_mid && !write_gate_up) {
            const ds4_halo_span live[] = {
                {gate->ptr,gate->bytes},{mid->ptr,mid->bytes},{down->ptr,down->bytes},
                {x->ptr,x->bytes},{out->ptr,out->bytes},{selected_exec->ptr,selected_exec->bytes},
                {weights->ptr,weights->bytes},
                {sorted_pairs,pair_count64*sizeof(uint32_t)},
                {sorted_offsets,(n_total_expert+1u)*sizeof(uint32_t)},
                {sorted_counts,n_total_expert*sizeof(uint32_t)},
                {gate_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {up_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {down_w,(uint64_t)n_total_expert*down_expert_bytes}
            };
            const int selected = ds4_rocm_halo_s9_run(n_tokens,gate_w,up_w,
                (const float*)x->ptr,sorted_pairs,sorted_offsets,(const float*)weights->ptr,
                (float*)mid->ptr,(uint16_t*)iq2_hot_mid_h,clamp,up->ptr,up->bytes,
                live,sizeof(live)/sizeof(live[0]));
            if (selected < 0) return 0;
            mmq_gateup_done = selected;
        }
        if (ok && g_deepseek41_model && use_rocm_mmq_gateup && !g_glm_model &&
            !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
            n_tokens == 2048u && expert_in_dim == 5120u &&
            expert_mid_dim == 2304u && out_dim == 5120u && n_total_expert == 384u &&
            n_expert == 6u && gate_type == 16u && down_type == 10u &&
            gate_expert_bytes == 3041280u && gate_row_bytes == 1320u &&
            down_expert_bytes == 3870720u && down_row_bytes == 756u &&
            use_expert_tiles && sorted_pairs && sorted_offsets && sorted_counts &&
            use_iq2_hot_f16_mid && !write_gate_up) {
            const ds4_halo_span live[] = {
                {gate->ptr,gate->bytes},{mid->ptr,mid->bytes},{down->ptr,down->bytes},
                {x->ptr,x->bytes},{out->ptr,out->bytes},{selected_exec->ptr,selected_exec->bytes},
                {weights->ptr,weights->bytes},
                {sorted_pairs,pair_count64*sizeof(uint32_t)},
                {sorted_offsets,(n_total_expert+1u)*sizeof(uint32_t)},
                {sorted_counts,n_total_expert*sizeof(uint32_t)},
                {gate_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {up_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {down_w,(uint64_t)n_total_expert*down_expert_bytes}
            };
            const int selected = ds4_rocm_v41_gate_up(n_tokens,gate_w,up_w,
                (const float*)x->ptr,sorted_pairs,sorted_offsets,(const float*)weights->ptr,
                (float*)mid->ptr,(uint16_t*)iq2_hot_mid_h,clamp,up->ptr,up->bytes,
                live,sizeof(live)/sizeof(live[0]));
            if (selected < 0) return 0;
            mmq_gateup_done = selected;
            static int logged_v41_gate_up = 0;
            if (selected && !logged_v41_gate_up) {
                logged_v41_gate_up = 1;
                fprintf(stderr, "ds4: ROCm V4.1 co-computed IQ2 gate/up rows=%u\n", n_tokens);
            }
        }
        if (ok && use_rocm_mmq_gateup && !mmq_gateup_done) {
            int mmq_rc = ds4_mmq_init(0) == 0 ? 0 : -1;
            const int use_fused_swiglu = mmq_epilogue_mid_h != NULL;
            /* The gfx1151 MMQ pair is stable through 2048 token rows. Tile
             * larger prefills instead of letting its flattened assignment
             * grid corrupt the tail of a 4096-row batch. */
            const uint32_t mmq_token_cap = 2048u;
            for (uint32_t token0 = 0; mmq_rc == 0 && token0 < n_tokens; ) {
                const uint32_t tile_tokens =
                    n_tokens - token0 < mmq_token_cap ?
                    n_tokens - token0 : mmq_token_cap;
                const uint64_t x_offset =
                    (uint64_t)token0 * expert_in_dim;
                const uint64_t pair_offset =
                    (uint64_t)token0 * n_expert;
                const uint64_t out_offset =
                    pair_offset * expert_mid_dim;
                if (use_fused_swiglu) {
                    mmq_rc = ds4_mmq_iq2_xxs_moe_pair_swiglu(
                        gate_w, up_w,
                        (const float *)x->ptr + x_offset,
                        (const int32_t *)selected_exec->ptr + pair_offset,
                        (const float *)weights->ptr + pair_offset,
                        (float *)gate->ptr + out_offset,
                        (float *)up->ptr + out_offset,
                        (float *)mid->ptr + out_offset,
                        mmq_epilogue_mid_h + out_offset,
                        (int)expert_mid_dim, (int)expert_in_dim,
                        (int)tile_tokens, (int)n_total_expert, (int)n_expert,
                        clamp, (cudaStream_t)0);
                } else {
                    mmq_rc = ds4_mmq_iq2_xxs_moe_pair(
                        gate_w, up_w,
                        (const float *)x->ptr + x_offset,
                        (const int32_t *)selected_exec->ptr + pair_offset,
                        (float *)gate->ptr + out_offset,
                        (float *)up->ptr + out_offset,
                        (int)expert_mid_dim, (int)expert_in_dim,
                        (int)tile_tokens, (int)n_total_expert, (int)n_expert,
                        (cudaStream_t)0);
                }
                token0 += tile_tokens;
            }
            if (mmq_rc == 0) {
                const uint64_t mid_count = pair_count64 * expert_mid_dim;
                if (use_fused_swiglu) {
                    mmq_gateup_done = 1;
                } else {
                    moe_swiglu_weighted_f32_kernel<<<
                        (uint32_t)((mid_count + 255u) / 256u), 256>>>(
                        (float *)mid->ptr, (const float *)gate->ptr,
                        (const float *)up->ptr, (const float *)weights->ptr,
                        mid_count, expert_mid_dim, clamp);
                    mmq_gateup_done = cuda_ok(
                        cudaGetLastError(), "routed_moe MMQ gate/up epilogue launch");
                    if (mmq_gateup_done && use_iq2_hot_f16_mid) {
                        f32_to_f16_kernel<<<
                            (uint32_t)((mid_count + 255u) / 256u), 256>>>(
                            iq2_hot_mid_h, (const float *)mid->ptr, mid_count);
                        mmq_gateup_done = cuda_ok(
                            cudaGetLastError(), "routed_moe MMQ mid f16 launch");
                    }
                }
                static int logged_mmq_gateup = 0;
                if (mmq_gateup_done && !logged_mmq_gateup) {
                    logged_mmq_gateup = 1;
                    fprintf(stderr,
                        "ds4: ROCm routed MoE using tuned MMQ IQ2 gate/up%s\n",
                        use_fused_swiglu ? " with fused SwiGLU write-back" : "");
                }
            } else if (!v41_mmq_topology) {
                (void)cudaGetLastError();
                static int logged_mmq_fallback = 0;
                if (!logged_mmq_fallback) {
                    logged_mmq_fallback = 1;
                    fprintf(stderr, "ds4: ROCm MMQ IQ2 gate/up returned %d; falling back\n", mmq_rc);
                }
            }
            if (v41_mmq_topology && !mmq_gateup_done) {
                /* This route skipped Q8_K x preparation. A legacy fallback
                 * would consume unwritten xq or stale half mid storage. */
                fprintf(stderr, "ds4: V4.1 MMQ gate/up or epilogue failed (%d)\n", mmq_rc);
                if (!cuda_ok(cudaStreamSynchronize((cudaStream_t)0),
                             "V4.1 MMQ failure drain")) abort();
                return 0;
            }
        }
        int split_gateup_done = 0;
        if (ok && split_selected) {
            const int split_supported =
                iq2_gate_path &&
                n_tokens == 1u &&
                n_expert <= DS4_ROCM_N_EXPERT_USED &&
                !q4k_path &&
                !sorted_pairs &&
                stream_resident_mask != 0 &&
                stream_missing_mask != 0;
            if (split_supported) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (use_v41_wave_gate) {
                    moe_v41_gate_up_wave_ptrs_kernel<4><<<dim3((expert_mid_dim + 3u) / 4u, n_expert), 128>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, stream_resident_mask, clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_resident_mask,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp);
                }
                ok = d20_ok(cudaGetLastError(), "routed_moe split resident gate/up launch");
                if (!ok) {
                    (void)cuda_stream_selected_finish_pending_missing(0);
                } else {
                    ok = cuda_stream_selected_finish_pending_missing(0);
                }
                if (ok && use_v41_wave_gate) {
                    moe_v41_gate_up_wave_ptrs_kernel<4><<<dim3((expert_mid_dim + 3u) / 4u, n_expert), 128>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, stream_missing_mask, clamp);
                } else if (ok && use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_missing_mask,
                        clamp);
                } else if (ok) {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp);
                }
                if (ok) ok = d20_ok(cudaGetLastError(), "routed_moe split missing gate/up launch");
                split_gateup_done = ok;
            } else {
                ok = cuda_stream_selected_finish_pending_missing(
                        stream_resident_mask | stream_missing_mask);
            }
        }
        if (ok && !split_gateup_done && !mmq_gateup_done) {
            if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
                fprintf(stderr,
                        "ds4: moe gate/up launch check: sorted_pairs=%p offsets=%p counts=%p "
                        "tile_total=%p tile_experts=%p tile_starts=%p\n",
                        (void *)sorted_pairs, (void *)sorted_offsets,
                        (void *)sorted_counts, (void *)tile_total,
                        (void *)tile_experts, (void *)tile_starts);
            }
            dim3 mgrid((expert_mid_dim + 31u) / 32u, pair_count, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    if (use_glm_q4_row8) {
                        dim3 row8grid((expert_mid_dim + 7u) / 8u, tile_capacity, 1);
                        moe_gate_up_mid_q4K_expert_tile4_row8_kernel<<<row8grid, 64>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                        if (expert_tile_m == 8u) {
                            moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                0u, write_gate_up, clamp);
                        } else {
                            moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                0u, write_gate_up, clamp);
                        }
                    }
                } else if (mxfp4_path) {
                    if (use_mxfp4_tile32 && tile32_total && tile32_experts && tile32_starts) {
                        dim3 t32grid((expert_mid_dim + 31u) / 32u, tile32_capacity, 1);
                        moe_gate_up_mid_mxfp4_expert_tile32_row32_kernel<<<t32grid, 256, 0, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile32_total, tile32_experts, tile32_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_ldsB && tile128_total && tile128_experts && tile128_starts) {
                        /* B-staged prefill path: 8 output rows x up to 128
                         * tokens per block; expert weights read ~once. */
                        const uint32_t ldsB_shmem = 16u * (uint32_t)gate_row_bytes;
                        if (ldsB_shmem > 48u * 1024u) {
                            static int ldsB_shmem_attr_set = 0;
                            if (!ldsB_shmem_attr_set) {
                                const cudaError_t attr_err = cudaFuncSetAttribute(
                                    (const void *)moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    16u * 3808u);
                                if (attr_err != cudaSuccess) {
                                    (void)cudaGetLastError();
                                    return 0;
                                }
                                ldsB_shmem_attr_set = 1;
                            }
                        }
                        dim3 bgrid(expert_mid_dim / 8u, tile128_capacity, 1);
                        moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel<<<bgrid, 256, ldsB_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile128_total, tile128_experts, tile128_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_tile4) {
                        const uint32_t tile4_shmem = xq_blocks <= 28u ?
                            4u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                        dim3 tgrid(tile4_capacity, (expert_mid_dim + 31u) / 32u, 1);
                        moe_gate_up_mid_mxfp4_expert_tileN_kernel<4u, 32u><<<tgrid, 256, tile4_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile4_total, tile4_experts, tile4_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_row64) {
                        const uint32_t row64_shmem = xq_blocks <= 28u ?
                            8u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                        if (row64_shmem > 48u * 1024u) {
                            static int row64_shmem_attr_set = 0;
                            if (!row64_shmem_attr_set) {
                                /* Hoisted: the compat macro cannot see a
                                 * comma inside a template-id argument. */
                                static const void *row64_fn =
                                    (const void *)&moe_gate_up_mid_mxfp4_expert_tileN_kernel<8u, 64u>;
                                const cudaError_t attr_err = cudaFuncSetAttribute(
                                    row64_fn,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    8u * 28u * 260u);
                                if (attr_err != cudaSuccess) {
                                    (void)cudaGetLastError();
                                    return 0;
                                }
                                row64_shmem_attr_set = 1;
                            }
                        }
                        dim3 tgrid(tile_capacity, (expert_mid_dim + 63u) / 64u, 1);
                        moe_gate_up_mid_mxfp4_expert_tileN_kernel<8u, 64u><<<tgrid, 512, row64_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                    dim3 tgrid(tile_capacity, (expert_mid_dim + 31u) / 32u, 1);
                    if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
                        fprintf(stderr,
                                "ds4: moe mxfp4 gate/up tile8 launch grid=(%u,%u) mid_dim=%u tile_capacity=%u xq_blocks=%u\n",
                                tgrid.x, tgrid.y, expert_mid_dim, tile_capacity, xq_blocks);
                    }
                    /* LDS staging of 8 activation rows as aligned quant
                     * slices plus scales, sized to the actual xq_blocks.
                     * Raised to 28 blocks (7168-dim x at Q8_K=256): 56.9 KiB
                     * needs the >48 KiB dynamic-smem opt-in, set once. */
                    const uint32_t tile8_shmem = xq_blocks <= 28u ?
                        8u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                    if (tile8_shmem > 48u * 1024u) {
                        static int tile8_shmem_attr_set = 0;
                        if (!tile8_shmem_attr_set) {
                            const cudaError_t attr_err = cudaFuncSetAttribute(
                                (const void *)moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                8u * 28u * 260u);
                            if (attr_err != cudaSuccess) {
                                (void)cudaGetLastError();
                                return 0;
                            }
                            tile8_shmem_attr_set = 1;
                        }
                    }
                    moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel<<<tgrid, 256, tile8_shmem, 0 /* default stream */>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        d20_wmma_active ? d20_wmma_min : 0u, write_gate_up, clamp);
                    if (d20_wmma_active) {
                        ok = d20_ok(cudaGetLastError(), "routed_moe gate/up launch");
                        d20_wmma_tiles_kernel<<<1, 64, 0, 0>>>(sorted_counts, d20_wmma_total,
                            d20_wmma_experts, d20_wmma_starts, n_total_expert, d20_wmma_min, d20_wmma_capacity);
                        ok = ok && d20_ok(cudaGetLastError(), "routed_moe d20 wmma tiles launch");
                        dim3 wgrid((expert_mid_dim + 63u) / 64u, d20_wmma_capacity, 1);
                        d20_wmma_mxfp4_kernel<true><<<wgrid, 128, 0, 0>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            d20_wmma_total, d20_wmma_experts, d20_wmma_starts,
                            (const float *)weights->ptr, gate_expert_bytes, gate_row_bytes,
                            xq_blocks, expert_mid_dim, n_expert, write_gate_up, clamp);
                    }
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else if (g_glm_model && ds4_rocm_is_gfx1151() && n_tokens == 2u &&
                           expert_in_dim == 4096u && expert_mid_dim == 2048u) {
                    // The number of nonempty tiles cannot exceed the routed token/expert pairs.
                    dim3 tgrid((expert_mid_dim + 3u) / 4u, n_tokens * n_expert, 1);
                    glm53_rocm_iq2_gate_up_verify_kernel<<<tgrid, 32>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    const char *direct_env = getenv("DS4_ROCM_V41_IQ2_VERIFY_DIRECT_SIGN");
                    const bool direct_sign = g_deepseek41_model && ds4_rocm_is_gfx1151() &&
                        n_tokens >= 2u && n_tokens <= 6u && n_total_expert == 384u &&
                        n_expert == 6u && expert_in_dim == 5120u && expert_mid_dim == 2304u &&
                        out_dim == 5120u && (!direct_env || !strcmp(direct_env,"1"));
                    if (direct_sign) {
                    moe_gate_up_mid_expert_tile4_row32_kernel<true><<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                    moe_gate_up_mid_expert_tile4_row32_kernel<false><<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                if (q4k_path) {
                    moe_gate_up_mid_q4K_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (mxfp4_path && zero_copy_selected) {
                    /* The fork's Indirect instantiation of the same kernel:
                     * gate/up "base" is the slot pointer table. */
                    dim3 mxgrid((expert_mid_dim + 7u) / 8u, pair_count, 1);
                    moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true><<<mxgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        (const char *)gate_slot_ptrs,
                        (const char *)up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        0,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (mxfp4_path) {
                    dim3 mxgrid((expert_mid_dim + 7u) / 8u, pair_count, 1);
                    moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<<<mxgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_v41_wave_gate && zero_copy_selected) {
                    /* Zero-copy twin of the wave kernel below (the fork's own
                     * pointer-table variant): same rows, lanes and sum order,
                     * expert base read from the slot table. */
                    const dim3 v41_grid((expert_mid_dim + 3u) / 4u, n_expert);
                    moe_v41_gate_up_wave_ptrs_kernel<4><<<v41_grid, 128>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, 0xffffffffu, clamp);
                } else if (use_v41_wave_gate) {
                    // V4.1 scalar: raw IQ2 and existing Q8_K activation.
                    // Four waves own four output rows; each lane handles32 values.
                    const dim3 v41_grid((expert_mid_dim + 3u) / 4u, n_expert);
                    moe_v41_gate_up_wave_pairs_kernel<4><<<v41_grid, 128>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr, gate_expert_bytes, gate_row_bytes,
                        xq_blocks, expert_mid_dim, n_expert, write_gate_up,
                        0xffffffffu, clamp);
                } else if (use_decode_lut_gate && zero_copy_selected) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        0xffffffffu,
                        clamp);
                } else if (zero_copy_selected) {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        0xffffffffu,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                }
            }
            ok = d20_ok(cudaGetLastError(), "routed_moe gate/up launch");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            if (ok && use_iq2_gate_wmma && iq2_gate_hot_count != 0u) {
                constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
                const uint32_t wmma_mtiles = 4u;
                if (wmma_mtiles == 4u) {
                    constexpr uint32_t mt = 4u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                } else {
                    constexpr uint32_t mt = 8u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                }
                ok = d20_ok(cudaGetLastError(), "routed_moe iq2 wmma hot gate/up launch");
            }
#endif
        }
        const uint32_t use_iq2_q2_float_down =
            ok && iq2_path && n_tokens > 1u &&
            n_expert <= DS4_ROCM_N_EXPERT_USED &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts;
        const uint32_t use_v41_q8_down =
            ok && g_deepseek41_model && iq2_path && n_tokens == 1u && !g_quality_mode &&
            n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && ds4_rocm_is_gfx1151();
        const uint32_t use_iq2_q2_decode_float_down =
            ok && iq2_path && n_tokens == 1u && !use_v41_q8_down &&
            n_expert <= DS4_ROCM_N_EXPERT_USED;
        if (ok && !use_iq2_q2_float_down &&
            !use_iq2_q2_decode_float_down) {
            dim3 midq_grid(midq_blocks, pair_count, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        int direct_iq2_down_done = 0;
        if (ok && use_v41_q8_down) {
            if (split_gateup_done || zero_copy_selected) {
                /* Pointer-table twin of moe_v41_down_wave: same slot (route)
                 * order of the fused sum. */
                moe_v41_down_wave_ptrs_kernel<4><<<(out_dim + 3u) / 4u, 128>>>(
                    (float *)out->ptr, down_slot_ptrs, midq,
                    (const int32_t *)selected_exec->ptr, 0,
                    down_row_bytes, midq_blocks, out_dim, n_expert);
            } else {
                moe_v41_down_wave<4><<<(out_dim + 3u) / 4u, 128>>>(
                    (float *)out->ptr, down_w, midq,
                    (const int32_t *)selected_exec->ptr, down_expert_bytes,
                    down_row_bytes, midq_blocks, out_dim, n_expert);
            }
            ok = d20_ok(cudaGetLastError(), "V4.1 quantized wave down launch");
            direct_iq2_down_done = ok;
        } else if (ok && use_iq2_q2_decode_float_down) {
            const ds4_rocm_runtime_config *runtime_cfg = cuda_runtime_config();
            uint32_t rows_per_block = runtime_cfg->moe_decode_down_rpb;
            if (rows_per_block == 0u) rows_per_block = 1u;
            const uint32_t threads = rows_per_block * 32u;
            const dim3 dgrid(
                (out_dim + rows_per_block - 1u) / rows_per_block,
                n_tokens,
                1);
            if (split_gateup_done) {
                moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<dgrid, threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            } else if (zero_copy_selected) {
                moe_down_q2K_sum_rows_w32_ptrs_kernel<<<dgrid, threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            } else {
                moe_down_q2K_sum_rows_w32_kernel<<<dgrid, threads>>>(
                    (float *)out->ptr,
                    down_w,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_expert_bytes,
                    down_row_bytes,
                    n_expert);
            }
            ok = d20_ok(cudaGetLastError(),
                         "routed_moe decode fp32 q2 down launch");
            direct_iq2_down_done = ok;
        } else if (ok && iq2_iq2_path) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
            if (split_gateup_done) {
                moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            } else {
                moe_down_iq2_sum_qwarp32_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            }
            ok = d20_ok(cudaGetLastError(), "routed_moe iq2 down launch");
            direct_iq2_down_done = ok;
        }
        int split_ptr_down_done = 0;
        if (ok && !direct_iq2_down_done && split_gateup_done) {
            moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    midq,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            ok = d20_ok(cudaGetLastError(), "routed_moe split ptr down launch");
            split_ptr_down_done = ok;
        }
        if (ok) {
            if (direct_iq2_down_done) {
                /* The IQ2 direct-sum kernel writes final token rows. */
            } else if (split_ptr_down_done) {
                /* The split pointer-table path writes the final token row. */
            } else if (use_iq2_q2_float_down) {
                if (!g_quality_mode &&
                    g_dspark_verify_mode && n_tokens <= 6u &&
                    ds4_rocm_gfx1151_flag(
                        "DS4_ROCM_DSPARK_Q2_DOWN_MMVQ") &&
                    ds4_mmq_init(0) == 0 &&
                    ds4_mmq_q2_K_moe_down_sum6_vec(
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (float *)out->ptr,
                        (int)out_dim,
                        (int)expert_mid_dim,
                        (int)n_tokens,
                        (int)n_total_expert,
                        (int)n_expert,
                        (cudaStream_t)0) == 0) {
                    ok = 1;
                } else {
                    ok = routed_moe_q2_float_down_launch(
                            out, down, mid, iq2_hot_mid_h, use_iq2_hot_f16_mid, down_w,
                            sorted_counts, sorted_offsets, sorted_pairs, tile_experts,
                            n_tokens, n_total_expert, n_expert, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes, tile_capacity);
                }
            } else {
            dim3 dgrid((out_dim + 31u) / 32u, pair_count, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_mxfp4_draft_down) {
                moe_down_mxfp4_draft_fused_kernel<32, 1, 1><<<dim3((out_dim + 31u) / 32u, n_tokens), 256>>>(
                    (float *)out->ptr, down_w, midq,
                    (const int32_t *)selected_exec->ptr, out_dim);
                const char *trace = getenv("DS4_ROCM_V41_MXFP4_DRAFT_DOWN_TRACE");
                if (trace && strcmp(trace, "1") == 0)
                    fprintf(stderr, "ds4: ROCm fused MXFP4 draft down rows=%u experts=%u mid=%u out=%u\n",
                            n_tokens, n_expert, expert_mid_dim, out_dim);
            } else if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else if (mxfp4_path) {
                    dim3 mxgrid((out_dim + 7u) / 8u, n_tokens, 1);
                    if (n_tokens == 1u && zero_copy_selected) {
                        moe_down_mxfp4_sum6_qwarp32_kernel<false, true><<<mxgrid, 256>>>(
                            (float *)out->ptr,
                            (const char *)down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            0,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                    } else if (n_tokens == 1u) {
                        moe_down_mxfp4_sum6_qwarp32_kernel<false><<<mxgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                    } else {
                        moe_down_mxfp4_sum6_qwarp32_kernel<true><<<mxgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                    }
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = d20_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_mxfp4_draft_down || use_direct_down_sum6) {
                /* The direct kernels already write the final token rows. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    if (use_glm_q4_row8) {
                        dim3 row8grid((out_dim + 7u) / 8u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile4_row8_kernel<<<row8grid, 64>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        if (expert_tile_m == 8u) {
                            moe_down_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                                use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert, use_atomic_down);
                        } else {
                            moe_down_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                                use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert, use_atomic_down);
                        }
                    }
                } else if (mxfp4_path) {
                    const uint32_t row_blocks = (out_dim + 31u) / 32u;
                    dim3 tgrid((row_blocks + down_row_groups - 1u) / down_row_groups, down_tile_capacity, 1);
                    if (midq_blocks == 9u && (d20_down_stage9_enabled() || d20_wmma_active))
                        moe_down_mxfp4_expert_tile8_row32_stage9_kernel<<<tgrid, 256, 0, 0 /* default stream */>>>(
                            d20_wmma_active ? d20_wmma_min : 0u,
                            (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts,
                            down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, 0u, down_row_groups);
                    else
                    moe_down_mxfp4_expert_tile8_row32_kernel<<<tgrid, 256, 0, 0 /* default stream */>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts,
                        down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, 0u, down_row_groups);
                    if (d20_wmma_active) {
                        ok = d20_ok(cudaGetLastError(), "routed_moe down launch");
                        dim3 wgrid((out_dim + 63u) / 64u, d20_wmma_capacity, 1);
                        d20_wmma_mxfp4_kernel<false><<<wgrid, 128, 0, 0>>>(
                            NULL, NULL, (float *)down->ptr, down_w, NULL, midq,
                            sorted_pairs, sorted_offsets, sorted_counts,
                            d20_wmma_total, d20_wmma_experts, d20_wmma_starts, NULL,
                            down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                            n_expert, 0u, 0.0f);
                        ok = ok && d20_ok(cudaGetLastError(), "routed_moe d20 wmma down launch");
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                if (q4k_path) {
                    moe_down_q4K_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = d20_ok(cudaGetLastError(), "routed_moe down launch");
            }
        }
        if (ok && !direct_iq2_down_done && !use_atomic_down &&
            !use_mxfp4_draft_down && !use_direct_down_sum6 && !use_iq2_q2_float_down) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = d20_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (zero_copy_selected) {
            /* Ordering guarantee 3 (cuda_stream_selected_apply_ptrs): record
             * the reuse event after the last kernel that reads the resident
             * slots, even if a later launch failed, so no slot is recycled
             * while an enqueued kernel can still read it. */
            const int inflight_ok = cuda_stream_selected_mark_inflight();
            ok = ok && inflight_ok;
        } else if (ok && compact_selected) {
            ok = cuda_stream_selected_mark_inflight();
        }
        return ok;
    }

    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (q2k_path && (batch_stream_selected || batch_stream_split_selected)) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                       pair_count,
                       1);
        if (batch_stream_split_selected) {
            if (stream_batch_resident_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        resident_gate_slot_ptrs,
                        resident_up_slot_ptrs,
                        stream_batch_pair_missing,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = d20_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch resident gate/up launch");
            }
            if (!ok) {
                (void)cuda_stream_batch_selected_finish_pending_missing();
            } else {
                ok = cuda_stream_batch_selected_finish_pending_missing();
            }
            if (ok && stream_batch_missing_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        missing_gate_slot_ptrs,
                        missing_up_slot_ptrs,
                        stream_batch_pair_missing,
                        1u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = d20_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch missing gate/up launch");
            }
        } else {
            moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_slot_ptrs,
                    up_slot_ptrs,
                    NULL,
                    0u,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    0xffffffffu,
                    clamp,
                    store_gate_up);
            ok = d20_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch gate/up launch");
        }
        if (ok) {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block,
                           n_tokens,
                           1);
            moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            ok = d20_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch down launch");
        }
        if (ok) ok = cuda_stream_batch_selected_mark_inflight();
        return ok;
    }

    if (q2k_path && n_tokens >= 32u && !cfg->graph_dump) {
        const uint32_t bucket_count = n_total_expert;
        const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
        const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        uint64_t sorted_bytes = 0;
        const uint64_t hot_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t hot_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = !g_quality_mode &&
                                 expert_in_dim % 16u == 0u &&
                                 expert_mid_dim % 16u == 0u &&
                                 out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
        uint64_t f16_mid_bytes = 0;
        uint64_t f16_down_bytes = 0;
        uint64_t wmma_x_bytes = 0;
        if (!cuda_u64_mul_checked(pair_count64, sizeof(uint32_t), &sorted_bytes) ||
            (moe_wmma_hot && (
                !cuda_u64_mul3_checked(pair_count64, expert_mid_dim, sizeof(__half), &f16_mid_bytes) ||
                !cuda_u64_mul3_checked(pair_count64, out_dim, sizeof(__half), &f16_down_bytes) ||
                !cuda_u64_mul3_checked(n_tokens, expert_in_dim, sizeof(__half), &wmma_x_bytes)))) {
            return 0;
        }
        uint64_t wmma_list_base = 0;
        uint64_t base_scratch_end = 0;
        uint64_t tmp64 = 0;
        uint64_t f16_mid_off = 0;
        uint64_t f16_down_off = 0;
        uint64_t wmma_x_off = 0;
        uint64_t scratch_bytes = 0;
        if (!routed_moe_u64_add_checked(counts_bytes, offsets_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, cursors_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, sorted_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, hot_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, hot_down_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_down_bytes, &base_scratch_end) ||
            !routed_moe_align256_checked(base_scratch_end, &f16_mid_off) ||
            !routed_moe_u64_add_checked(f16_mid_off, f16_mid_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &f16_down_off) ||
            !routed_moe_u64_add_checked(f16_down_off, f16_down_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &wmma_x_off) ||
            !routed_moe_u64_add_checked(wmma_x_off, wmma_x_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &scratch_bytes)) {
            return 0;
        }
        uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + f16_low_gate_bytes);
        __half *wmma_mid_h = moe_wmma_hot ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_hot ? (__half *)(scratch + f16_down_off) : NULL;
        __half *wmma_x_h = moe_wmma_hot ? (__half *)(scratch + wmma_x_off) : NULL;
        ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                    sorted_pairs,
                    offsets,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_hot) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = d20_ok(cudaGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_low[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t wmma_hot_threshold = 8u;
        const uint32_t wmma_f16_low_threshold = 64u;
        if (moe_wmma_hot) {
            if (!cuda_ok(cudaMemcpy(h_counts, counts, bucket_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < bucket_count; e++) {
                const uint32_t c = h_counts[e];
                if (c >= wmma_hot_threshold) {
                    if (c < wmma_f16_low_threshold) {
                        h_f16_low[wmma_f16_low_count++] = e;
                        if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                    } else {
                        h_f16_hot[wmma_f16_hot_count++] = e;
                        if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                    }
                }
            }
        }
        const uint32_t gate_rpb = 16u;
        const uint32_t down_rpb = 16u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = 4u * 256u * sizeof(float);
        const size_t down_shmem = 4u * 256u * sizeof(float);
        const uint32_t scalar_max = moe_wmma_hot && (wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)
            ? wmma_hot_threshold : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, bucket_count, 1);
        moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_in_dim, expert_mid_dim,
                gate_expert_bytes, gate_row_bytes, n_expert, clamp);
        if (!d20_ok(cudaGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!d20_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!d20_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
#endif
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, bucket_count, 1);
        if (moe_wmma_hot) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                    (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        }
        if (!d20_ok(cudaGetLastError(), "routed_moe q2 expert down launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt4 * bm * bn) * sizeof(float) + 2u * bn * 84u;
            if (!cuda_ok(cudaMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!d20_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
            if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!d20_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
        }
#endif
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_hot) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = d20_ok(cudaGetLastError(), "routed_moe q2 expert sum launch");
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        return ok;
    }

    if (q2k_path) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        const int q8k_gateup = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= xq_bytes;
        const int decode_profile =
            n_tokens == 1u && routed_moe_decode_profile_enabled();
        ds4_rocm_moe_decode_profile_record decode_profile_rec = {0};
        if (decode_profile) {
            if (!routed_moe_decode_profile_ensure_events()) return 0;
            g_moe_decode_profile_stats.calls++;
            if (split_selected) g_moe_decode_profile_stats.split_calls++;
        }
        int ok_gateup = 1;
        if (split_selected) {
            const uint32_t compact_mask = stream_resident_mask | stream_missing_mask;
            if (compact_mask == 0u) return 0;
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                           pair_count,
                           1);
            if (stream_resident_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                            "Q2 decode MoE profile resident gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = d20_ok(cudaGetLastError(),
                                     "routed_moe q2 split resident gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_resident = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                            "Q2 decode MoE profile resident gate/up end")) {
                        return 0;
                    }
                }
            }
            if (!ok_gateup) {
                (void)cuda_stream_selected_finish_pending_missing(0);
                return 0;
            }
            const double finish_missing_t0 =
                decode_profile ? cuda_wall_sec() : 0.0;
            ok_gateup = cuda_stream_selected_finish_pending_missing(0);
            if (decode_profile) {
                g_moe_decode_profile_stats.finish_missing_ms +=
                    (cuda_wall_sec() - finish_missing_t0) * 1000.0;
            }
            if (ok_gateup && stream_missing_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                            "Q2 decode MoE profile missing gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = d20_ok(cudaGetLastError(),
                                     "routed_moe q2 split missing gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_missing = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                            "Q2 decode MoE profile missing gate/up end")) {
                        return 0;
                    }
                }
            }
        } else if (q8k_gateup) {
            if (decode_profile) g_moe_decode_profile_stats.q8_gateup_calls++;
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            cuda_block_q8_K *xq_gate = (cuda_block_q8_K *)down->ptr;
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq_gate, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok_gateup = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate input quantize launch");
            if (ok_gateup) {
                dim3 gate_grid((expert_mid_dim + 255u) / 256u, pair_count, 1);
                moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<gate_grid, 256u>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq_gate,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        (uint32_t)store_gate_up,
                        clamp);
                ok_gateup = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate/up launch");
            }
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else if (gate_rows_per_block == 1u) {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid(expert_mid_dim, pair_count, 1);
            moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<gate_grid, 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block, pair_count, 1);
            moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        }
        if (!ok_gateup) return 0;
        int ok_decode_moe = 1;
        const int q8k_down = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= midq_bytes;
        if (decode_profile && q8k_down) {
            g_moe_decode_profile_stats.q8_down_calls++;
        }
        if (q8k_down) {
            cuda_block_q8_K *midq = (cuda_block_q8_K *)down->ptr;
            dim3 midq_grid(midq_blocks, pair_count, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                        "Q2 decode MoE profile mid quant start")) {
                return 0;
            }
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok_decode_moe = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k mid quantize launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.mid_quant = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                        "Q2 decode MoE profile mid quant end")) {
                    return 0;
                }
            }
            if (ok_decode_moe) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                            "Q2 decode MoE profile down start")) {
                    return 0;
                }
                if (split_selected) {
                    moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                }
                ok_decode_moe = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k down launch");
                if (decode_profile && ok_decode_moe) {
                    decode_profile_rec.down = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                            "Q2 decode MoE profile down end")) {
                        return 0;
                    }
                }
            }
        } else {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block, n_tokens, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                        "Q2 decode MoE profile down start")) {
                return 0;
            }
            if (split_selected) {
                moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_row_bytes,
                        n_expert);
            } else {
                moe_down_q2K_sum_rows_w32_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_expert_bytes,
                        down_row_bytes,
                        n_expert);
            }
            ok_decode_moe = d20_ok(cudaGetLastError(), "routed_moe q2 oldhip rows down launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.down = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                        "Q2 decode MoE profile down end")) {
                    return 0;
                }
            }
        }
        if (ok_decode_moe && compact_selected) {
            ok_decode_moe = cuda_stream_selected_mark_inflight();
        }
        if (ok_decode_moe && decode_profile) {
            ok_decode_moe =
                routed_moe_decode_profile_collect(&decode_profile_rec);
        }
        return ok_decode_moe;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, pair_count, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = d20_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, pair_count, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected_exec->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = d20_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = d20_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in, uint32_t layer_index, bool force_resident) {
    if (add_in) {
        fprintf(stderr, "ds4: routed MoE addend fold is Metal-only\n");
        return 0;
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, 1,
                             force_resident);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, n_tokens,
                             force_resident);
}

/* Test hook for tests/test_mxfp4_stream_rows_rocm.c (model-free).  Uploads
 * n_union MXFP4 experts to separately allocated device slots, builds the
 * slot pointer table, and runs on identical inputs:
 *   (a) the streamed rows path: Q8_K of all rows, then
 *       routed_moe_mxfp4_ptrs_rows_launch with the given down mode;
 *   (b) the one-token decode launches of routed_moe_launch (n_tokens == 1,
 *       zero-copy MXFP4), once per row on per-row views.
 * Returns mid (n_tokens * n_expert * mid) and out (n_tokens * out_dim) of
 * both.  Outputs are pre-filled with different patterns (NaN for (a), a
 * large finite value for (b)) so an unwritten element always differs. */
extern "C" int ds4_gpu_test_mxfp4_stream_rows(
        const void *gate_host,
        const void *up_host,
        const void *down_host,
        uint32_t n_union,
        const float *x_host,
        const int32_t *selected_host,
        const float *weights_host,
        uint32_t n_tokens,
        uint32_t n_expert,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        float clamp,
        int down_per_row,
        float *mid_rows_host,
        float *mid_ref_host,
        float *out_rows_host,
        float *out_ref_host) {
    if (!gate_host || !up_host || !down_host || !x_host || !selected_host ||
        !weights_host || n_union == 0u || n_union > 64u || n_tokens == 0u ||
        n_tokens > DS4_ROCM_MXFP4_STREAM_MAX_ROWS ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim % CUDA_QK_K != 0u || expert_mid_dim % CUDA_QK_K != 0u ||
        out_dim == 0u) return 0;
    const uint64_t gate_row_bytes = (uint64_t)expert_in_dim / 32u * 17u;
    const uint64_t down_row_bytes = (uint64_t)expert_mid_dim / 32u * 17u;
    const uint64_t gate_expert_bytes = gate_row_bytes * expert_mid_dim;
    const uint64_t down_expert_bytes = down_row_bytes * out_dim;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint32_t pairs = n_tokens * n_expert;
    const size_t mid_bytes = (size_t)pairs * expert_mid_dim * sizeof(float);
    const size_t out_bytes = (size_t)n_tokens * out_dim * sizeof(float);
    int ok = 1;
    char *slots[3][64] = {{0}};
    const char *tables_host[3][64] = {{0}};
    const char **tables[3] = {0};
    float *x = NULL, *w = NULL, *scratch = NULL;
    float *mid_a = NULL, *mid_b = NULL, *out_a = NULL, *out_b = NULL;
    int32_t *sel = NULL;
    cuda_block_q8_K *xq_a = NULL, *xq_b = NULL, *midq_a = NULL, *midq_b = NULL;
    const void *src[3] = {gate_host, up_host, down_host};
    const uint64_t eb[3] = {gate_expert_bytes, gate_expert_bytes, down_expert_bytes};
    for (uint32_t k = 0; ok && k < 3u; k++) {
        for (uint32_t u = 0; ok && u < n_union; u++) {
            ok = cuda_ok(cudaMalloc((void **)&slots[k][u], (size_t)eb[k]), "test slot alloc") &&
                 cuda_ok(cudaMemcpy(slots[k][u], (const char *)src[k] + (uint64_t)u * eb[k],
                                    (size_t)eb[k], cudaMemcpyHostToDevice), "test slot upload");
            tables_host[k][u] = slots[k][u];
        }
        ok = ok && cuda_ok(cudaMalloc((void **)&tables[k], n_union * sizeof(char *)), "test table alloc") &&
             cuda_ok(cudaMemcpy((void *)tables[k], tables_host[k], n_union * sizeof(char *),
                                cudaMemcpyHostToDevice), "test table upload");
    }
    ok = ok &&
        cuda_ok(cudaMalloc((void **)&x, (size_t)n_tokens * expert_in_dim * sizeof(float)), "test x") &&
        cuda_ok(cudaMalloc((void **)&w, pairs * sizeof(float)), "test w") &&
        cuda_ok(cudaMalloc((void **)&sel, pairs * sizeof(int32_t)), "test sel") &&
        cuda_ok(cudaMalloc((void **)&scratch, mid_bytes), "test scratch") &&
        cuda_ok(cudaMalloc((void **)&mid_a, mid_bytes), "test mid a") &&
        cuda_ok(cudaMalloc((void **)&mid_b, mid_bytes), "test mid b") &&
        cuda_ok(cudaMalloc((void **)&out_a, out_bytes), "test out a") &&
        cuda_ok(cudaMalloc((void **)&out_b, out_bytes), "test out b") &&
        cuda_ok(cudaMalloc((void **)&xq_a, (size_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K)), "test xq a") &&
        cuda_ok(cudaMalloc((void **)&xq_b, (size_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K)), "test xq b") &&
        cuda_ok(cudaMalloc((void **)&midq_a, (size_t)pairs * midq_blocks * sizeof(cuda_block_q8_K)), "test midq a") &&
        cuda_ok(cudaMalloc((void **)&midq_b, (size_t)pairs * midq_blocks * sizeof(cuda_block_q8_K)), "test midq b");
    ok = ok &&
        cuda_ok(cudaMemcpy(x, x_host, (size_t)n_tokens * expert_in_dim * sizeof(float), cudaMemcpyHostToDevice), "test x up") &&
        cuda_ok(cudaMemcpy(w, weights_host, pairs * sizeof(float), cudaMemcpyHostToDevice), "test w up") &&
        cuda_ok(cudaMemcpy(sel, selected_host, pairs * sizeof(int32_t), cudaMemcpyHostToDevice), "test sel up") &&
        cuda_ok(cudaMemset(mid_a, 0xff, mid_bytes), "test poison") &&
        cuda_ok(cudaMemset(out_a, 0xff, out_bytes), "test poison") &&
        cuda_ok(cudaMemset(mid_b, 0x7e, mid_bytes), "test poison") &&
        cuda_ok(cudaMemset(out_b, 0x7e, out_bytes), "test poison");
    /* (a) streamed rows path, as routed_moe_launch's batch-stream branch. */
    if (ok) {
        q8_K_quantize_kernel<<<dim3(xq_blocks, n_tokens, 1), 256>>>(xq_a, x, expert_in_dim, n_tokens);
        ok = d20_ok(cudaGetLastError(), "test x quantize") &&
             routed_moe_mxfp4_ptrs_rows_launch(out_a, scratch, scratch, mid_a,
                                               tables[0], tables[1], tables[2],
                                               xq_a, midq_a, sel, w,
                                               gate_row_bytes, down_row_bytes,
                                               expert_in_dim, expert_mid_dim, out_dim,
                                               n_expert, n_tokens, clamp, down_per_row);
    }
    /* (b) one-token decode launches (routed_moe_launch, n_tokens == 1,
     * mxfp4_path && zero_copy_selected), one row at a time. */
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        cuda_block_q8_K *xq_t = xq_b + (uint64_t)t * xq_blocks;
        cuda_block_q8_K *midq_t = midq_b + (uint64_t)t * n_expert * midq_blocks;
        float *mid_t = mid_b + (uint64_t)t * n_expert * expert_mid_dim;
        q8_K_quantize_kernel<<<dim3(xq_blocks, 1, 1), 256>>>(
                xq_t, x + (uint64_t)t * expert_in_dim, expert_in_dim, 1u);
        moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true><<<
                dim3((expert_mid_dim + 7u) / 8u, n_expert, 1), 256>>>(
                scratch, scratch, mid_t,
                (const char *)tables[0], (const char *)tables[1], xq_t,
                sel + (uint64_t)t * n_expert, w + (uint64_t)t * n_expert,
                0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, 0u, clamp);
        q8_K_quantize_kernel<<<dim3(midq_blocks, n_expert, 1), 256>>>(
                midq_t, mid_t, expert_mid_dim, n_expert);
        moe_down_mxfp4_sum6_qwarp32_kernel<false, true><<<
                dim3((out_dim + 7u) / 8u, 1u, 1), 256>>>(
                out_b + (uint64_t)t * out_dim, (const char *)tables[2], midq_t,
                sel + (uint64_t)t * n_expert, 0, down_row_bytes, midq_blocks,
                out_dim, n_expert);
        ok = d20_ok(cudaGetLastError(), "test decode reference launch");
    }
    ok = ok && cuda_ok(cudaDeviceSynchronize(), "test sync") &&
        cuda_ok(cudaMemcpy(mid_rows_host, mid_a, mid_bytes, cudaMemcpyDeviceToHost), "test mid a down") &&
        cuda_ok(cudaMemcpy(mid_ref_host, mid_b, mid_bytes, cudaMemcpyDeviceToHost), "test mid b down") &&
        cuda_ok(cudaMemcpy(out_rows_host, out_a, out_bytes, cudaMemcpyDeviceToHost), "test out a down") &&
        cuda_ok(cudaMemcpy(out_ref_host, out_b, out_bytes, cudaMemcpyDeviceToHost), "test out b down");
    void *frees[] = {x, w, sel, scratch, mid_a, mid_b, out_a, out_b, xq_a, xq_b, midq_a, midq_b,
                     (void *)tables[0], (void *)tables[1], (void *)tables[2]};
    for (size_t i = 0; i < sizeof(frees) / sizeof(frees[0]); i++) if (frees[i]) (void)cudaFree(frees[i]);
    for (uint32_t k = 0; k < 3u; k++)
        for (uint32_t u = 0; u < n_union; u++) if (slots[k][u]) (void)cudaFree(slots[k][u]);
    return ok;
}

/* ---- D2: asynchronous router readback and pending union loads ---------- */

static int32_t *g_d2_readback_host;
static cudaEvent_t g_d2_readback_event;
static uint32_t g_d2_readback_n;
static int g_d2_readback_pending;

extern "C" int ds4_gpu_d2_decode_mode(void) {
    return cuda_stream_d2_decode_mode();
}

extern "C" int ds4_gpu_d2_verify_enabled(void) {
    return cuda_stream_d2_verify_enabled();
}

/* Enqueue the D2H copy of the router ids into a pinned mailbox behind an
 * event on stream 0.  The caller enqueues independent work (the shared
 * expert) and then calls _finish, which waits for this event only.  The
 * mailbox is rewritten only by the next _start, after _finish consumed it. */
extern "C" int ds4_gpu_selected_readback_start(const ds4_gpu_tensor *selected,
                                               uint32_t n_ids) {
    g_d2_readback_pending = 0;
    if (n_ids == 0u || n_ids > DS4_ROCM_D2_MAX_PAIRS ||
        !cuda_tensor_has_i32(selected, n_ids)) {
        return 0;
    }
    if (!g_d2_readback_host) {
        cudaError_t err = cudaMallocHost((void **)&g_d2_readback_host,
                                         DS4_ROCM_D2_MAX_PAIRS * sizeof(int32_t));
        if (err != cudaSuccess) {
            g_d2_readback_host = NULL;
            return cuda_ok(err, "D2 router readback pinned allocation");
        }
    }
    if (!g_d2_readback_event &&
        !cuda_ok(cudaEventCreateWithFlags(&g_d2_readback_event, cudaEventDisableTiming),
                 "D2 router readback event creation")) {
        g_d2_readback_event = NULL;
        return 0;
    }
    if (!cuda_ok(cudaMemcpyAsync(g_d2_readback_host, selected->ptr,
                                 (size_t)n_ids * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, 0),
                 "D2 router readback enqueue") ||
        !cuda_ok(cudaEventRecord(g_d2_readback_event, 0),
                 "D2 router readback event record")) {
        return 0;
    }
    g_d2_readback_n = n_ids;
    g_d2_readback_pending = 1;
    return 1;
}

extern "C" int ds4_gpu_selected_readback_finish(int32_t *ids, uint32_t n_ids) {
    if (!ids || !g_d2_readback_pending || g_d2_readback_n != n_ids) {
        g_d2_readback_pending = 0;
        return 0;
    }
    g_d2_readback_pending = 0;
    if (!cuda_ok(cudaEventSynchronize(g_d2_readback_event), "D2 router readback wait")) {
        return 0;
    }
    memcpy(ids, g_d2_readback_host, (size_t)n_ids * sizeof(int32_t));
    return 1;
}

/* Start the union load of a verify batch without waiting for its reads (the
 * launcher's batch_stream_overlap consumes it).  Skipped (returns 1, nothing
 * pending) where the launcher would not read per-expert slots: no SSD
 * streaming, a full-layer cache for this layer, or a fully cached table. */
extern "C" int ds4_gpu_stream_expert_cache_begin_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t n_tokens,
        uint32_t n_selected) {
    if (!table || !selected_ids) return 0;
    if (!g_ssd_streaming_mode || n_tokens < 2u ||
        n_tokens > DS4_ROCM_MXFP4_STREAM_MAX_ROWS ||
        n_selected == 0u || n_selected > DS4_ROCM_N_EXPERT_USED) {
        return 1;
    }
    const char *gw = NULL, *uw = NULL, *dw = NULL;
    if (cuda_stream_layer_expert_cache_apply(table->model_map, table->layer,
                                             table->n_total_expert,
                                             table->gate_offset, table->up_offset,
                                             table->down_offset,
                                             table->gate_expert_bytes,
                                             table->down_expert_bytes,
                                             &gw, &uw, &dw)) {
        return 1;
    }
    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!cuda_u64_mul_checked(table->n_total_expert, table->gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(table->n_total_expert, table->down_expert_bytes, &down_bytes)) {
        return 0;
    }
    if (routed_moe_full_table_is_cached(table->model_map, table->gate_offset,
                                        table->up_offset, table->down_offset,
                                        gate_bytes, down_bytes)) {
        return 1;
    }
    const ds4_gpu_tensor *selected_exec = NULL;
    const char **gate_ptrs = NULL, **up_ptrs = NULL, **down_ptrs = NULL;
    uint32_t unique = 0;
    return cuda_stream_batch_selected_prepare_from_host(table->model_map,
                                                        table->model_size,
                                                        table->layer,
                                                        selected_ids,
                                                        n_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes,
                                                        &selected_exec,
                                                        &gate_ptrs,
                                                        &up_ptrs,
                                                        &down_ptrs,
                                                        &unique,
                                                        1);
}

/* Test hook for tests/test_mxfp4_overlap_rocm.c (model-free).  Same setup as
 * ds4_gpu_test_mxfp4_stream_rows.  On identical inputs runs
 *   (a) the reference: n_tokens == 1, the sequential zero-copy decode
 *       launches (gate/up<true> over all pairs, mid quantize, down<false,
 *       true>); n_tokens > 1, D1's routed_moe_mxfp4_ptrs_rows_launch;
 *   (b) the D2 order: gate/up through routed_moe_mxfp4_gate_up_pairs_launch
 *       in groups (group[pair] = 0..n_groups-1, launched in group order,
 *       like resident-first then misses as they land), then
 *       routed_moe_mxfp4_ptrs_rows_tail_launch (with down_per_row forced on
 *       for one row, as the decode overlap path does).
 * Returns mid and out of both; outputs are poisoned differently first. */
extern "C" int ds4_gpu_test_mxfp4_overlap(
        const void *gate_host,
        const void *up_host,
        const void *down_host,
        uint32_t n_union,
        const float *x_host,
        const int32_t *selected_host,
        const float *weights_host,
        uint32_t n_tokens,
        uint32_t n_expert,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        float clamp,
        int down_per_row,
        const uint8_t *group,
        uint32_t n_groups,
        float *mid_d2_host,
        float *mid_ref_host,
        float *out_d2_host,
        float *out_ref_host) {
    if (!gate_host || !up_host || !down_host || !x_host || !selected_host ||
        !weights_host || !group || n_union == 0u || n_union > 64u || n_tokens == 0u ||
        n_tokens > DS4_ROCM_MXFP4_STREAM_MAX_ROWS ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        n_tokens * n_expert > DS4_ROCM_D2_MAX_PAIRS ||
        expert_in_dim % CUDA_QK_K != 0u || expert_mid_dim % CUDA_QK_K != 0u ||
        out_dim == 0u) return 0;
    const uint64_t gate_row_bytes = (uint64_t)expert_in_dim / 32u * 17u;
    const uint64_t down_row_bytes = (uint64_t)expert_mid_dim / 32u * 17u;
    const uint64_t gate_expert_bytes = gate_row_bytes * expert_mid_dim;
    const uint64_t down_expert_bytes = down_row_bytes * out_dim;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint32_t pairs = n_tokens * n_expert;
    const size_t mid_bytes = (size_t)pairs * expert_mid_dim * sizeof(float);
    const size_t out_bytes = (size_t)n_tokens * out_dim * sizeof(float);
    int ok = 1;
    char *slots[3][64] = {{0}};
    const char *tables_host[3][64] = {{0}};
    const char **tables[3] = {0};
    float *x = NULL, *w = NULL, *scratch = NULL;
    float *mid_a = NULL, *mid_b = NULL, *out_a = NULL, *out_b = NULL;
    int32_t *sel = NULL;
    cuda_block_q8_K *xq = NULL, *midq_a = NULL, *midq_b = NULL;
    const void *src[3] = {gate_host, up_host, down_host};
    const uint64_t eb[3] = {gate_expert_bytes, gate_expert_bytes, down_expert_bytes};
    for (uint32_t k = 0; ok && k < 3u; k++) {
        for (uint32_t u = 0; ok && u < n_union; u++) {
            ok = cuda_ok(cudaMalloc((void **)&slots[k][u], (size_t)eb[k]), "test slot alloc") &&
                 cuda_ok(cudaMemcpy(slots[k][u], (const char *)src[k] + (uint64_t)u * eb[k],
                                    (size_t)eb[k], cudaMemcpyHostToDevice), "test slot upload");
            tables_host[k][u] = slots[k][u];
        }
        ok = ok && cuda_ok(cudaMalloc((void **)&tables[k], n_union * sizeof(char *)), "test table alloc") &&
             cuda_ok(cudaMemcpy((void *)tables[k], tables_host[k], n_union * sizeof(char *),
                                cudaMemcpyHostToDevice), "test table upload");
    }
    ok = ok &&
        cuda_ok(cudaMalloc((void **)&x, (size_t)n_tokens * expert_in_dim * sizeof(float)), "test x") &&
        cuda_ok(cudaMalloc((void **)&w, pairs * sizeof(float)), "test w") &&
        cuda_ok(cudaMalloc((void **)&sel, pairs * sizeof(int32_t)), "test sel") &&
        cuda_ok(cudaMalloc((void **)&scratch, mid_bytes), "test scratch") &&
        cuda_ok(cudaMalloc((void **)&mid_a, mid_bytes), "test mid a") &&
        cuda_ok(cudaMalloc((void **)&mid_b, mid_bytes), "test mid b") &&
        cuda_ok(cudaMalloc((void **)&out_a, out_bytes), "test out a") &&
        cuda_ok(cudaMalloc((void **)&out_b, out_bytes), "test out b") &&
        cuda_ok(cudaMalloc((void **)&xq, (size_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K)), "test xq") &&
        cuda_ok(cudaMalloc((void **)&midq_a, (size_t)pairs * midq_blocks * sizeof(cuda_block_q8_K)), "test midq a") &&
        cuda_ok(cudaMalloc((void **)&midq_b, (size_t)pairs * midq_blocks * sizeof(cuda_block_q8_K)), "test midq b");
    ok = ok &&
        cuda_ok(cudaMemcpy(x, x_host, (size_t)n_tokens * expert_in_dim * sizeof(float), cudaMemcpyHostToDevice), "test x up") &&
        cuda_ok(cudaMemcpy(w, weights_host, pairs * sizeof(float), cudaMemcpyHostToDevice), "test w up") &&
        cuda_ok(cudaMemcpy(sel, selected_host, pairs * sizeof(int32_t), cudaMemcpyHostToDevice), "test sel up") &&
        cuda_ok(cudaMemset(mid_a, 0xff, mid_bytes), "test poison") &&
        cuda_ok(cudaMemset(out_a, 0xff, out_bytes), "test poison") &&
        cuda_ok(cudaMemset(mid_b, 0x7e, mid_bytes), "test poison") &&
        cuda_ok(cudaMemset(out_b, 0x7e, out_bytes), "test poison");
    if (ok) {
        q8_K_quantize_kernel<<<dim3(xq_blocks, n_tokens, 1), 256>>>(xq, x, expert_in_dim, n_tokens);
        ok = d20_ok(cudaGetLastError(), "test x quantize");
    }
    /* (a) reference */
    if (ok && n_tokens == 1u) {
        moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true><<<
                dim3((expert_mid_dim + 7u) / 8u, n_expert, 1), 256>>>(
                scratch, scratch, mid_b,
                (const char *)tables[0], (const char *)tables[1], xq,
                sel, w, 0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, 0u, clamp);
        q8_K_quantize_kernel<<<dim3(midq_blocks, n_expert, 1), 256>>>(
                midq_b, mid_b, expert_mid_dim, n_expert);
        moe_down_mxfp4_sum6_qwarp32_kernel<false, true><<<
                dim3((out_dim + 7u) / 8u, 1u, 1), 256>>>(
                out_b, (const char *)tables[2], midq_b, sel, 0, down_row_bytes,
                midq_blocks, out_dim, n_expert);
        ok = d20_ok(cudaGetLastError(), "test decode reference launch");
    } else if (ok) {
        ok = routed_moe_mxfp4_ptrs_rows_launch(out_b, scratch, scratch, mid_b,
                                               tables[0], tables[1], tables[2],
                                               xq, midq_b, sel, w,
                                               gate_row_bytes, down_row_bytes,
                                               expert_in_dim, expert_mid_dim, out_dim,
                                               n_expert, n_tokens, clamp, down_per_row);
    }
    /* (b) D2 order */
    for (uint32_t g = 0; ok && g < n_groups; g++) {
        uint8_t list[DS4_ROCM_D2_MAX_PAIRS];
        uint32_t n = 0;
        for (uint32_t p = 0; p < pairs; p++) if (group[p] == g) list[n++] = (uint8_t)p;
        ok = routed_moe_mxfp4_gate_up_pairs_launch(scratch, scratch, mid_a,
                                                   tables[0], tables[1], xq, sel, w,
                                                   gate_row_bytes, expert_in_dim,
                                                   expert_mid_dim, n_expert, 0u, clamp,
                                                   list, n);
    }
    if (ok) {
        ok = routed_moe_mxfp4_ptrs_rows_tail_launch(out_a, mid_a, tables[2], midq_a, sel,
                                                    down_row_bytes, expert_mid_dim, out_dim,
                                                    n_expert, n_tokens,
                                                    n_tokens == 1u ? 1 : down_per_row);
    }
    ok = ok && cuda_ok(cudaDeviceSynchronize(), "test sync") &&
        cuda_ok(cudaMemcpy(mid_d2_host, mid_a, mid_bytes, cudaMemcpyDeviceToHost), "test mid a down") &&
        cuda_ok(cudaMemcpy(mid_ref_host, mid_b, mid_bytes, cudaMemcpyDeviceToHost), "test mid b down") &&
        cuda_ok(cudaMemcpy(out_d2_host, out_a, out_bytes, cudaMemcpyDeviceToHost), "test out a down") &&
        cuda_ok(cudaMemcpy(out_ref_host, out_b, out_bytes, cudaMemcpyDeviceToHost), "test out b down");
    void *frees[] = {x, w, sel, scratch, mid_a, mid_b, out_a, out_b, xq, midq_a, midq_b,
                     (void *)tables[0], (void *)tables[1], (void *)tables[2]};
    for (size_t i = 0; i < sizeof(frees) / sizeof(frees[0]); i++) if (frees[i]) (void)cudaFree(frees[i]);
    for (uint32_t k = 0; k < 3u; k++)
        for (uint32_t u = 0; u < n_union; u++) if (slots[k][u]) (void)cudaFree(slots[k][u]);
    return ok;
}

/* D20 model-free test hook: one expert, `count` (row, expert) pairs, V4.1
 * down shape (K = 2304 = 9 Q8_K blocks, 5120 outputs), random MXFP4 weights
 * and Q8_K activations from `seed`.  Runs the original tile8 down kernel
 * (unstaged at 9 blocks) and the stage-9 copy on identical inputs with the
 * given row_groups, and counts output elements whose bits differ.
 * Returns 1 when both ran. */
extern "C" int ds4_gpu_test_d20_down_stage9(uint32_t count, uint32_t seed, uint32_t row_groups,
                                            uint64_t *n_diff, uint64_t *n_total) {
    const uint32_t midq_blocks = 9u, out_dim = 5120u, mid_dim = 2304u;
    const uint64_t row_bytes = (uint64_t)mid_dim / 32u * sizeof(cuda_block_mxfp4);
    const uint64_t expert_bytes = row_bytes * out_dim;
    if (!count || count > 4096u || !n_diff || !n_total || !row_groups || row_groups > 8u) return 0;
    uint64_t st = 0x9e3779b97f4a7c15ull ^ ((uint64_t)seed * 0xbf58476d1ce4e5b9ull);
    auto rnd = [&st]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    std::vector<unsigned char> w(expert_bytes);
    for (uint64_t r = 0; r < (uint64_t)out_dim * (mid_dim / 32u); r++) {
        cuda_block_mxfp4 *blk = (cuda_block_mxfp4 *)(w.data() + r * sizeof(cuda_block_mxfp4));
        blk->e = (uint8_t)(120u + rnd() % 12u);
        for (int q = 0; q < 16; q++) blk->qs[q] = (uint8_t)rnd();
    }
    std::vector<cuda_block_q8_K> xq((size_t)count * midq_blocks);
    for (auto &b : xq) {
        memset(&b, 0, sizeof(b));
        b.d = (float)((int32_t)(rnd() % 2001u) - 1000) * 1e-5f + 1e-6f;
        for (int q = 0; q < 256; q++) b.qs[q] = (int8_t)(rnd() & 0xff);
    }
    const uint32_t tiles = (count + 7u) / 8u;
    std::vector<uint32_t> pairs(count), tile_exp(tiles, 0u), tile_start(tiles);
    for (uint32_t i = 0; i < count; i++) pairs[i] = i;
    for (uint32_t t = 0; t < tiles; t++) tile_start[t] = 8u * t;
    const uint32_t offsets0 = 0u, counts0 = count, total = tiles;
    char *dw = NULL; cuda_block_q8_K *dq = NULL; float *o1 = NULL, *o2 = NULL;
    uint32_t *dp = NULL, *doff = NULL, *dcnt = NULL, *dtot = NULL, *dte = NULL, *dts = NULL;
    const size_t out_bytes = (size_t)count * out_dim * sizeof(float);
    int ok = cuda_ok(cudaMalloc((void **)&dw, expert_bytes), "d20 test w") &&
        cuda_ok(cudaMalloc((void **)&dq, xq.size() * sizeof(cuda_block_q8_K)), "d20 test xq") &&
        cuda_ok(cudaMalloc((void **)&o1, out_bytes), "d20 test o1") &&
        cuda_ok(cudaMalloc((void **)&o2, out_bytes), "d20 test o2") &&
        cuda_ok(cudaMalloc((void **)&dp, count * 4u), "d20 test pairs") &&
        cuda_ok(cudaMalloc((void **)&doff, 4u), "d20 test off") &&
        cuda_ok(cudaMalloc((void **)&dcnt, 4u), "d20 test cnt") &&
        cuda_ok(cudaMalloc((void **)&dtot, 4u), "d20 test tot") &&
        cuda_ok(cudaMalloc((void **)&dte, tiles * 4u), "d20 test te") &&
        cuda_ok(cudaMalloc((void **)&dts, tiles * 4u), "d20 test ts");
    if (ok) ok = cuda_ok(cudaMemcpy(dw, w.data(), expert_bytes, cudaMemcpyHostToDevice), "d20 up w") &&
        cuda_ok(cudaMemcpy(dq, xq.data(), xq.size() * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice), "d20 up xq") &&
        cuda_ok(cudaMemcpy(dp, pairs.data(), count * 4u, cudaMemcpyHostToDevice), "d20 up p") &&
        cuda_ok(cudaMemcpy(doff, &offsets0, 4u, cudaMemcpyHostToDevice), "d20 up o") &&
        cuda_ok(cudaMemcpy(dcnt, &counts0, 4u, cudaMemcpyHostToDevice), "d20 up c") &&
        cuda_ok(cudaMemcpy(dtot, &total, 4u, cudaMemcpyHostToDevice), "d20 up t") &&
        cuda_ok(cudaMemcpy(dte, tile_exp.data(), tiles * 4u, cudaMemcpyHostToDevice), "d20 up te") &&
        cuda_ok(cudaMemcpy(dts, tile_start.data(), tiles * 4u, cudaMemcpyHostToDevice), "d20 up ts") &&
        cuda_ok(cudaMemset(o1, 0xff, out_bytes), "d20 fill o1") &&
        cuda_ok(cudaMemset(o2, 0x7f, out_bytes), "d20 fill o2");
    if (ok) {
        const uint32_t row_blocks = (out_dim + 31u) / 32u;
        dim3 grid((row_blocks + row_groups - 1u) / row_groups, tiles, 1);
        moe_down_mxfp4_expert_tile8_row32_kernel<<<grid, 256>>>(o1, dw, dq, dp, doff, dcnt, dtot, dte, dts,
            expert_bytes, row_bytes, midq_blocks, out_dim, 1u, 0u, row_groups);
        ok = cuda_ok(cudaGetLastError(), "d20 test original");
        moe_down_mxfp4_expert_tile8_row32_stage9_kernel<<<grid, 256>>>(0u, o2, dw, dq, dp, doff, dcnt, dtot, dte, dts,
            expert_bytes, row_bytes, midq_blocks, out_dim, 1u, 0u, row_groups);
        ok = ok && cuda_ok(cudaGetLastError(), "d20 test stage9") && cuda_ok(cudaDeviceSynchronize(), "d20 test sync");
    }
    if (ok) {
        std::vector<uint32_t> a((size_t)count * out_dim), b((size_t)count * out_dim);
        ok = cuda_ok(cudaMemcpy(a.data(), o1, out_bytes, cudaMemcpyDeviceToHost), "d20 down a") &&
             cuda_ok(cudaMemcpy(b.data(), o2, out_bytes, cudaMemcpyDeviceToHost), "d20 down b");
        uint64_t d = 0;
        for (size_t i = 0; i < a.size(); i++) d += a[i] != b[i];
        *n_diff = d;
        *n_total = a.size();
    }
    cudaFree(dw); cudaFree(dq); cudaFree(o1); cudaFree(o2); cudaFree(dp); cudaFree(doff);
    cudaFree(dcnt); cudaFree(dtot); cudaFree(dte); cudaFree(dts);
    return ok;
}

/* D20 step 3 model-free test hook.  kind 0: gate/up (K 5120, 2304 rows,
 * SwiGLU mid and raw gate/up), kind 1: down (K 2304, 5120 rows).  One
 * expert with `count` pairs (n_expert 6, pair i -> token i, slot i % 6),
 * random MXFP4 weights, Q8_K activations and route weights from `seed`.
 * Runs the tile8 kernel (gate/up: the default staged kernel; down: the
 * original) and the WMMA kernel on identical inputs; counts differing output
 * bits (gate/up: mid, gate and up) and times each over `reps` launches. */
extern "C" int ds4_gpu_test_d20_wmma(uint32_t kind, uint32_t count, uint32_t seed, uint32_t reps,
                                     uint64_t *n_diff, uint64_t *n_total, float *ms_tile8, float *ms_wmma) {
    const bool gu = kind == 0u;
    const uint32_t K = gu ? 5120u : 2304u, rows = gu ? 2304u : 5120u, nb = K / 256u;
    const uint32_t n_expert = 6u;
    const uint64_t row_bytes = (uint64_t)K / 32u * sizeof(cuda_block_mxfp4);
    const uint64_t expert_bytes = row_bytes * rows;
    if (!count || count > 8192u || !n_diff || !n_total || !reps) return 0;
    uint64_t st = 0x2545f4914f6cdd1dull ^ ((uint64_t)seed * 0x9e3779b97f4a7c15ull);
    auto rnd = [&st]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    const uint32_t nm = gu ? 2u : 1u;
    std::vector<unsigned char> w(expert_bytes * nm);
    for (uint64_t r = 0; r < (uint64_t)nm * rows * (K / 32u); r++) {
        cuda_block_mxfp4 *blk = (cuda_block_mxfp4 *)(w.data() + r * sizeof(cuda_block_mxfp4));
        blk->e = (uint8_t)(118u + rnd() % 14u);
        for (int q = 0; q < 16; q++) blk->qs[q] = (uint8_t)rnd();
    }
    const uint32_t n_act = gu ? count : count * n_expert;   /* gate/up: by token; down: by pair */
    std::vector<cuda_block_q8_K> act((size_t)n_act * nb);
    for (auto &b : act) {
        memset(&b, 0, sizeof(b));
        b.d = (float)((int32_t)(rnd() % 2001u) - 1000) * 1e-4f + 1e-6f;
        for (int q = 0; q < 256; q++) b.qs[q] = (int8_t)(rnd() & 0xff);
    }
    std::vector<float> rw((size_t)count * n_expert);
    for (auto &x : rw) x = (float)(rnd() % 1000u) * 1e-3f;
    std::vector<uint32_t> pairs(count);
    for (uint32_t i = 0; i < count; i++) pairs[i] = i * n_expert + (i % n_expert);
    const uint32_t t8 = (count + 7u) / 8u, t16 = (count + 15u) / 16u;
    std::vector<uint32_t> te8(t8, 0u), ts8(t8), te16(t16, 0u), ts16(t16);
    for (uint32_t t = 0; t < t8; t++) ts8[t] = 8u * t;
    for (uint32_t t = 0; t < t16; t++) ts16[t] = 16u * t;
    const uint32_t zero = 0u;
    char *dw = NULL; cuda_block_q8_K *da = NULL; float *drw = NULL;
    float *o[6] = {0};
    uint32_t *dp = NULL, *doff = NULL, *dcnt = NULL, *dt8 = NULL, *de8 = NULL, *ds8 = NULL, *dt16 = NULL, *de16 = NULL, *ds16 = NULL;
    const size_t out_elems = (size_t)count * n_expert * rows;
    const size_t out_bytes = out_elems * sizeof(float);
    int ok = cuda_ok(cudaMalloc((void **)&dw, w.size()), "d20w w") &&
        cuda_ok(cudaMalloc((void **)&da, act.size() * sizeof(cuda_block_q8_K)), "d20w act") &&
        cuda_ok(cudaMalloc((void **)&drw, rw.size() * 4u), "d20w rw") &&
        cuda_ok(cudaMalloc((void **)&dp, count * 4u), "d20w p") &&
        cuda_ok(cudaMalloc((void **)&doff, 4u), "d20w off") && cuda_ok(cudaMalloc((void **)&dcnt, 4u), "d20w cnt") &&
        cuda_ok(cudaMalloc((void **)&dt8, 4u), "d20w t8") && cuda_ok(cudaMalloc((void **)&de8, t8 * 4u), "d20w e8") &&
        cuda_ok(cudaMalloc((void **)&ds8, t8 * 4u), "d20w s8") && cuda_ok(cudaMalloc((void **)&dt16, 4u), "d20w t16") &&
        cuda_ok(cudaMalloc((void **)&de16, t16 * 4u), "d20w e16") && cuda_ok(cudaMalloc((void **)&ds16, t16 * 4u), "d20w s16");
    for (int k = 0; ok && k < 6; k++) ok = cuda_ok(cudaMalloc((void **)&o[k], out_bytes), "d20w out");
    if (ok) ok = cuda_ok(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "u w") &&
        cuda_ok(cudaMemcpy(da, act.data(), act.size() * sizeof(cuda_block_q8_K), cudaMemcpyHostToDevice), "u a") &&
        cuda_ok(cudaMemcpy(drw, rw.data(), rw.size() * 4u, cudaMemcpyHostToDevice), "u rw") &&
        cuda_ok(cudaMemcpy(dp, pairs.data(), count * 4u, cudaMemcpyHostToDevice), "u p") &&
        cuda_ok(cudaMemcpy(doff, &zero, 4u, cudaMemcpyHostToDevice), "u off") &&
        cuda_ok(cudaMemcpy(dcnt, &count, 4u, cudaMemcpyHostToDevice), "u cnt") &&
        cuda_ok(cudaMemcpy(dt8, &t8, 4u, cudaMemcpyHostToDevice), "u t8") &&
        cuda_ok(cudaMemcpy(de8, te8.data(), t8 * 4u, cudaMemcpyHostToDevice), "u e8") &&
        cuda_ok(cudaMemcpy(ds8, ts8.data(), t8 * 4u, cudaMemcpyHostToDevice), "u s8") &&
        cuda_ok(cudaMemcpy(dt16, &t16, 4u, cudaMemcpyHostToDevice), "u t16") &&
        cuda_ok(cudaMemcpy(de16, te16.data(), t16 * 4u, cudaMemcpyHostToDevice), "u e16") &&
        cuda_ok(cudaMemcpy(ds16, ts16.data(), t16 * 4u, cudaMemcpyHostToDevice), "u s16");
    for (int k = 0; ok && k < 6; k++) ok = cuda_ok(cudaMemset(o[k], k < 3 ? 0xff : 0x7f, out_bytes), "fill");
    cudaEvent_t e0, e1, e2;
    if (ok) ok = cuda_ok(cudaEventCreate(&e0), "ev") && cuda_ok(cudaEventCreate(&e1), "ev") && cuda_ok(cudaEventCreate(&e2), "ev");
    const float clamp = 10.0f;
    if (ok) {
        dim3 g8((rows + 31u) / 32u, t8, 1), g8gu(t8, (rows + 31u) / 32u, 1), g16((rows + 63u) / 64u, t16, 1);
        const uint32_t sh = 8u * nb * (256u + (uint32_t)sizeof(float));
        (void)cudaEventRecord(e0, 0);
        for (uint32_t r = 0; r < reps; r++) {
            if (gu) moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel<<<g8gu, 256, sh>>>(o[1], o[2], o[0], dw, dw + expert_bytes, da,
                        dp, doff, dcnt, dt8, de8, ds8, drw, expert_bytes, row_bytes, nb, rows, n_expert, 0u, 1u, clamp);
            else moe_down_mxfp4_expert_tile8_row32_kernel<<<g8, 256>>>(o[0], dw, da, dp, doff, dcnt, dt8, de8, ds8,
                        expert_bytes, row_bytes, nb, rows, n_expert, 0u, 1u);
        }
        (void)cudaEventRecord(e1, 0);
        for (uint32_t r = 0; r < reps; r++) {
            if (gu) d20_wmma_mxfp4_kernel<true><<<g16, 128>>>(o[4], o[5], o[3], dw, dw + expert_bytes, da, dp, doff, dcnt,
                        dt16, de16, ds16, drw, expert_bytes, row_bytes, nb, rows, n_expert, 1u, clamp);
            else d20_wmma_mxfp4_kernel<false><<<g16, 128>>>(NULL, NULL, o[3], dw, NULL, da, dp, doff, dcnt,
                        dt16, de16, ds16, NULL, expert_bytes, row_bytes, nb, rows, n_expert, 0u, 0.0f);
        }
        (void)cudaEventRecord(e2, 0);
        ok = cuda_ok(cudaGetLastError(), "d20w launch") && cuda_ok(cudaDeviceSynchronize(), "d20w sync");
        float a = 0, b = 0;
        (void)cudaEventElapsedTime(&a, e0, e1);
        (void)cudaEventElapsedTime(&b, e1, e2);
        if (ms_tile8) *ms_tile8 = a / reps;
        if (ms_wmma) *ms_wmma = b / reps;
    }
    if (ok) {
        std::vector<uint32_t> x(out_elems), y(out_elems);
        uint64_t d = 0, tot = 0;
        for (int k = 0; ok && k < (gu ? 3 : 1); k++) {
            ok = cuda_ok(cudaMemcpy(x.data(), o[k], out_bytes, cudaMemcpyDeviceToHost), "d x") &&
                 cuda_ok(cudaMemcpy(y.data(), o[3 + k], out_bytes, cudaMemcpyDeviceToHost), "d y");
            for (uint32_t i = 0; i < count; i++)
                for (uint32_t r = 0; r < rows; r++) {
                    const size_t off = (size_t)pairs[i] * rows + r;
                    d += x[off] != y[off];
                    tot++;
                }
        }
        *n_diff = d;
        *n_total = tot;
    }
    cudaFree(dw); cudaFree(da); cudaFree(drw); cudaFree(dp); cudaFree(doff); cudaFree(dcnt);
    cudaFree(dt8); cudaFree(de8); cudaFree(ds8); cudaFree(dt16); cudaFree(de16); cudaFree(ds16);
    for (int k = 0; k < 6; k++) cudaFree(o[k]);
    return ok;
}
