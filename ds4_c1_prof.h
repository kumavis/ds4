/* C1 decode compute profiler (ROCm only).  Off by default and free when off:
 * every mark is one load of g_ds4_c1_active and a branch.
 *
 *   DS4_ROCM_C1_PROF=coarse   host wall time at the marks that already follow
 *                             a device drain (layer end, logits); no GPU events,
 *                             no extra synchronisation.
 *   DS4_ROCM_C1_PROF=events   every mark also records a timing hipEvent on the
 *                             null stream (no synchronisation); GPU time per
 *                             segment = elapsed between consecutive events.
 *   DS4_ROCM_C1_PROF=sync     every mark first drains the device
 *                             (hipDeviceSynchronize) and then records; host time
 *                             per segment = isolated latency of that segment.
 *   DS4_ROCM_C1_PROF_TOKENS=N skip the first N decode tokens (default 4).
 *   DS4_ROCM_C1_PROF_LAYERS=1 also print the per-layer table.
 *
 * Each mark closes the segment that began at the previous mark; the segment
 * takes the mark's label.  A report is printed at exit (and by
 * ds4_gpu_c1_report). */
#ifndef DS4_C1_PROF_H
#define DS4_C1_PROF_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    C1_START = 0,
    C1_EMBED,        /* engram hash/read start, pre write, token embedding */
    C1_LAYER_PRE,    /* draft capture, engram rows upload */
    C1_HC_PRE,       /* engram add, hc mix (norm+matmul+sinkhorn), hc sum, attn norm */
    C1_ATTN_PROJ,    /* q_a/kv projections, norms, q_b */
    C1_ATTN_ROPE,    /* rope q/kv, kv quantize, window write */
    C1_COMPRESSOR,   /* KV compressor / pool / index-k publish */
    C1_INDEXER,      /* indexer q/proj/scores + candidate filter + top-k */
    C1_ATTN_CORE,    /* gather compressed KV, attention, bf16, inverse rope */
    C1_ATTN_OUT,     /* output a/b projections, bf16 */
    C1_HC_POST,      /* hc expand, hc mix, hc sum, ffn norm */
    C1_ROUTER,       /* router logits matmul, top-k select, id readback start */
    C1_SHARED,       /* shared expert gate/up/swiglu/down */
    C1_SEL_WAIT,     /* host: wait router ids, start expert loads (cache lookup) */
    C1_MOE_PREP,     /* routed: plan, pointer tables, upload wait, x quantize */
    C1_MOE_GU_RES,   /* routed: gate/up of resident experts */
    C1_MOE_READ_WAIT,/* routed: host waits for missing experts (reads) */
    C1_MOE_GU_MISS,  /* routed: gate/up of experts that were read */
    C1_MOE_DOWN,     /* routed: mid quantize + fused down (sum of 6) */
    C1_MOE_TAIL,     /* routed (overlap path): reuse event, return */
    C1_MOE_ALLRES,   /* routed (no read pending): gate/up + down of 6 resident experts */
    C1_MOE_FIN,      /* shared + routed add, bf16 */
    C1_HC_EXPAND,    /* hc expand after FFN, pre copy */
    C1_DRAIN,        /* layer-end device drain (end_commands) */
    C1_QUIESCE,      /* stream cache quiesce after the last layer */
    C1_HEAD,         /* final hc sum, output norm, vocab head matmul */
    C1_LOGITS_READ,  /* logits device->host copy (drains the head) */
    C1_STEP_TAIL,    /* d0 log, engram wait, bookkeeping */
    C1_N_LABELS
};

int ds4_gpu_c1_mode(void);                 /* 0 off, 1 coarse, 2 events, 3 sync */
void ds4_gpu_c1_token_begin(void);
void ds4_gpu_c1_mark(int label, int layer);
void ds4_gpu_c1_mark_fine(int label, int layer);  /* skipped in coarse mode */
void ds4_gpu_c1_mark_routed_end(int layer);   /* MOE_TAIL or MOE_ALLRES */
void ds4_gpu_c1_token_end(void);
void ds4_gpu_c1_note_token_ms(double total_ms); /* whole perf-token wall time */
void ds4_gpu_c1_report(void);

#ifdef __cplusplus
}
#endif


#endif
