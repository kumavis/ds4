__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return ds4_precise_expf(x);
    return ds4_precise_log1pf(ds4_precise_expf(x));
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t N_EXPERT>
__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        float expert_weight_scale,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * N_EXPERT;
    float *prob = probs + (uint64_t)t * N_EXPERT;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    __shared__ float sprob[4][N_EXPERT];
    float local_prob[N_EXPERT / 32u];
    float local_score[N_EXPERT / 32u];

    #pragma unroll
    for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = ds4_precise_sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * n_expert_used;
            float sum = 0.0f;
            for (uint32_t j = 0; j < n_expert_used; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < N_EXPERT) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * expert_weight_scale;
        }
        return;
    }

    float out_prob[DS4_ROCM_N_EXPERT_USED] = {0.0f};
    uint32_t out_idx[DS4_ROCM_N_EXPERT_USED] = {0};
    for (uint32_t k = 0; k < n_expert_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(FULL_WARP_MASK, best_score, mask);
            const float other_prob = __shfl_xor_sync(FULL_WARP_MASK, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(FULL_WARP_MASK, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < n_expert_used; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * expert_weight_scale;
    }
}

extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : DS4_ROCM_N_EXPERT;
    const uint32_t active_n_expert_used = n_expert_used != 0u ? n_expert_used : DS4_ROCM_N_EXPERT_USED;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : DS4_ROCM_EXPERT_WEIGHT_SCALE;
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != DS4_ROCM_DSPARK_N_EXPERT &&
         active_n_expert != DS4_ROCM_N_EXPERT &&
         active_n_expert != DS4_ROCM_GLM53_N_EXPERT &&
         active_n_expert != DS4_ROCM_MAX_N_EXPERT) ||
        active_n_expert_used > DS4_ROCM_N_EXPERT_USED ||
        !(active_scale > 0.0f) ||
        !cuda_tensor_has_f32(logits, active_n_expert) ||
        !cuda_tensor_has_f32(probs, active_n_expert) ||
        !cuda_tensor_has_i32(selected, active_n_expert_used) ||
        !cuda_tensor_has_f32(weights, active_n_expert_used)) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset, active_n_expert * sizeof(float))) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, active_n_expert * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        if (hash_rows == 0u) ok = 0;
        else {
            const uint64_t hash_bytes = (uint64_t)hash_rows * active_n_expert_used * sizeof(int32_t);
            if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes)) ok = 0;
            else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
            if (!hash) ok = 0;
        }
    }
    if (ok) {
        dim3 block(32, 4, 1);
        if (active_n_expert == DS4_ROCM_DSPARK_N_EXPERT) {
            router_select_warp_topk_kernel<DS4_ROCM_DSPARK_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_n_expert_used, active_scale, has_bias && !hash_mode, hash_mode);
        } else if (active_n_expert == DS4_ROCM_MAX_N_EXPERT) {
            router_select_warp_topk_kernel<DS4_ROCM_MAX_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_n_expert_used, active_scale, has_bias && !hash_mode, hash_mode);
        } else if (active_n_expert == DS4_ROCM_GLM53_N_EXPERT) {
            router_select_warp_topk_kernel<DS4_ROCM_GLM53_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_n_expert_used, active_scale, has_bias && !hash_mode, hash_mode);
        } else {
            router_select_warp_topk_kernel<DS4_ROCM_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_n_expert_used, active_scale, has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : DS4_ROCM_N_EXPERT;
    const uint32_t active_n_expert_used = n_expert_used != 0u ? n_expert_used : DS4_ROCM_N_EXPERT_USED;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : DS4_ROCM_EXPERT_WEIGHT_SCALE;
    if (!selected || !weights || !probs || !logits || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != DS4_ROCM_DSPARK_N_EXPERT &&
         active_n_expert != DS4_ROCM_N_EXPERT &&
         active_n_expert != DS4_ROCM_GLM53_N_EXPERT &&
         active_n_expert != DS4_ROCM_MAX_N_EXPERT) ||
        active_n_expert_used > DS4_ROCM_N_EXPERT_USED ||
        !(active_scale > 0.0f) ||
        (hash_mode && !cuda_tensor_has_i32(tokens, n_tokens)) ||
        !cuda_tensor_has_elems2(logits, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(probs, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, active_n_expert_used, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, active_n_expert_used, sizeof(float))) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset, active_n_expert * sizeof(float))) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, active_n_expert * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        if (hash_rows == 0u) return 0;
        const uint64_t hash_bytes = (uint64_t)hash_rows * active_n_expert_used * sizeof(int32_t);
        if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes)) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    dim3 block(32, 4, 1);
    if (active_n_expert == DS4_ROCM_DSPARK_N_EXPERT) {
        router_select_warp_topk_kernel<DS4_ROCM_DSPARK_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                tokens ? (const int32_t *)tokens->ptr : NULL,
                0,
                hash_rows,
                n_tokens,
                active_n_expert_used,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    } else if (active_n_expert == DS4_ROCM_MAX_N_EXPERT) {
        router_select_warp_topk_kernel<DS4_ROCM_MAX_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                tokens ? (const int32_t *)tokens->ptr : NULL,
                0,
                hash_rows,
                n_tokens,
                active_n_expert_used,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    } else if (active_n_expert == DS4_ROCM_GLM53_N_EXPERT) {
        router_select_warp_topk_kernel<DS4_ROCM_GLM53_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                tokens ? (const int32_t *)tokens->ptr : NULL,
                0,
                hash_rows,
                n_tokens,
                active_n_expert_used,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    } else {
        router_select_warp_topk_kernel<DS4_ROCM_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                tokens ? (const int32_t *)tokens->ptr : NULL,
                0,
                hash_rows,
                n_tokens,
                active_n_expert_used,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

/* ---- D16: cache-conditional routing ("Cache-Prior", exploration) ---------
 * DS4_ROCM_ROUTE_CACHE_PRIOR=<lambda> (default 0 = off: nothing below runs and
 * the router is the unmodified kernel above), DS4_ROCM_ROUTE_CACHE_KEEP=<n>
 * (default 3).  Output-changing: runs with lambda > 0 must be labelled
 * (DS4_OUTPUT_CLASS=approx-routing).
 *
 * Definition (the recipe's route(), after MoCCE Eq. 9/10):
 *   s[e] = sqrt(softplus(logit[e]))       (router_select_warp_topk_kernel's probs)
 *   c[e] = s[e] + bias[e]
 *   r_obs = max(c) - min(c)
 *   R = Rbar[layer] if Rbar[layer] > 0 else r_obs
 *   P = unbiased top-keep (the first keep picks of the unmodified kernel)
 *   c[e] += lambda * R  for e resident in the transaction's snapshot or in P
 *   picks = top-6 of c (ties -> lower index); weights from s of the picks,
 *   normalised and scaled exactly as the unmodified kernel does.
 * The second kernel runs after the unmodified one and overwrites its picks and
 * weights; it reads the unbiased picks first, so it also records per row and
 * layer: r_obs, which unbiased ranks were dropped, and the weight share of the
 * dropped exact picks ("drift").
 *
 * Snapshot: a 40 x 384 residency mask of the SSD expert cache (model_map's
 * entries) plus the frozen per-layer Rbar, taken and uploaded by
 * ds4_gpu_route_prior_open() when a transaction opens (a decode step, or a
 * whole verify round). */
enum { DS4_RP_MAX_LAYERS = 64, DS4_RP_MAX_ROWS = 8, DS4_RP_STAT = 4 };
static unsigned char *g_rp_mask_dev;
static float *g_rp_rbar_dev;
static float *g_rp_stats_dev;
static unsigned char *g_rp_mask_host;
static float *g_rp_rbar_host;
static uint32_t g_rp_layers;
static float g_rp_lambda = -1.0f;
static int g_rp_keep = -1;
static uint64_t g_rp_snap_resident;

extern "C" float ds4_gpu_route_prior_lambda(void) {
    if (g_rp_lambda < 0.0f) {
        g_rp_lambda = 0.0f;
        const char *env = getenv("DS4_ROCM_ROUTE_CACHE_PRIOR");
        if (env && env[0]) {
            char *end = NULL;
            const double v = strtod(env, &end);
            if (end != env && *end == '\0' && isfinite(v) && v >= 0.0 && v <= 10.0) g_rp_lambda = (float)v;
            else fprintf(stderr, DS4_GPU_LOG_PREFIX "D16: ignoring DS4_ROCM_ROUTE_CACHE_PRIOR=%s\n", env);
        }
    }
    return g_rp_lambda;
}

extern "C" uint32_t ds4_gpu_route_prior_keep(void) {
    if (g_rp_keep < 0) {
        g_rp_keep = 3;
        const char *env = getenv("DS4_ROCM_ROUTE_CACHE_KEEP");
        if (env && env[0]) {
            char *end = NULL;
            const long v = strtol(env, &end, 10);
            if (end != env && *end == '\0' && v >= 0 && v <= 6) g_rp_keep = (int)v;
            else fprintf(stderr, DS4_GPU_LOG_PREFIX "D16: ignoring DS4_ROCM_ROUTE_CACHE_KEEP=%s\n", env);
        }
    }
    return (uint32_t)g_rp_keep;
}

static int cuda_rp_alloc(uint32_t n_layer) {
    if (g_rp_mask_dev) return n_layer <= g_rp_layers;
    if (n_layer == 0 || n_layer > DS4_RP_MAX_LAYERS) return 0;
    const size_t mask_bytes = (size_t)DS4_RP_MAX_LAYERS * DS4_ROCM_N_EXPERT;
    if (cudaMalloc((void **)&g_rp_mask_dev, mask_bytes) != cudaSuccess ||
        cudaMalloc((void **)&g_rp_rbar_dev, DS4_RP_MAX_LAYERS * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&g_rp_stats_dev, (size_t)DS4_RP_MAX_ROWS * DS4_RP_MAX_LAYERS *
                   DS4_RP_STAT * sizeof(float)) != cudaSuccess ||
        cudaMallocHost((void **)&g_rp_mask_host, mask_bytes) != cudaSuccess ||
        cudaMallocHost((void **)&g_rp_rbar_host, DS4_RP_MAX_LAYERS * sizeof(float)) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, DS4_GPU_LOG_PREFIX "D16: allocation failed\n");
        return 0;
    }
    g_rp_layers = DS4_RP_MAX_LAYERS;
    return 1;
}

/* Open a routing transaction: snapshot residency of model_map's experts and
 * upload it with the frozen per-layer Rbar (rbar[il] <= 0: the row uses its
 * own range).  Ordered on the null stream ahead of the routers that read it.
 * Callers open only when the previous transaction's routers have run (a
 * decode step or verify round ends with a host drain), so the pinned host
 * buffers are not overwritten while a copy is pending. */
extern "C" int ds4_gpu_route_prior_open(const void *model_map, uint32_t n_layer,
                                        const float *rbar) {
    if (!model_map || !rbar || !cuda_rp_alloc(n_layer)) return 0;
    memset(g_rp_mask_host, 0, (size_t)n_layer * DS4_ROCM_N_EXPERT);
    uint64_t n = 0;
    for (const cuda_stream_resident_expert &e : g_stream_resident_experts) {
        if (e.model_map != model_map || e.layer >= n_layer || e.expert < 0 ||
            (uint32_t)e.expert >= DS4_ROCM_N_EXPERT) continue;
        g_rp_mask_host[(size_t)e.layer * DS4_ROCM_N_EXPERT + (uint32_t)e.expert] = 1;
        n++;
    }
    g_rp_snap_resident = n;
    memcpy(g_rp_rbar_host, rbar, n_layer * sizeof(float));
    return cuda_ok(cudaMemcpyAsync(g_rp_mask_dev, g_rp_mask_host, (size_t)n_layer * DS4_ROCM_N_EXPERT,
                                   cudaMemcpyHostToDevice, 0), "D16 mask upload") &&
        cuda_ok(cudaMemcpyAsync(g_rp_rbar_dev, g_rp_rbar_host, n_layer * sizeof(float),
                                cudaMemcpyHostToDevice, 0), "D16 rbar upload");
}

extern "C" uint64_t ds4_gpu_route_prior_snapshot_count(void) { return g_rp_snap_resident; }

template <uint32_t N_EXPERT>
__global__ static void route_prior_kernel(
        int32_t *selected, float *weights, const float *probs, const float *bias,
        const unsigned char *mask, const float *rbar, float *stats,
        uint32_t layer, uint32_t n_layer, uint32_t row_offset, uint32_t n_tokens,
        uint32_t n_expert_used, uint32_t keep, float lambda, float expert_weight_scale) {
    const uint32_t lane = threadIdx.x;
    const uint32_t t = blockIdx.x * blockDim.y + threadIdx.y;
    if (t >= n_tokens || lane >= 32u) return;
    const float *prob = probs + (uint64_t)t * N_EXPERT;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    int32_t exact[DS4_ROCM_N_EXPERT_USED];
    #pragma unroll
    for (uint32_t k = 0; k < DS4_ROCM_N_EXPERT_USED; k++) exact[k] = k < n_expert_used ? sel[k] : -1;
    float local_prob[N_EXPERT / 32u];
    float local_score[N_EXPERT / 32u];
    float vmax = -INFINITY, vmin = INFINITY;
    #pragma unroll
    for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = prob[e];
        const float c = p + bias[e];
        local_prob[j] = p;
        local_score[j] = c;
        vmax = fmaxf(vmax, c);
        vmin = fminf(vmin, c);
    }
    #pragma unroll
    for (uint32_t m = 16u; m > 0u; m >>= 1u) {
        vmax = fmaxf(vmax, __shfl_xor_sync(FULL_WARP_MASK, vmax, m));
        vmin = fminf(vmin, __shfl_xor_sync(FULL_WARP_MASK, vmin, m));
    }
    const float r_obs = vmax - vmin;
    const float rb = rbar[layer];
    const float bonus = lambda * (rb > 0.0f ? rb : r_obs);
    const unsigned char *lmask = mask + (uint64_t)layer * N_EXPERT;
    #pragma unroll
    for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
        bool favoured = lmask[e] != 0;
        for (uint32_t k = 0; k < keep && k < n_expert_used; k++)
            if (exact[k] == (int32_t)e) favoured = true;
        if (favoured) local_score[j] += bonus;
    }
    float out_prob[DS4_ROCM_N_EXPERT_USED] = {0.0f};
    uint32_t out_idx[DS4_ROCM_N_EXPERT_USED] = {0};
    for (uint32_t k = 0; k < n_expert_used; k++) {
        float best_score = -INFINITY, best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            if (router_score_better(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t m = 16u; m > 0u; m >>= 1u) {
            const float os = __shfl_xor_sync(FULL_WARP_MASK, best_score, m);
            const float op = __shfl_xor_sync(FULL_WARP_MASK, best_prob, m);
            const uint32_t oi = __shfl_xor_sync(FULL_WARP_MASK, best_idx, m);
            if (router_score_better(os, oi, best_score, best_idx)) {
                best_score = os; best_prob = op; best_idx = oi;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++)
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        out_idx[k] = best_idx;
        out_prob[k] = best_prob;
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < n_expert_used; k++) sum += out_prob[k];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < n_expert_used; k++) {
            sel[k] = (int32_t)out_idx[k];
            w[k] = out_prob[k] / sum * expert_weight_scale;
        }
        /* Stats: which unbiased ranks were dropped, and their weight share. */
        uint32_t dropped = 0;
        float drop_p = 0.0f, exact_p = 0.0f;
        for (uint32_t k = 0; k < n_expert_used; k++) {
            if (exact[k] < 0) continue;
            const float pk = prob[exact[k]];
            exact_p += pk;
            bool kept = false;
            for (uint32_t i = 0; i < n_expert_used; i++) kept = kept || out_idx[i] == (uint32_t)exact[k];
            if (!kept) { dropped |= 1u << k; drop_p += pk; }
        }
        float *st = stats + ((uint64_t)(row_offset + t) * n_layer + layer) * DS4_RP_STAT;
        st[0] = r_obs;
        st[1] = (float)dropped;
        st[2] = exact_p > 0.0f ? drop_p / exact_p : 0.0f;
        st[3] = bonus;
    }
}

/* Re-select rows [0, n_rows) of an already routed layer under the open
 * transaction's snapshot.  row_offset places the rows' stats (a scalar verify
 * round routes one row per step). */
extern "C" int ds4_gpu_route_prior_apply(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint32_t layer, uint32_t n_layer, uint32_t n_rows,
        uint32_t row_offset, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale) {
    const float lambda = ds4_gpu_route_prior_lambda();
    if (!(lambda > 0.0f)) return 1;
    if (!g_rp_mask_dev || n_expert != DS4_ROCM_N_EXPERT || n_expert_used > DS4_ROCM_N_EXPERT_USED ||
        layer >= n_layer || n_layer > g_rp_layers || n_rows == 0 ||
        row_offset + n_rows > DS4_RP_MAX_ROWS || !selected || !weights || !probs ||
        !cuda_tensor_has_elems2(probs, n_rows, n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_rows, n_expert_used, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_rows, n_expert_used, sizeof(float))) return 0;
    if (!cuda_model_range_fits(model_size, bias_offset, n_expert * sizeof(float))) return 0;
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset,
                                                            n_expert * sizeof(float), "router_bias");
    if (!bias) return 0;
    dim3 block(32, 4, 1);
    route_prior_kernel<DS4_ROCM_N_EXPERT><<<(n_rows + 3u) / 4u, block>>>(
        (int32_t *)selected->ptr, (float *)weights->ptr, (const float *)probs->ptr, bias,
        g_rp_mask_dev, g_rp_rbar_dev, g_rp_stats_dev, layer, n_layer, row_offset, n_rows,
        n_expert_used, ds4_gpu_route_prior_keep(), lambda, expert_weight_scale);
    return cuda_ok(cudaGetLastError(), "D16 route prior launch");
}

/* Copy rows [0, n_rows) x n_layer stats (DS4_RP_STAT floats each) to host;
 * synchronous (call after the transaction's work has drained). */
extern "C" int ds4_gpu_route_prior_stats(float *out, uint32_t n_rows, uint32_t n_layer) {
    if (!g_rp_stats_dev || !out || n_rows > DS4_RP_MAX_ROWS || n_layer > g_rp_layers) return 0;
    return cuda_ok(cudaMemcpy(out, g_rp_stats_dev,
                              (size_t)n_rows * n_layer * DS4_RP_STAT * sizeof(float),
                              cudaMemcpyDeviceToHost), "D16 stats readback");
}

extern "C" uint64_t ds4_gpu_stream_batch_miss_count(void) { return g_stream_cache_stats.batch_misses; }
