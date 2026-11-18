"""D17 part B (c): the token-major threshold as a setting.

DS4_ROCM_V41_TOKEN_MAJOR_MAX=N (N >= 1): on ROCm, an append of fewer than N
rows is prefilled token by token (decode steps on the warm cache) instead of a
layer sweep. Unset: ROCm's built-in 256 (32 for two-machine TP), unchanged.
N = 1 means always sweep. Token-major steps use the decode kernels, so a
different N can change the output (output class approx-*, to be measured)."""
p = "/data/ds4-d17b/ds4.c"
s = open(p).read()
a = """#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    minimum = g->tp_world == 2 &&
        !getenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL") ? 32u : 256u;
#endif"""
b = """#if defined(__APPLE__) || defined(DS4_ROCM_BUILD)
    minimum = g->tp_world == 2 &&
        !getenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL") ? 32u : 256u;
#endif
#ifdef DS4_ROCM_BUILD
    {
        /* D17: measured token-major / sweep crossover (see D17.md). */
        static int tm = -1;
        if (tm < 0) {
            const char *e = getenv("DS4_ROCM_V41_TOKEN_MAJOR_MAX");
            tm = e && e[0] ? atoi(e) : 0;
            if (tm < 0) tm = 0;
        }
        if (tm > 0 && g->tp_world != 2) minimum = (uint32_t)tm;
    }
#endif"""
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w").write(s)
print("patched")
