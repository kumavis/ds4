# kumanix-int2: D4 + C1 + D5 on kumanix-int

Branch `kumanix-int2` lives in worktree `/data/ds4-int2`. It starts from
`kumanix-int` (6e69e59), and each branch came in with its own merge commit:

| merge | branch, commit | conflicts |
|---|---|---|
| b85f666 | d4-cache-memory 44f3310 | none |
| 4b97453 | c1-compute-profile 497af2b | none |
| 28dfbb7 | d5-prefetch at **26f52a4** (not the newer branch tip 4b91934) | ds4.c (2) |

The binary measured below was built from 28dfbb7. The commit that adds this
file and `scripts/int2/` changes no code.

**Bottom line.**
- The build passes, and the model-free tests give the same results as the
  parents.
- All gates pass: build defaults, prefetch on, and everything reverted. DSpark
  `--dspark-verify exact` matches plain bit for bit.
- Plain MXFP4 decode runs 7-8% faster than kumanix-int with the build defaults,
  and 10-11% faster with `DS4_ROCM_D5_PREFETCH=1`.
- DSpark (scheduler old) runs 5-10% faster than kumanix-int's old scheduler.
- No default was changed.

## Merge notes and conflicts

**d4 and c1** merged cleanly:
- D4 changes engine configuration, admission and scheduler code.
- C1 changes the per-layer decode path (profiler marks), the norm and
  HC-projection kernels, and `ds4_rocm.cu`.
- The two never touch the same lines.

**d5 at 26f52a4.** The merge produced two conflicts, both in ds4.c and both a
C1 profiler mark against a D5 hook on the same line:

1. **`ds41_moe_partial`, after `ds41_stream_selected_finish_async`.** C1 put
   `C1F(C1_SEL_WAIT, il)` here; D5 put `d5_prefetch_issue(...)` here.
   - Resolution: keep D5's issue immediately after the demand reads start, as
     on D5, then the C1 mark. D5's order relative to the demand reads is
     unchanged.
   - Effect on the profile: under `DS4_ROCM_C1_PROF` the speculative issue
     counts in the SEL_WAIT segment. Its label already reads "start expert
     loads", and a comment at the site says so.
2. **`ds41_graph_before_attention`, after the attention norm.** C1 put
   `C1F(C1_HC_PRE, il)` here; D5 put its stage-1 probe here (measurement only,
   `DS4_ROCM_D5_PROBE` + `DS4_ROCM_D5_TRACE`).
   - Resolution: the C1 mark first, then the probe. The probe drains the device
     twice, so placing it after the mark keeps that time out of HC_PRE.

**Merge check.** Every line that any of the three branches adds against its
merge base is present in the merged tree. The one exception is the D5 comment
line I extended in conflict 1.

### Interactions checked by reading

**C1 profiler against D5's per-layer code.**
- D5's probe start runs at the top of `ds41_moe_partial`: the next layer's norm
  and router matmul plus an async D2H copy, all on stream 0. It sits before
  `C1F(C1_ROUTER)`, so with the profiler on it is counted in the ROUTER segment.
- D5's adoption waits happen inside the selected load (SEL_WAIT) and the
  overlap wait (C1R `MOE_READ_WAIT`).
- No C1 mark changes D5's ordering. C1's marks are a load and a branch when the
  profiler is off. With `PROF=sync`, the drains also wait for speculative
  uploads, so profiling with prefetch on mixes the two (not run).
- C1's norm kernels are stateless; D5's probe calls `ds41_norm` with another
  layer's weight, which is safe.

**D4 memory accounting against D5's shelf.**
- The shelf is allocated lazily with `cudaMalloc` on the first prefetch, after
  D4 has sized the cache from MemAvailable at engine open. It is therefore not
  in D4's plan.
- At the default size it is 4 x 17.93 MiB = 72 MiB (at most 32 slots, 574 MiB,
  with `DS4_ROCM_D5_SHELF`). D4's fixed 2 GiB runtime/IO allowance covers it.
- D5 reuses the existing read pool's pinned staging.
- Adoption swaps buffers between the shelf and the slot pool, so the total
  stays the same.
- Measured: prefetch lowers `avail_min_mb` by 4-363 MiB (mean 206) across the
  eight paired plain runs, and by 250-290 MiB on DSpark json (see Memory
  below).
- D4 does not charge C1's buffers, but they are host arrays and, only with
  `PROF=events`, hipEvents: negligible.

**D4's admission fix.** It skips the drafter charge in `ds41_memory_admit` when
the drafter is already resident under memory sizing. Neither C1 nor D5 touches
admission, drafter loading or `support_model_loaded`. DSpark runs with and
without prefetch passed admission at about 4260 dynamic slots.

**D4's scheduler default (old)** applies to every DSpark run below. D5's own
DSpark numbers used kumanix-int's online default.

## Runtime switches (defaults unchanged from the branches)

| switch | default | source |
|---|---|---|
| `DS4_ROCM_STREAM_CACHE_SIZING` memory / legacy | memory | d4 |
| `DS4_ROCM_STREAM_MEM_RESERVE_GB` (12..96) | 13 | d4 |
| `DS4_DSPARK_SCHED` old / online | old | d4 (was online on kumanix-int) |
| `DS4_ROCM_C1_PROF` coarse/events/sync | off | c1 |
| `DS4_ROCM_C1_NORM`, `DS4_ROCM_C1_HC_PROJ` | on | c1 |
| `DS4_ROCM_C1_Q8_PIPE` (changes tokens) | off | c1 |
| `DS4_ROCM_D5_PREFETCH` | off | d5 |
| `DS4_ROCM_D5_K` / `_DEPTH` / `_SHELF` / `_INFLIGHT` | 1 / 6 / 4 / 16 | d5 |
| `DS4_ROCM_D5_DIAG`, `_TRACE`, `_PROBE` | off | d5 |
| all kumanix-int switches | as INTEGRATION.md | |

## Build and model-free tests

- `make strix-halo` under the GPU lock: BUILD_EXIT=0 (`/data/ds4-int2/build.log`).
- Tests ran from `scripts/int2/tests.sh` under the lock, with output in
  `/data/ds4-bench/int2-tests.log`.

| test | result |
|---|---|
| test_mxfp4_stream_rows_rocm (D1) | EXIT=0, PASS (0 failing cases) |
| test_d0_verify_rows_rocm (D0) | EXIT=1, "D0 kernel rows: FAIL" (expected) |
| test_mxfp4_overlap_rocm (D2) | EXIT=0, PASS (0 of 44) |
| test_mxfp4_rocm (routed MXFP4 MoE) | EXIT=0, PASS |
| test_d5_prefetch_rocm (D5) | EXIT=0, PASS |
| tools/c1/c1_exact.cpp (C1) | EXIT=0, "C1 EXACT SOME DIFF" (expected) |

**D0.** All 155 EQUAL/DIFF verdict lines are identical to kumanix-int's run
(`/data/ds4-bench/int-tests.log`). It is the same documented failure, not a
regression.

**D5.** The make target's recipe is `DS4_ROCM_D5_PREFETCH=1
./tests/test_d5_prefetch_rocm`. I built the binary with
`scripts/d0/build-incremental.sh` (the strix-halo flags, as kumanix-int built
its tests) and ran that recipe. I did not call `make test-d5-prefetch-rocm`,
because a plain `make` could rebuild shared objects with different flags.

**C1.** Built with the flags in its header. Its verdicts and differing-element
counts are identical to C1's own final run (`/data/ds4-bench/c1-exact3.log`):
- The shipped kernels are EQUAL on every seed: the plain norm preload, the
  bf16 weighted-norm preload, and HC projection step 128.
- The DIFF lines are the rejected candidates: the 16-thread sinkhorn (iters 7
  and 20) and the Q8 pipe4 (`DS4_ROCM_C1_Q8_PIPE`, default off).
- c1_exact prints "SOME DIFF" but always exits 0.

## Gates (all pass)

Each digest is the md5 of `"N ID"` over the `perf token` lines (first 12 hex
characters), followed by the line count. All runs set `DS4_TOKEN_TIMING=1` and
`DS4_D8_STEP_TRACE=1`. The log is `/data/ds4-bench/int2-queue.log`.

| tag | switches | measured | reference | decode tok/s | |
|---|---|---|---|---:|---|
| int2-smoke-mx-8 | defaults | b1cd778da121 / 7 | b1cd778da121 / 7 | 9.09 | pass |
| int2-smoke-mx-64 | defaults | 3098d05a1cef / 63 | 3098d05a1cef / 63 | 8.95 | pass |
| int2-smoke-mx-pf-64 | prefetch | 3098d05a1cef / 63 | 3098d05a1cef / 63 | 9.40 | pass |
| int2-gate-mx-code (`--nothink`, logits log) | defaults | 8290541d324b / 511 | 8290541d324b / 511 | 9.15 | pass |
| int2-gate-mx-prose | defaults | 88ed0e11adfd / 511 | 88ed0e11adfd / 511 | 9.83 | pass |
| int2-gate-mx-hash | defaults | 4c37f5fee667 / 410 | 4c37f5fee667 / 410 | 9.09 | pass |
| int2-gate-mx-long (256) | defaults | c27a3b087fb5 / 255 | c27a3b087fb5 / 255 | 8.64 | pass |
| int2-gate-q2-code (`--nothink`) | defaults | f0a2cd53a247 / 511 | f0a2cd53a247 / 511 | 12.12 | pass |
| int2-gate-pf-mx-code | `DS4_ROCM_D5_PREFETCH=1` | 8290541d324b / 511 | 8290541d324b / 511 | 9.37 | pass |
| int2-gate-pf-mx-prose | prefetch | 88ed0e11adfd / 511 | 88ed0e11adfd / 511 | 10.04 | pass |
| int2-gate-pf-mx-hash | prefetch | 4c37f5fee667 / 410 | 4c37f5fee667 / 410 | 9.52 | pass |
| int2-gate-pf-mx-long | prefetch | c27a3b087fb5 / 255 | c27a3b087fb5 / 255 | 8.89 | pass |
| int2-gate-pf-q2-code | prefetch (does not engage on Q2) | f0a2cd53a247 / 511 | f0a2cd53a247 / 511 | 12.05 | pass |
| int2-gate-rev-mx-code | legacy sizing, C1_NORM=0, C1_HC_PROJ=0 | 8290541d324b / 511 | 8290541d324b / 511 | **8.45** | pass |
| int2-gate-rev-mx-prose | reverted | 88ed0e11adfd / 511 | 88ed0e11adfd / 511 | **9.14** | pass |

The reverted pair reproduces kumanix-int's speed: code 8.45 (int: 8.45-8.48)
and prose 9.14 (int: 9.12-9.17), at the legacy cache size (MemAvailable
minimum about 23.7 GB, as before).

**DSpark exactness** (MXFP4 code, `--dspark --dspark-verify exact`, reference
int2-gate-mx-code with its logits log, scheduler old):

| tag | mode | div_first_div | common prefix / compared | tf_agree | flips | logit maxabs | bitexact frac | rounds / accepted |
|---|---|---|---|---:|---:|---:|---:|---|
| int2-gate-mx-code-exact | free (`DS4_REF_TAG`, as kumanix-int) | none | 511 / 511 | 1.0 | 0 | 0.0 | 1.0 | 121 / 391 |
| int2-gate-mx-code-exact-tf | teacher-forced (`DS4_REF_FORCE=1`) | none | 511 / 511 | 1.0 | 0 | 0.0 | 1.0 | 121 / 391 |

Both give the same tokens and bit-identical logits at every position. These ran
without prefetch.

## Measurements

These come from `scripts/int2/measure.sh` (log section "MEASURE QUEUE start"
to "QUEUE DONE"); every number is from `results.jsonl`.
- All runs: `DS4_DIR=/data/ds4-int2`, `DS4_TOKEN_TIMING=1`,
  `DS4_D8_STEP_TRACE=1`, `--rocm --ssd-streaming --temp 0`.
- MXFP4 runs use two drives (replica).
- DSpark runs add `DS4_ROCM_DSPARK_PHASE_TRACE=1` and `DS4_DSPARK_SCHED_TRACE=1`.
- `--nothink` is used on code only.
- No measurement run writes a logits log. kumanix-int's DSpark runs did (they
  carried `DS4_REF_TAG`), which is a small confound against them.

### Summary: decode tok/s, mean of two runs (single run where noted)

| line | kumanix-int | int2 defaults | vs int | int2 + prefetch | prefetch gain | vs int |
|---|---:|---:|---:|---:|---:|---:|
| suite MXFP4 decode (hash) | 8.60 | 9.19 (9.14, 9.24) | +6.9% | | | |
| suite MXFP4 long | 8.17 | 8.76 (8.80, 8.72) | +7.2% | | | |
| suite Q2 decode | 11.05 | 11.28 (11.27, 11.29) | +2.1% | | | |
| suite Q2 long | 11.63 | 12.48 (12.43, 12.52) | +7.3% | | | |
| MXFP4 plain code | 8.45 | 9.15 (9.13, 9.17) | +8.3% | 9.37 (9.38, 9.36) | **+2.4%** | +10.9% |
| MXFP4 plain prose | 9.14 | 9.81 (9.80, 9.81) | +7.3% | 10.04 (10.04, 10.03) | **+2.3%** | +9.8% |
| MXFP4 plain hash | 8.58 (gate) | 9.12 (9.12, 9.11) | +6.2% | 9.48 (9.48, 9.48) | **+4.0%** | +10.5% |
| MXFP4 plain long | 8.21 (gate) | 8.74 (8.79, 8.68) | +6.4% | 9.04 (9.05, 9.02) | **+3.4%** | +10.0% |
| MXFP4 DSpark code | 10.89 old / 11.42 online | 11.90 (11.83, 11.96) | +9.2% / +4.2% | 12.01 (single) | +1.0% | |
| MXFP4 DSpark prose | 8.93 / 8.94 | 9.74 (9.72, 9.75) | +9.0% | | | |
| MXFP4 DSpark json | 11.18 / 10.88 | 11.79 (11.68, 11.90) | +5.5% | 11.99 (single) | +1.7% | |
| MXFP4 DSpark translate | 11.77 / 10.07 | 12.66 (12.65, 12.66) | +7.5% | | | |
| MXFP4 DSpark list | 9.63 / 9.35 | 10.57 (10.54, 10.60) | +9.8% | | | |
| Q2 plain code (single) | 11.77 | 12.17 | +3.4% | | | |
| Q2 DSpark code (single) | 14.26 old / 14.63 online | 14.55 | +2.0% / -0.5% | | | |

How each "vs int" is computed:
- The int2 DSpark columns compare against kumanix-int's old scheduler, today's
  default.
- The int plain hash and long values are kumanix-int's gate runs (it ran no
  measurement pair for them).

**DSpark rounds / accepted drafts are identical to kumanix-int's old-scheduler
runs** on every prompt:

| prompt | rounds / accepted |
|---|---|
| code | 172 / 340 |
| prose | 435 / 77 |
| json | 207 / 305 |
| translate | 176 / 336 |
| list | 403 / 109 |
| Q2 code | 170 / 342 |

So the speedup is per-step speed, not different drafting.

**Does the prefetch gain survive the bigger cache?** Yes:
- It measures +2.3% to +4.0% (code +2.4, prose +2.3, hash +4.0, long +3.4).
  kumanix-int's smaller cache gave +3.2 / +3.3 / +3.8 / +2.6 (D5.md).
- Code and prose lose about 1 point. The memory-sized cache leaves fewer misses
  to hide: code misses fall from 39.1 to 34.1 per token on this table's
  measure.
- Hash and long gain slightly more.
- D5 counters on int2-mx-code-pf-r1:
  - 14004 speculative experts issued, 6305 adopted (4302 ready, 2003 in flight),
    7699 dropped;
  - 245 GiB of speculative reads; 417 ms of adoption waits in all.
- DSpark with prefetch: +1.0% on code and +1.7% on json, one run each.

**Do D4 + C1 + D5 add up?** Roughly, yes, multiplicatively. The predictions
below are estimates: products of each branch's own measured gain over
kumanix-int-era baselines.

| line | D4 | C1 | D5 | predicted | measured |
|---|---|---|---|---|---|
| code | +6.1% | +1.7% | +3.2% | 9.41 | 9.37 (-0.4%) |
| prose | +5.2% | +2.0% | +3.3% | 10.13 | 10.04 (-1%) |
| hash | +5.7% | not measured | +3.8% | 9.43 | 9.48 |
| long | +6.0% | not measured | +2.6% | 8.93 | 9.04 |

- **D4 + C1 alone** come out as predicted. Code: 8.45 x 1.061 x 1.017 = 9.12,
  measured 9.15. Prose: 9.14 x 1.052 x 1.020 = 9.81, measured 9.81.
- **D5 on top** loses a little on code and prose and gains a little on hash and
  long. That matches its gain moving with the miss count.

### Memory and temperature

**MemAvailable.** The lowest `avail_min_mb` was **12787 MiB**
(int2-mx-json-dsp-pf, DSpark json with prefetch). It stays above the 12288 MiB
floor, but only by 499 MiB. Next lowest:
- int2-mx-translate-dsp-r1: 12823
- int2-mx-json-dsp-r1 and -r2: 13035 and 13080

D4's claim that the MemAvailable minimum sits about 0.6 GiB above the 13 GiB
reserve (13312 MiB) holds for the plain runs (13228-14105) and for DSpark code,
prose and list. It does not hold for DSpark json and translate:
- Those dip up to about 0.5 GiB below the reserve, without prefetch.
- D4 measured DSpark only on code, prose and list.
- Prefetch adds about 250 MiB more on json.

No run went below 12 GiB, so nothing was stopped.

**Temperature.** The highest `nvme_temp_max_c` was **84 C**
(int2-mx-translate-dsp-r1); json and translate DSpark runs reached 83. That is
below the 89 C throttle point. Prefetch runs ran about 3-5 C hotter than their
plain pairs (79-81 against 75-80).

### All runs (`scripts/int2/table.py int2- suite-int2`)

- misses/tok for plain runs is decode_misses / ntok_max.
- For DSpark runs, prefill misses are first subtracted using the matching
  plain run (as kumanix-int's table.py). It is "-" for json, translate and list,
  which had no plain run in this queue.
- dyn slots is the memory-sized cache's dynamic slot count from the run log;
  "-" means legacy sizing.
- On the smoke runs, misses/tok is dominated by prefill.

| tag | prefill tok/s | decode tok/s | misses/tok | dyn slots | DSpark rounds / accepted | avail min MiB | NVMe max C | exit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int2-smoke-mx-8 | 5.82 | 9.09 | 703.0 | 4740 | - | 13757 | 73 | 0 |
| int2-smoke-mx-64 | 5.71 | 8.95 | 109.2 | 4715 | - | 14181 | 74 | 0 |
| int2-smoke-mx-pf-64 | 6.33 | 9.40 | 109.1 | 4727 | - | 13786 | 77 | 0 |
| int2-smoke-mx-dsp-64 | 5.73 | 10.14 | - | 4270 | 18 / 46 | 13340 | 75 | 0 |
| int2-smoke-q2-64 | 6.06 | 10.20 | 100.0 | 9613 | - | 45378 | 72 | 0 |
| int2-gate-mx-code | 5.89 | 9.15 | 34.1 | 4736 | - | 13727 | 72 | 0 |
| int2-gate-mx-prose | 5.38 | 9.83 | 29.6 | 4735 | - | 13764 | 75 | 0 |
| int2-gate-mx-hash | 4.74 | 9.09 | 26.7 | 4737 | - | 13735 | 75 | 0 |
| int2-gate-mx-long | 42.28 | 8.64 | 24.5 | 4739 | - | 13241 | 81 | 0 |
| int2-gate-q2-code | 6.07 | 12.12 | 19.2 | 9608 | - | 13960 | 72 | 0 |
| int2-gate-pf-mx-code | 6.33 | 9.37 | 34.1 | 4737 | - | 13653 | 80 | 0 |
| int2-gate-pf-mx-prose | 5.86 | 10.04 | 29.7 | 4729 | - | 13827 | 79 | 0 |
| int2-gate-pf-mx-hash | 5.08 | 9.52 | 26.7 | 4725 | - | 13909 | 75 | 0 |
| int2-gate-pf-mx-long | 42.32 | 8.89 | 24.7 | 4727 | - | 13228 | 69 | 0 |
| int2-gate-pf-q2-code | 6.02 | 12.05 | 19.2 | 9618 | - | 13942 | 53 | 0 |
| int2-gate-rev-mx-code | 5.78 | 8.45 | 39.1 | - | - | 23694 | 67 | 0 |
| int2-gate-rev-mx-prose | 5.38 | 9.14 | 33.5 | - | - | 23679 | 73 | 0 |
| int2-gate-mx-code-exact | 5.72 | 7.59 | 39.9 | 4263 | 121 / 391 | 13573 | 74 | 0 |
| int2-gate-mx-code-exact-tf | 5.85 | 7.63 | 40.0 | 4258 | 121 / 391 | 13566 | 75 | 0 |
| suite-int2-mxfp4-decode | 4.76 | 9.14 | 26.7 | 4740 | - | 13744 | 76 | 0 |
| suite-int2-mxfp4-long | 41.40 | 8.80 | 24.5 | 4739 | - | 13258 | 79 | 0 |
| suite-int2-q2-decode | 4.94 | 11.27 | 17.9 | 9627 | - | 17898 | 74 | 0 |
| suite-int2-q2-long | 49.76 | 12.43 | 6.8 | 9615 | - | 13405 | 79 | 0 |
| suite-int2r2-mxfp4-decode | 4.72 | 9.24 | 26.7 | 4738 | - | 13737 | 73 | 0 |
| suite-int2r2-mxfp4-long | 41.60 | 8.72 | 24.5 | 4734 | - | 13389 | 79 | 0 |
| suite-int2r2-q2-decode | 4.98 | 11.29 | 17.9 | 9616 | - | 17897 | 72 | 0 |
| suite-int2r2-q2-long | 51.12 | 12.52 | 6.9 | 9611 | - | 13432 | 80 | 0 |
| int2-mx-code-plain-r1 | 5.86 | 9.13 | 34.2 | 4716 | - | 14024 | 75 | 0 |
| int2-mx-code-pf-r1 | 6.23 | 9.38 | 34.1 | 4738 | - | 13661 | 80 | 0 |
| int2-mx-prose-plain-r1 | 5.38 | 9.80 | 29.7 | 4719 | - | 14105 | 75 | 0 |
| int2-mx-prose-pf-r1 | 5.80 | 10.04 | 29.7 | 4729 | - | 13858 | 79 | 0 |
| int2-mx-hash-plain-r1 | 4.74 | 9.12 | 26.7 | 4730 | - | 13901 | 75 | 0 |
| int2-mx-hash-pf-r1 | 5.09 | 9.48 | 26.6 | 4742 | - | 13623 | 79 | 0 |
| int2-mx-long-plain-r1 | 41.81 | 8.79 | 24.7 | 4717 | - | 13688 | 79 | 0 |
| int2-mx-long-pf-r1 | 42.57 | 9.05 | 24.5 | 4731 | - | 13364 | 81 | 0 |
| int2-mx-code-plain-r2 | 5.86 | 9.17 | 34.1 | 4731 | - | 13876 | 77 | 0 |
| int2-mx-code-pf-r2 | 6.31 | 9.36 | 34.1 | 4727 | - | 13872 | 80 | 0 |
| int2-mx-prose-plain-r2 | 5.37 | 9.81 | 29.7 | 4726 | - | 13951 | 75 | 0 |
| int2-mx-prose-pf-r2 | 5.70 | 10.03 | 29.6 | 4739 | - | 13648 | 79 | 0 |
| int2-mx-hash-plain-r2 | 4.75 | 9.11 | 26.7 | 4731 | - | 13866 | 75 | 0 |
| int2-mx-hash-pf-r2 | 5.09 | 9.48 | 26.7 | 4730 | - | 13755 | 79 | 0 |
| int2-mx-long-plain-r2 | 42.13 | 8.68 | 24.6 | 4729 | - | 13440 | 80 | 0 |
| int2-mx-long-pf-r2 | 41.92 | 9.02 | 24.7 | 4726 | - | 13420 | 80 | 0 |
| int2-mx-code-dsp-r1 | 5.85 | 11.83 | 40.3 | 4265 | 172 / 340 | 13364 | 80 | 0 |
| int2-mx-prose-dsp-r1 | 5.41 | 9.72 | 32.7 | 4267 | 435 / 77 | 13305 | 75 | 0 |
| int2-mx-json-dsp-r1 | 11.38 | 11.68 | - | 4260 | 207 / 305 | 13035 | 83 | 0 |
| int2-mx-translate-dsp-r1 | 11.76 | 12.65 | - | 4270 | 176 / 336 | 12823 | 84 | 0 |
| int2-mx-list-dsp-r1 | 6.12 | 10.54 | - | 4260 | 403 / 109 | 13429 | 75 | 0 |
| int2-mx-code-dsp-r2 | 5.85 | 11.96 | 40.4 | 4260 | 172 / 340 | 13455 | 76 | 0 |
| int2-mx-prose-dsp-r2 | 5.40 | 9.75 | 32.7 | 4260 | 435 / 77 | 13417 | 75 | 0 |
| int2-mx-json-dsp-r2 | 11.30 | 11.90 | - | 4261 | 207 / 305 | 13080 | 81 | 0 |
| int2-mx-translate-dsp-r2 | 11.70 | 12.66 | - | 4249 | 176 / 336 | 13257 | 83 | 0 |
| int2-mx-list-dsp-r2 | 6.16 | 10.60 | - | 4270 | 403 / 109 | 13309 | 77 | 0 |
| int2-mx-code-dsp-pf | 6.30 | 12.01 | 40.3 | 4263 | 172 / 340 | 13356 | 79 | 0 |
| int2-mx-json-dsp-pf | 11.33 | 11.99 | - | 4273 | 207 / 305 | 12787 | 83 | 0 |
| int2-q2-code-plain | 6.10 | 12.17 | 19.2 | 9616 | - | 13937 | 71 | 0 |
| int2-q2-code-dsp | 6.07 | 14.55 | 19.6 | 8751 | 170 / 342 | 13336 | 74 | 0 |

Q2 is now memory-sized too: about 9600 dynamic slots. The smoke run's high
MemAvailable minimum (45 GB at 64 tokens) is a short run that never filled the
cache.

## Prefetch default: recommendation (decision is the PM's)

I would turn `DS4_ROCM_D5_PREFETCH=1` on by default, but only after one more
gate: DSpark `--dspark-verify exact` with prefetch on.

**For turning it on:**
- Token IDs are identical on all five gates and both smoke runs.
- It gives +2.3% to +4.0% on plain MXFP4 decode with the bigger cache, and
  +1.0 / +1.7% on DSpark code / json (single runs).
- It does nothing on Q2, as designed (12.05 vs 12.12, noise).

**Against, or to watch:**
- About 245 GiB of extra SSD reads per 512-token code run (roughly half of
  them wasted), and NVMe runs 3-5 C hotter.
- Up to about 370 MiB less MemAvailable (mean 206 MiB on the plain pairs). The worst run (DSpark json) ended at
  12787 MiB, 0.5 GiB above the 12 GiB floor.
- D5 is still under development (the branch has moved on past 26f52a4).
- Exactness under DSpark with prefetch was not gated here.

If it goes on, I would also charge the shelf in D4's fixed memory term, rather
than relying on the 2 GiB allowance.

## Not run or not verified

- The DSpark exact gate with prefetch on.
- C1's profiler (`DS4_ROCM_C1_PROF`) on the merged binary, and the profiler
  with prefetch on.
- D5's stage-1 probe and trace (`DS4_ROCM_D5_PROBE` / `_TRACE`).
- No DSpark divergence or teacher-forced quality runs except the exact gate.
- DSpark online scheduler: not measured (the default is now old).
- Prefetch on DSpark prose, translate and list.
- Plain json, translate and list (so no misses/tok for those DSpark runs).
- The D5 test was built with the incremental strix-halo build and run with the
  make target's recipe, not via `make test-d5-prefetch-rocm` itself.
- The suite ran twice (labels `int2`, `int2r2`); kumanix-int ran it once.
