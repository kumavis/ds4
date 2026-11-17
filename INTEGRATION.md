# kumanix-int: D0 + D1 + D2 + D3 + D8 integration

Branch `kumanix-int` in worktree `/data/ds4-int`, started from
`d1-mxfp4-dspark` (1341ab3) and merged, in this order, with real merge
commits (history kept):

| merge | branch tip | conflicts |
|---|---|---|
| 840cd0b | d0-exactness 57a6605 | ds4.c (4), rocm/ds4_rocm_moe_launch.cuh (2), Makefile (1) |
| ec1d08f | d2-overlap-reads 92d8c21 | Makefile (1) |
| 53b4cd4 | d8-drafter-overhead 1cd562c | ds4.c (2) |
| 2584711 | d3-dspark-scheduler 2b3271d | ds4.c (1) |

Then 948859a adds the integration queue scripts (`scripts/int/`). The
binary measured below was built from 2584711. The code in 948859a and in the
commit that adds this file is the same.

## Conflict resolutions

**d0 into d1.** d0 has its own copy of the dspark-study commit (869ec47, the
same patch as d1's e9d234b). All seven conflicts were that study switch
(`DS4_ROCM_DSPARK_SSD_EXPERIMENT=1`, off by default) against d1's renamed,
default-on `DS4_ROCM_DSPARK_SSD` gate. I kept d1's version everywhere
(`ds41_dspark_ssd_enabled` at its three call sites). The 3-way merge also
silently re-added the study's inline MXFP4 batch gate/up and down launches in
`routed_moe_launch`, which D1 had moved into
`routed_moe_mxfp4_ptrs_rows_launch`. d0 made no other change to
`rocm/ds4_rocm_moe_launch.cuh`, so I restored that file to d1's version.
Check: the merged ds4.c diff against d1 equals d0's diff against 869ec47,
line for line. The d0 scripts still export `DS4_ROCM_DSPARK_SSD_EXPERIMENT=1`;
nothing reads that variable any more, so it does nothing.

**d2.** Only the Makefile conflicted, and both test targets are kept. D2's
decode overlap (`ds41_moe_partial`) and few-row verify overlap
(`ds41_moe_batch` -> `ds41_moe_batch_d2`) sit below D0's verifier dispatch.
D0's `exact` setting runs one-token steps, so it takes the same D2 decode
path as plain decode. Check: the tree's diff against d2-overlap-reads is
exactly d0's change set.

**d8.** There were two conflicts, both with D0:
1. After `ds41_bf16`, D0's harness block and D8's drafter F16 helpers are
   independent, so I kept both.
2. D0 split `ds41_draft_verify` into `_scalar`, `_batch`, `_shadow` and a
   dispatcher. D8 had put a `ds41_draft_flush` (the deferred drafter
   bookkeeping, `DS4_D8_LAZY_PENDING`) at the top of the old function. The
   flush now sits in the dispatcher, after the argument checks and before the
   shadow, exact or batch choice, so every verify setting sees a projected
   drafter ring.

Check: the ds4.c diff of the merge equals d8's diff against d1, plus the
moved flush.

**d3.** There was one conflict, in `ds41_session_spec`'s draft-or-plain
decision. D3 replaced the `ds41_draft_should_propose` call with the online
scheduler and moved the old call into an else branch. D8 had wrapped that
same call in its `DS4_D8_NO_PROPOSE` diagnostic. Resolution: D3's structure,
with `DS4_D8_NO_PROPOSE` honoured in both branches. Under online it never
calls `ds41_sched_decide`; the round becomes a plain step with
why=d8_no_propose and no scheduler_skips count.

## Runtime switches and defaults (unchanged from the branches)

| switch | default | source |
|---|---|---|
| `DS4_ROCM_DSPARK_SSD=0` | on (DSpark with SSD streaming) | d1 |
| `DS4_ROCM_STREAM_REPLICA_POLICY` | chunk | d1 |
| `DS4_ROCM_MXFP4_STREAM_DOWN_PER_ROW` | 0 | d1 |
| `--dspark-verify` / `DS4_DSPARK_VERIFY` fast,attn,attn-moe,exact | fast (`DS4_ROCM_DSPARK_SCALAR_VERIFY` = exact) | d0 |
| `DS4_D0_LOGITS`, `DS4_D0_FORCE`, `DS4_D0_SHADOW`, `_TRACE`, `DS4_D0_LOGITS_MAXPOS` | off | d0 |
| `DS4_ROCM_D2_DECODE` (0, readback, on) | on | d2 |
| `DS4_ROCM_D2_VERIFY=0` | on | d2 |
| `DS4_D8_DRAFT_F16=0` | on | d8 |
| `DS4_D8_LAZY_PENDING` 0,1,2 | 0 | d8 |
| `DS4_D8_STEP_TRACE`, `DS4_D8_PROPOSE_DETAIL`, `DS4_D8_NO_PROPOSE` | off | d8 |
| `DS4_DSPARK_SCHED` online or old | online (as d3 ships; old still selectable) | d3 |
| `DS4_DSPARK_SCHED_{HALFLIFE,EXPLORE,MARGIN,MAX_WIDTH,WARM,MOMENTUM,FORCE,PRINT,TRACE}` | as d3 | d3 |

The tests and scripts of every branch are present: `scripts/d0`, `d1`,
`d2`, `d3`, and the tests `test_d0_verify_rows_rocm.c`,
`test_mxfp4_stream_rows_rocm.c` and `test_mxfp4_overlap_rocm.c` with their
Makefile targets. The d8 queue scripts were never committed on d8 (they live
in `/data/ds4-d8/d8/`), so there is nothing of d8's to carry over.

## Build and model-free tests

`make strix-halo` under the GPU lock: BUILD_EXIT=0 (`/data/ds4-int/build.log`).
The tests ran under the lock (`scripts/int/tests.sh`, output in
`/data/ds4-bench/int-tests.log`):

| test | result |
|---|---|
| test_mxfp4_stream_rows_rocm (D1) | EXIT=0, PASS (0 failing cases) |
| test_mxfp4_overlap_rocm (D2) | EXIT=0, PASS (0 of 44 cases failing) |
| test_mxfp4_rocm (routed MXFP4 MoE) | EXIT=0, PASS |
| test_d0_verify_rows_rocm (D0) | EXIT=1, "D0 kernel rows: FAIL" |

The D0 test checks whether the fast verifier's multi-row kernels are bitwise
equal to one-token decode. It fails exactly as it does on d0-exactness itself
(`/data/ds4-d0/d0-ktest.log`, 22:34): all 75 "exact" lines have the same
EQUAL/DIFF verdicts. The DIFF lines are q8 rows q_a and head slice at rows 2-6,
attention output A+B at rows 2-6, and MXFP4 routed verify-mode at rows 5-6.
This is the documented reason `fast` is not exact, not a regression.

## Correctness gate (all pass)

Each digest is the md5 of `"N ID"` over the `ds4: perf token` lines (first 12
hex characters), followed by the line count. Runs use default settings and
`DS4_TOKEN_TIMING=1`. Results are in `/data/ds4-bench/int-queue.log`.

| run | tokens | measured | reference | |
|---|---|---|---|---|
| int-gate-mx-code (`--nothink`) | 512 | 8290541d324b / 511 | 8290541d324b / 511 | pass |
| int-gate-mx-prose | 512 | 88ed0e11adfd / 511 | 88ed0e11adfd / 511 | pass |
| int-gate-mx-hash | 512 | 4c37f5fee667 / 410 | 4c37f5fee667 / 410 | pass |
| int-gate-mx-long | 256 | c27a3b087fb5 / 255 | c27a3b087fb5 / 255 | pass |
| int-gate-q2-code (`--nothink`) | 512 | f0a2cd53a247 / 511 | f0a2cd53a247 / 511 | pass |
| int-gate-mx-code-d2off (`DS4_ROCM_D2_DECODE=0 DS4_ROCM_D2_VERIFY=0`) | 512 | 8290541d324b / 511 | 8290541d324b / 511 | pass |

The measurement plain runs reproduce these too: int-mx-code-plain-r1/r2 give
8290541d324b / 511, int-mx-prose-plain-r1/r2 give 88ed0e11adfd / 511, and
int-q2-code-plain gives f0a2cd53a247 / 511.

**DSpark `--dspark-verify exact`, MXFP4 code** (int-gate-mx-code-exact). This
was a free run, not teacher-forced, with DS4_REF_TAG=int-gate-mx-code. Under
DSpark the run log has no perf token lines (digest 68b329da9893 / 0 is the
empty input), so the check uses the logits-log comparison (diverge.py) in the
record:

- div_first_div null and div_common_prefix 511 of div_compared 511
- div_tf_agree 1.0, div_tf_flips 0
- div_logit_maxabs 0.0, div_bitexact_frac 1.0

The tokens are the same as plain's, and so are the logits, bit for bit, at
every position. The verify setting printed is "exact". 447 rounds and 65
accepted drafts: the online scheduler kept most rounds plain, because exact
verification costs one decode step per row.

## Measurements

These ran under `scripts/int/measure.sh`, log `/data/ds4-bench/int-queue.log`
from "MEASURE QUEUE start" to "QUEUE DONE". All runs: `DS4_DIR=/data/ds4-int`,
`DS4_TOKEN_TIMING=1`, `DS4_D8_STEP_TRACE=1`, the default automatic stream
cache (as d8 queue B), `--rocm --ssd-streaming --temp 0`, and two drives on
MXFP4 (chunk policy). DSpark runs add `DS4_ROCM_DSPARK_PHASE_TRACE=1` and
`DS4_DSPARK_SCHED_TRACE=1`. Plain MXFP4 runs in (b) write
`DS4_D0_LOGITS=auto`. `--nothink` is used on code only.

### Summary: decode tok/s, mean of r1 and r2 (single run where noted)

| line | int plain | int DSpark old | int DSpark online | best single branch tonight |
|---|---:|---:|---:|---|
| MXFP4 code | 8.45 (8.47, 8.43) | 10.89 (10.88, 10.90) | 11.42 (11.51, 11.32) | plain 8.48 (d2); DSpark 9.26 (d8) |
| MXFP4 prose | 9.14 (9.13, 9.15) | 8.93 (8.93, 8.92) | 8.94 (8.98, 8.90) | plain 8.68 (9.13 on d2); DSpark 8.29 (d8) |
| MXFP4 json | 8.95 (8.96, 8.93) | 11.18 (11.13, 11.23) | 10.88 (10.80, 10.95) | none (d3 ran json with `--nothink`) |
| MXFP4 translate | 9.02 (9.03, 9.01) | 11.77 (11.69, 11.84) | 10.07 (10.13, 10.01) | none (d3 used `--nothink`) |
| MXFP4 list | 9.60 (9.60, 9.59) | 9.63 (9.61, 9.64) | 9.35 (8.97, 9.73) | none (d3 used `--nothink`) |
| Q2 code (single) | 11.77 | 14.26 | 14.63 | plain 11.77; DSpark 14.27 (d8) |
| suite MXFP4 decode / long | 8.60 / 8.17 | | | |
| suite Q2 decode / long | 11.05 / 11.63 | | | |

**Against the best single branch.** Every line is at least as fast except
MXFP4 plain code: 8.45 against d2's 8.48 (8.47 and 8.49). Run 1 matched
exactly (8.47); run 2 was 0.06 lower. I traced it with the per-token timing.
The summed token ms are 60445 and 60717 here against 60443 and 60282 on d2;
misses per token are identical (39.1) and the hit pattern is the same. The
whole difference is read wait (26287 and 26533 ms against 26320 and 26176).

These int plain runs write the D0 logits log (about 0.5 MB per token, 260 MB
per run) to /data, the same drive that holds the primary model. The int
plain runs that write no log match their branch exactly:

- int-gate-mx-prose: 9.14, against d2's 9.13
- int-gate-mx-code-d2off: 7.98 (summed 64122 ms), against d2 old's 7.98 (64150 ms)

So I read the gap as logits-log IO plus noise, not a merge regression.

DSpark code is 2.2 tok/s above the best branch (d8's 9.26). D2's verify
overlap and D8's drafter fast path stack.

**Scheduler (for the default decision).** Online beats old on:

- code: 11.42 vs 10.89
- Q2: 14.63 vs 14.26, one run each

It ties on prose (8.94 vs 8.93).

It loses on:

- translate: 10.07 vs 11.77. The online scheduler drafts almost every round
  (218 of 226) but picks short widths: widths 1:86, 2:69, 3:40, 4:6, 5:3.
  Its verify fit V = 72 + 44 x rows ms makes long widths look expensive.
  Old verifies longer drafts and accepts 336 drafts against 286.
- json: 10.88 vs 11.18.
- list: 9.35 vs 9.63, with r1 at 8.97. In list r1 online ran 215 plain
  steps, with a rate_draft estimate of 8.69 against rate_plain 9.42 tok/s.

That is D3's own policy behaviour, not something the merge changed.

On prose, DSpark (about 8.9) is below plain (9.14) under both schedulers.

**Divergence.** Free-running DSpark (fast verifier) leaves plain at:

- code: position 75
- prose: 108 (old) and 7 (online)
- json: 37
- translate: 32 (old) and 54 (online)
- list: 78 to 109

For free runs, div_tf_agree only covers the common prefix: prose online has
7 positions, so its 0.8571 is not meaningful. The teacher-forced runs (`-tf`,
default verify fast, scheduler online, every position compared) give the
quality numbers:

| prompt | tf_agree |
|---|---:|
| code | 0.9941 |
| prose | 0.9765 |
| json | 0.9824 |
| translate | 0.9883 |
| list | 0.9667 |

All are above D0's 0.95 threshold. Their tok/s is not comparable.

### All runs (from results.jsonl, `scripts/int/table.py`)

misses/tok counts stream-cache misses after prefill per generated token. For
plain runs that is decode_misses / ntok_max. For DSpark runs it is
(decode_misses + prefill_misses - the matching plain run's prefill_misses) /
ntok_max, because the record's prefill_misses counts every multi-row call,
verify rows included. The DSpark column is dspark_cycles / dspark_accepted.
div_first_div is "none" when nothing diverged and "-" when there was no
reference. Q2 DSpark runs had no reference tag.

| tag | prefill tok/s | decode tok/s | misses/tok | DSpark rounds / accepted drafts | div_tf_agree | div_first_div | ssd max C | exit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| int-gate-mx-code | 5.81 | 8.43 | 39.1 | - | - | - | 72 | 0 |
| int-gate-mx-prose | 5.37 | 9.14 | 33.5 | - | - | - | 71 | 0 |
| int-gate-mx-hash | 4.72 | 8.58 | 30.5 | - | - | - | 71 | 0 |
| int-gate-mx-long | 42.06 | 8.21 | 29.3 | - | - | - | 75 | 0 |
| int-gate-mx-code-d2off | 5.51 | 7.98 | 39.1 | - | - | - | 71 | 0 |
| int-gate-mx-code-exact | 5.78 | 7.82 | - | 447 / 65 | 1.0000 | none | 72 | 0 |
| int-gate-q2-code | 6.00 | 11.78 | 19.5 | - | - | - | 72 | 0 |
| suite-int-mxfp4-decode | 4.72 | 8.60 | 30.5 | - | - | - | 70 | 0 |
| suite-int-mxfp4-long | 43.28 | 8.17 | 29.3 | - | - | - | 76 | 0 |
| suite-int-q2-decode | 4.91 | 11.05 | 18.0 | - | - | - | 75 | 0 |
| suite-int-q2-long | 51.68 | 11.63 | 9.6 | - | - | - | 80 | 0 |
| int-mx-code-plain-r1 | 5.82 | 8.47 | 39.1 | - | - | - | 74 | 0 |
| int-mx-code-old-r1 | 5.79 | 10.88 | 46.2 | 172 / 340 | 0.9867 | 75 | 66 | 0 |
| int-mx-code-online-r1 | 5.79 | 11.51 | 45.4 | 136 / 376 | 0.9867 | 75 | 73 | 0 |
| int-mx-prose-plain-r1 | 5.38 | 9.13 | 33.5 | - | - | - | 73 | 0 |
| int-mx-prose-old-r1 | 5.31 | 8.93 | 38.4 | 435 / 77 | 0.9907 | 108 | 75 | 0 |
| int-mx-prose-online-r1 | 5.32 | 8.98 | 39.0 | 406 / 106 | 0.8571 | 7 | 74 | 0 |
| int-mx-json-plain-r1 | 12.03 | 8.96 | 23.0 | - | - | - | 80 | 0 |
| int-mx-json-old-r1 | 11.32 | 11.13 | 26.5 | 207 / 305 | 0.9730 | 37 | 82 | 0 |
| int-mx-json-online-r1 | 11.31 | 10.80 | 25.8 | 207 / 305 | 0.9730 | 37 | 82 | 0 |
| int-mx-translate-plain-r1 | 12.43 | 9.03 | 22.3 | - | - | - | 81 | 0 |
| int-mx-translate-old-r1 | 11.74 | 11.69 | 25.9 | 176 / 336 | 0.9688 | 32 | 81 | 0 |
| int-mx-translate-online-r1 | 11.77 | 10.13 | 28.8 | 226 / 286 | 0.9815 | 54 | 83 | 0 |
| int-mx-list-plain-r1 | 6.08 | 9.60 | 32.0 | - | - | - | 75 | 0 |
| int-mx-list-old-r1 | 5.99 | 9.61 | 37.6 | 403 / 109 | 0.9872 | 78 | 75 | 0 |
| int-mx-list-online-r1 | 6.00 | 8.97 | 42.9 | 372 / 140 | 0.9907 | 107 | 74 | 0 |
| int-mx-code-plain-r2 | 5.81 | 8.43 | 39.1 | - | - | - | 73 | 0 |
| int-mx-code-old-r2 | 5.77 | 10.90 | 46.2 | 172 / 340 | 0.9867 | 75 | 76 | 0 |
| int-mx-code-online-r2 | 5.77 | 11.32 | 45.3 | 145 / 367 | 0.9867 | 75 | 79 | 0 |
| int-mx-prose-plain-r2 | 5.37 | 9.15 | 33.5 | - | - | - | 76 | 0 |
| int-mx-prose-old-r2 | 5.31 | 8.92 | 38.4 | 435 / 77 | 0.9907 | 108 | 75 | 0 |
| int-mx-prose-online-r2 | 5.31 | 8.90 | 40.6 | 386 / 126 | 0.8571 | 7 | 75 | 0 |
| int-mx-json-plain-r2 | 12.02 | 8.93 | 23.0 | - | - | - | 81 | 0 |
| int-mx-json-old-r2 | 11.34 | 11.23 | 26.5 | 207 / 305 | 0.9730 | 37 | 63 | 0 |
| int-mx-json-online-r2 | 11.32 | 10.95 | 25.2 | 206 / 306 | 0.9730 | 37 | 74 | 0 |
| int-mx-translate-plain-r2 | 12.40 | 9.01 | 22.3 | - | - | - | 70 | 0 |
| int-mx-translate-old-r2 | 11.69 | 11.84 | 25.9 | 176 / 336 | 0.9688 | 32 | 78 | 0 |
| int-mx-translate-online-r2 | 11.70 | 10.01 | 28.9 | 228 / 284 | 0.9815 | 54 | 82 | 0 |
| int-mx-list-plain-r2 | 6.07 | 9.59 | 32.0 | - | - | - | 75 | 0 |
| int-mx-list-old-r2 | 5.99 | 9.64 | 37.6 | 403 / 109 | 0.9872 | 78 | 79 | 0 |
| int-mx-list-online-r2 | 6.00 | 9.73 | 39.1 | 275 / 237 | 0.9908 | 109 | 75 | 0 |
| int-mx-code-tf | 5.74 | 11.86 | 44.6 | 134 / 378 | 0.9941 | none | 64 | 0 |
| int-mx-prose-tf | 5.32 | 9.22 | 39.2 | 354 / 158 | 0.9765 | none | 66 | 0 |
| int-mx-json-tf | 11.32 | 11.55 | 25.5 | 179 / 333 | 0.9824 | none | 82 | 0 |
| int-mx-translate-tf | 11.38 | 10.83 | 26.0 | 208 / 304 | 0.9883 | none | 82 | 0 |
| int-mx-list-tf | 5.99 | 9.93 | 38.0 | 284 / 228 | 0.9667 | none | 75 | 0 |
| int-q2-code-plain | 6.00 | 11.77 | 19.5 | - | - | - | 74 | 0 |
| int-q2-code-old | 6.00 | 14.26 | 20.8 | 170 / 342 | - | - | 78 | 0 |
| int-q2-code-online | 5.96 | 14.63 | 20.9 | 139 / 373 | - | - | 75 | 0 |
