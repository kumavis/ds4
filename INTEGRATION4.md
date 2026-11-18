# kumanix-int4: kumanix-int3 + D13 staging cache (on), D9 model scheduler, D11/D12 traces

Branch `kumanix-int4` in worktree `/data/ds4-int4`, started from `kumanix-int3` (bd19dab).

| commit | what |
|---|---|
| 1ecb469 | merge `d13-staging-cache` at 5689f1c, no conflicts |
| 4861c0f | merge `d12-cache-policy` at 73ff7a4, two conflicts resolved |
| 00ae963 | D13 staging cache on by default; the reclaim logs a D5 shelf swap |
| 6a48b38, 3cf3109, f03aeb9 | `scripts/int4/` (no code) |
| (this commit) | `INTEGRATION4.md` |

- **Merge message:** the first merge kept git's default message ("Merge branch 'd13-staging-cache'"). It is the merge at 5689f1c.
- **What the D13 merge brings:**
  - D11 (`d11-drafter-experts`, 0d818ed), which already contains the D9 merge (`d9-dspark-economics` a9d8cb9/3cc75a5 at c508165) and D10's notes. So D9's model scheduler and D11's drafter trace come in through it.
  - D13 itself.
- **D12:** taken only at 73ff7a4 (`DS4_D12_CACHE_TRACE`). D15 work on that branch is not included.
- **Binary:** every number below is from one binary, built at 6a48b38. Its code equals 00ae963; later commits are scripts only. Records show the script commit current at run time.

## Bottom line

- **Gates:** all pass.
  - Every plain digest equals the fork reference: MXFP4 code at 8/64/512, prose, hash, long; Q2 code, hash, long.
  - Exact verify is bit-exact against plain on code and json.
  - DSpark old-scheduler rounds/accepted equal the references on all five prompts and Q2.
  - With `DS4_ROCM_STREAM_STAGING_CACHE=0`, plain code and DSpark code reproduce int3: digest, rounds, slot counts, and no lending.
  - The 5000-slot admission is still refused.
  - Model-free tests are identical to int3.
- **Defaults:**
  - The staging cache is on by default.
  - D5 prefetch stays off.
  - The scheduler stays `old`.
  - D9/D11/D12 trace switches are off unless set.
- **Speed against int3 (two-run means):**

  | line | int3 → int4 | change |
  |---|---|---:|
  | MXFP4 plain code | 9.135 → 9.61 | +5.2% |
  | MXFP4 plain prose | 9.805 → 10.57 | +7.8% |
  | MXFP4 plain hash | 9.26 → 9.755 | +5.3% |
  | MXFP4 plain long | 8.81 → 9.455 | +7.3% |
  | DSpark code | 11.965 → 13.23 | +10.6% |
  | DSpark prose | | +4.8% |
  | DSpark json | | +5.0% |
  | DSpark translate | | +4.3% |
  | DSpark list | | +8.4% |
  | Q2 | flat | |

  - Standard suite: MXFP4 decode 9.665, long 9.415; Q2 decode 11.25, long 13.355.
- **Memory:** the lowest `avail_min_mb` over all 48 int4 runs that started decoding was **12,833 MiB**, 545 above the 12,288 floor (model-scheduler translate). On the default settings the lowest was 12,916 (DSpark json r1). The hottest NVMe was 83 °C.
- **Studies:**
  - Model scheduler (refitted) vs old: −1.5% over the five prompts; translate −13%. Keep `old`.
  - Prefetch on top of the staging cache: +1.0% plain code, +1.2% long, +0.2% DSpark code (one run each, within noise). The "stays off" decision holds.

## 1. Merges

**d13-staging-cache (5689f1c): clean.**
- int3 and D13 both descend from int2. D13's only runtime change is the lend/reclaim code and its guards. D5b had touched the same file elsewhere.
- I checked the reclaim against D5b's shelf by reading the code:
  - `d5_shelf_entry` is unchanged in D5b.
  - `cuda_d5_shelf_reset` still resets an entry the same way (wait for in-flight chunks, clear the tasks, keep the buffer), and the reclaim mirrors it.
  - D5b raised the default shelf from 4 to 16 slots, under the same `DS4_D5_MAX_SHELF` (32) bound the reclaim's array uses.
- No admission arithmetic was touched. int3's shelf charge in `ds41_mem_cache_plan_compute` stays as it was.

**d12-cache-policy at 73ff7a4: two conflicts, both resolved by keeping both sides.**
- Both are in `rocm/ds4_rocm_runtime.cuh`, at the head of the batch prepare and of the single-token selected load.
- D5b's `cuda_d5_release_unneeded(...)` call now runs first, then D12's `cuda_d12_call(...)` trace.
- The trace only records ids, so the order does not matter for output.

**D9 scheduler and D11/D12 traces:** already default-off on their branches and unchanged here:
- `DS4_DSPARK_SCHED=model`, with `old` still the default since D4;
- `DS4_D9_ROUND_TRACE`, `DS4_D9_PROBE`;
- `DS4_D11_DRAFT_TRACE`, `DS4_D12_CACHE_TRACE`.

## 2. Changes on the branch (00ae963)

**Default on.** `DS4_ROCM_STREAM_STAGING_CACHE` is now on unless set to `0`. With `=0`, `ds41_staging_cache_enabled()` is false and nothing is ever lent, so behaviour, slot counts and plan are int3's (gate in §4).

**Reclaim log for the D5 shelf.** If a reclaim finds D5 shelf buffers in staging memory, it prints:

`D13 staging reclaim: N D5 shelf buffer(s) were staging memory; swapped for non-staging free slots`

No test-only hook was needed; the natural sequence hits it (§4.5).

## 3. Model-free tests (`scripts/int4/tests.sh`, `/data/ds4-bench/int4-tests.log`)

The same set as int3. D13 added no unit test; its lend/reclaim path is gated by the sequence runs in §4.5.

| test | result |
|---|---|
| test_mxfp4_stream_rows_rocm, test_mxfp4_overlap_rocm, test_mxfp4_rocm | EXIT=0, PASS |
| test_d0_verify_rows_rocm | EXIT=1, "D0 kernel rows: FAIL", known |
| test_d5_prefetch_rocm, default limit and `TEMP_LIMIT_C=0` | EXIT=0, PASS |
| test_d5_prefetch_rocm, `TEMP_LIMIT_C=1` | EXIT=1, expected (every issue held, as int3) |
| c1_exact | "SOME DIFF", as int3 |

Every EQUAL/DIFF line is identical to `int3-tests.log`.

**Admission refusal (int4-gate-admit-5000).**
- `--ssd-streaming-cache-experts 5000` exits 1 at startup with no decode: "V4.1 ROCm needs 112.52 GiB additional usable memory including reserves; only 112.42 GiB is available".
- It is still refused, but only by 0.10 GiB on that run's MemAvailable.
- The admission charges the staging exactly as before. The memory-sized cache picks about 4,735 dynamic slots, so 5000 sits right at the edge.

## 4. Gates

Log: `/data/ds4-bench/int4-queue.log`. Every plain record has `ref_match` true.

### 4.1 Plain token digests (defaults: staging on)

| tag | digest | reference |
|---|---|---|
| int4-smoke-mx-8 / -64 | b1cd778da121/7, 3098d05a1cef/63 | same |
| int4-gate-mx-code (logits reference) | 8290541d324b/511 | same |
| int4-gate-mx-prose | 88ed0e11adfd/511 | same |
| int4-gate-mx-hash | 4c37f5fee667/410 | same |
| int4-gate-mx-long | c27a3b087fb5/255 | same |
| int4-gate-q2-code | f0a2cd53a247/511 | same |
| int4-gate-q2-hash | 63e4df5d2ff1/398 | same |
| int4-gate-q2-long | 46aca9ac6072/255 | same |

All headline, suite and prefetch-study plain runs also match their references (16 more runs).

### 4.2 `--dspark-verify exact`, bit-exact against plain decode of the same binary

| run | reference | first divergence | compared | bit-exact | max logit diff | rounds/acc |
|---|---|---|---:|---:|---:|---|
| int4-gate-mx-code-exact | int4-gate-mx-code | none | 511 | 1.0 | 0.0 | 121/391 |
| int4-gate-mx-json-exact | int4-gate-ref-mx-json | none | 511 | 1.0 | 0.0 | 239/273 |

### 4.3 DSpark fast verify, old scheduler, rounds/accepted

| prompt | reference | r1 | r2 |
|---|---|---|---|
| code | 172/340 | 172/340 | 172/340 |
| prose | 435/77 | 435/77 | 435/77 |
| json | 207/305 | 207/305 | 207/305 |
| translate | 176/336 | 176/336 | 176/336 |
| list | 403/109 | 403/109 | 403/109 |
| Q2 code | 170/342 | 170/342 | 170/342 |

### 4.4 Switch off (`DS4_ROCM_STREAM_STAGING_CACHE=0`) = int3

| run | result | dynamic slots / peak | lend lines | int3 |
|---|---|---|---|---|
| int4-gate-off-mx-code | 8290541d324b/511, 9.06 tok/s | 4737 / 4737 | 0 | 9.135 tok/s, 4720–4731 slots |
| int4-gate-off-mx-code-dsp | 172/340, 11.83 tok/s | 4264 / 4264 | 0 | 11.965, 4252–4262 |

The memory-sized slot count moves by about ±10 with MemAvailable at start, as in D4.

### 4.5 Reclaim with the D5 shelf (prefetch on), staging off vs on twin

The four-turn REPL sequence (`scripts/int4/seq.sh`): long, code, translate+json, prose. That is two layer sweeps per process. Prefetch is on in all four runs.

| run | tokens per turn | stdout md5 | step-token md5 (steps) | reclaims | shelf-swap lines |
|---|---:|---|---|---:|---:|
| int4-seq-pf-off | 96 | b552fe517570 | c81586d7fc70 (384) | 0 | 0 |
| int4-seq-pf-on | 96 | b552fe517570 | c81586d7fc70 (384) | 2 | **0** |
| int4-seq-pf320-off | 320 | 63bbd538d409 | e65fe49a52c2 (1280) | 0 | 0 |
| int4-seq-pf320-on | 320 | 63bbd538d409 | e65fe49a52c2 (1280) | 2 | **1** |

**96 tokens per turn never put a staging buffer into the shelf, because of eviction order.**
- After a sweep, the ~4,700 entries prefill seeded are older than anything decode loads.
- So the first ~4,700 evictions take slab buffers, and only then do staging-backed entries become LRU victims whose buffers an adoption swap can move into the shelf.
- About 4,000 misses happened before the second sweep.

**At 320 tokens per turn the natural run hit it.** The second reclaim logged:

```
D13 staging reclaimed for a layer sweep: evicted 768 entries (budget 4722, slots 4722, resident 4722)
D13 staging reclaim: 1 D5 shelf buffer(s) were staging memory; swapped for non-staging free slots
```

The output and every decode token were identical to the staging-off twin. D5 made 8,628 swaps in that run.

### 4.6 Memory floor and temperature

- **Memory:** every run that decoded is at or above 12,288 MiB. Lowest values:
  - all int4 runs: 12,833 (int4-mx-translate-dsp-model);
  - defaults: 12,916 (int4-mx-json-dsp-r1).
- **Temperature:** hottest NVMe 83 °C (int4-mx-translate-dsp-r2). The prefetch runs never reached the 84 °C limit (held=0).
- Per-run values are in §5 and the records.

## 5. Measurements (defaults, decode tok/s)

**Setup.**
- MXFP4 on two drives. `DS4_TOKEN_TIMING=1`, `DS4_D8_STEP_TRACE=1`.
- DSpark runs: scheduler old (set explicitly), plus phase, scheduler and `DS4_D9_ROUND_TRACE` traces. The round trace is new against int3; it is one line per round.
- Repeat 1 of prose, hash, long, Q2 code and the DSpark lines is the gate run. Plain code uses two separate runs, because its gate run wrote a logits log.
- The int2 and int3 columns are those builds' defaults (INTEGRATION2.md, INTEGRATION3.md "off").

| line | int2 | int3 | int4 (r1, r2 = mean) | vs int3 | avail_min_mb r1, r2 | NVMe max C r1, r2 | dynamic slots / peak resident |
|---|---:|---:|---|---:|---|---|---|
| MXFP4 plain code | 9.15 | 9.135 | 9.68, 9.54 = **9.61** | +5.2% | 13751, 13796 | 70, 57 | 4733/5501, 4735/5503 |
| MXFP4 plain prose | 9.81 | 9.805 | 10.58, 10.56 = **10.57** | +7.8% | 13625, 14078 | 74, 74 | 4746/5514, 4715/5483 |
| MXFP4 plain hash | 9.12 | 9.26 | 9.78, 9.73 = **9.755** | +5.3% | 13808, 13851 | 73, 75 | 4735/5503, 4731/5499 |
| MXFP4 plain long (256) | 8.74 | 8.81 | 9.48, 9.43 = **9.455** | +7.3% | 13378, 13648 | 80, 79 | 4736/5504, 4718/5486 |
| MXFP4 DSpark code | 11.90 | 11.965 | 13.16, 13.30 = **13.23** | +10.6% | 13258, 13429 | 72, 78 | 4276/5044, 4263/5031 |
| MXFP4 DSpark prose | 9.74 | 9.745 | 10.17, 10.26 = **10.215** | +4.8% | 13523, 13277 | 57, 75 | 4261/5029, 4271/5039 |
| MXFP4 DSpark json | 11.79 | 11.915 | 12.44, 12.58 = **12.51** | +5.0% | 12916, 12983 | 81, 82 | 4263/5031, 4267/5035 |
| MXFP4 DSpark translate | 12.66 | 12.66 | 13.21, 13.20 = **13.205** | +4.3% | 13166, 13167 | 70, 83 | 4255/5023 both |
| MXFP4 DSpark list | 10.57 | 10.55 | 11.45, 11.42 = **11.435** | +8.4% | 13199, 13387 | 70, 75 | 4268/5036, 4253/5021 |
| Q2 plain code | 12.17 | 12.11 | 12.12, 12.11 = **12.115** | 0.0% | 19979, 19958 | 70, 71 | 9636/9812, 9623/9812 (not full) |
| Q2 DSpark code | 14.55 | 14.58 | 14.59, 14.70 = **14.645** | +0.4% | 13381, 13646 | 71, 61 | 8754/9522, 8722/9490 |

Q2 hash and long ran once each, as gates: 11.23 and 13.38 tok/s.

### Standard suite (`suite.sh`, labels int4 and int4r2, mean of two)

| line | kumanix-int | int2 | int4 | vs int2 |
|---|---:|---:|---|---:|
| MXFP4 decode (hash) | 8.60 | 9.19 | 9.67, 9.66 = **9.665** | +5.2% |
| MXFP4 long | 8.17 | 8.76 | 9.42, 9.41 = **9.415** | +7.5% |
| Q2 decode | 11.05 | 11.28 | 11.23, 11.27 = **11.25** | −0.3% |
| Q2 long | 11.63 | 12.48 | 13.39, 13.32 = **13.355** | +7.0% |

- int3 did not run the suite.
- Q2 long gains because its 1.4k-token prompt sweep leaves the lent slots to decode on a full cache.
- Q2 decode does not fill its cache in 512 tokens, so it gains nothing.

## 6. Studies (one run each)

### 6.1 Model scheduler on the larger cache

**Refit** (`scripts/int4/refit.py` over the ten int4 DSpark old-scheduler runs: 1,050 verify rounds, 1,646 plain rounds).
- The old scheduler prints no DS41_SCHED lines, and the round trace's miss counter covers single-row loads only. So I fitted the scheduler's folded form, verify ms = VA + VB × rows, and did not redo D9's three-term fit.

| quantity | D9 (int, smaller cache) | int4 |
|---|---|---|
| verify+commit ms | VA 77 + VB 49 × rows (defaults, folded from 77.2 + 14.2 rows + 1.37 misses) | 81.0 + 37.9 × rows (R² 0.51, rmse 47 ms) |
| non-read ms | 68.9 + 9.2 × rows | 71.4 + 9.2 × rows |
| read wait ms per row | ~35 | 28.7 |
| plain step P | 118 ms | 97.6 ms |

- Rows cost less read time with 768 more slots. Compute per row is unchanged.

**Model scheduler with the refit** (`DS4_DSPARK_SCHED=model DS4_DSPARK_MODEL_P=98 VA=81 VB=38`) against old (int4 two-run means):

| prompt | old | model (refit) | change | rounds/acc old → model |
|---|---:|---:|---:|---|
| code | 13.23 | 13.93 | +5.3% | 172/340 → 124/388 |
| prose | 10.215 | 10.60 | +3.8% | 435/77 → 431/81 |
| json | 12.51 | 12.48 | −0.2% | 207/305 → 207/305 |
| translate | 13.205 | 11.47 | −13.1% | 176/336 → 233/279 |
| list | 11.435 | 11.23 | −1.8% | 403/109 → 392/120 |
| mean of five | 12.12 | 11.94 | −1.5% | |

- Output class is `dspark-fast` either way.
- **Not worth making the default.** It wins on code and prose but loses badly on translate. On translate it drafts more often and gets fewer drafts accepted, as D9's free-running runs also showed.
- I did not run the model scheduler with D9's default constants. The refit changes P and V in the same proportion (V/P at three rows: 1.90 with D9's constants, 1.99 with the refit), so a large difference is not expected (estimate).

### 6.2 D5 prefetch on top of the staging cache

| line | prefetch off (int4 mean) | prefetch on | change | check |
|---|---:|---:|---:|---|
| plain code | 9.61 | 9.71 | +1.0% | digest = ref; dynamic slots 4695 (shelf charged) |
| plain long | 9.455 | 9.57 | +1.2% | digest = ref |
| DSpark code | 13.23 | 13.25 | +0.2% | 172/340 |

- Issued 12.8k / 5.0k / 11.2k speculative reads. Thermal limit never held (68–79 °C).
- On int3 prefetch gave +2.2% plain code, +2.7% long and +2.7% DSpark code. With 768 more slots there are fewer misses left to predict.
- With one run each, these gains are inside run-to-run spread: plain code was 9.68 vs 9.54 across its two headline runs.
- **The "stays off" decision holds.**

## 7. Recommendation

- **Staging cache:** on (done in this branch). All gates pass, there is no memory cost, and it gives +5–8% plain and +4–11% DSpark on MXFP4.
- **Scheduler:** keep `old` as the default. The refitted model scheduler is −1.5% over the five prompts and −13% on translate.
- **Prefetch:** keep it off. It is worth about +1% or less on top of the staging cache, at 100–370 GiB of extra reads per run.

## 8. Deviations

- Repeat 1 of most headline lines is the gate run. int3 measured separately.
- DSpark runs carry `DS4_D9_ROUND_TRACE=1`, which was not in int3's measurements. It was needed for the refit and adds one stderr line per round.
- The shelf-swap gate needed 320 tokens per turn instead of 96; no test hook.
- Queue 2 (studies) was launched by hand after the refit, instead of in the same chain.
- The suite was added (queue 3) so the table lines up with int2. int3 had not run it.

## 9. Not verified

- DSpark studies have one run each. The model scheduler was not run with D9's default constants.
- The prefetch study did not cover DSpark json/translate/list, Q2, or a second run.
- The D11/D12 trace switches were not turned on in int4 runs. They are opt-in and only print.
- Server mode, multiple sessions, and contexts above 4,096. A 16k+ prompt (several reclaims per prompt) was not run on int4.
- The 5000-slot refusal margin depends on MemAvailable at start (0.10 GiB here). It was refused, but a host with ~0.1 GiB more free memory would admit it.
- `--dspark-verify exact` with prefetch on (int3 did it; not repeated here).

## Scripts (`scripts/int4/`)

| script | what |
|---|---|
| `common.sh` | from int3 |
| `build.sh`, `chain.sh` | build, tests, queue 1 |
| `tests.sh` | model-free tests |
| `q1.sh` | smoke, admission refusal, gates, headline |
| `seq.sh` | REPL sequence gate |
| `refit.py` | scheduler cost refit |
| `q2.sh` | shelf-swap sequence, model scheduler, prefetch |
| `q3.sh` | standard suite twice |

## Run tags

**Benchmark records.**
- Smoke and admission: int4-smoke-mx-{8,64}, int4-gate-admit-5000 (expected exit 1).
- Plain gates: int4-gate-mx-{code,prose,hash,long}, int4-gate-q2-{code,hash,long}.
- Exact verify: int4-gate-mx-code-exact, int4-gate-ref-mx-json, int4-gate-mx-json-exact.
- Switch off: int4-gate-off-mx-code, int4-gate-off-mx-code-dsp.
- Headline: int4-mx-{code,prose,json,translate,list}-dsp-r{1,2}, int4-q2-code-dsp-r{1,2}, int4-mx-code-plain-r{1,2}, int4-mx-{prose,hash,long}-plain-r2, int4-q2-code-plain-r2.
- Studies: int4-mx-{code,prose,json,translate,list}-dsp-model, int4-mx-{code,long}-plain-pf, int4-mx-code-dsp-pf.
- Suite: suite-int4-*, suite-int4r2-*.

**Sequences** (in `/data/ds4-bench/int4-seq/`): int4-seq-pf-{off,on}, int4-seq-pf320-{off,on}.
