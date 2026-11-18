# kumanix-int3: kumanix-int2 + D5b, shelf memory charge, thermal limit

Branch `kumanix-int3` lives in worktree `/data/ds4-int3`, started from
`kumanix-int2` (249ca97).

| commit | what |
|---|---|
| 170364f | merge `d5-prefetch` at its tip 06fac21 (D5b), no conflicts |
| 6b08ad5 | D4 charges the D5 shelf when prefetch is on; NVMe thermal limit for speculative reads |
| (this commit) | `INTEGRATION3.md`, `scripts/int3/` (no code) |

The binary measured below was built from 6b08ad5. No default was changed:
prefetch stays off unless `DS4_ROCM_D5_PREFETCH=1`.

**Bottom line.**
- The build passes, and the model-free tests match int2. The D5 test also
  passes with its 2-row batch loads.
- Every gate passes:
  - the five plain gates with prefetch off and on;
  - the forced thermal hold;
  - `--dspark-verify exact` with prefetch on, bit-exact against plain;
  - DSpark fast verify off against on: rounds, drafts and logits identical,
    and hit/miss counts identical at a fixed cache size.
- **Prefetch gain is smaller than on int2:**
  - plain: +2.1% to +2.7%;
  - DSpark: code +2.7%, prose +2.3%, list +1.1%, json +0.7%, translate -0.9%;
  - Q2: no effect.
- **Memory:** with the shelf charged, prefetch-on runs now hold the same
  MemAvailable margin as prefetch-off runs. The lowest minimum was 12923 MiB,
  635 MiB above the 12 GiB floor.
- **Thermal limit:** it engaged in 3 of 20 prefetch-on runs. The hottest NVMe
  reading anywhere was 84 C, and no run reached 86 C.

## 1. Merge (d5-prefetch 06fac21)

The merge applied cleanly. D5b adds the following in ds4.c:
- the verify-round probe and issue in `ds41_moe_batch_d2`;
- an optional layer-0 probe in `ds41_before_attention_batch`;
- runtime changes: stale-entry drop, busy-time cap, more knobs, and a shelf
  default of 16 slots.

Interactions checked by reading:
- **C1 profiler:** it has no marks in the batch (verify) path, so it does not
  meet D5b's verify-round code.
- **Verify-round probe:** it uses private scratch (8 rows x 5120 floats plus
  8 x 384 logits, about 176 KiB) and the same probe readback as the
  single-token path.
- **D4:** see section 2. The shelf default went from 4 to 16 slots
  (16 x 17.93 MiB = 287 MiB). The code comment still said 4 and is corrected.

## 2. Shelf charged in D4's memory plan (6b08ad5)

**Runtime side.** `ds4_gpu_d5_shelf_plan_bytes(slot)` returns the bytes the
shelf will allocate: `DS4_ROCM_D5_SHELF` (default 16) x one expert. It
returns 0 when no shelf is ever allocated:
- prefetch off;
- `DS4_ROCM_D5_DIAG=probe`;
- `DS4_ROCM_STREAM_NO_DIRECT`.

**Plan side.** `ds41_mem_cache_plan_compute` adds those bytes to its fixed
term. The slot size is taken from the first MXFP4 expert layer: prefetch only
engages on MXFP4. So the cache gets about 16 fewer dynamic slots when
prefetch is on, and the plan prints
`fixed includes the D5 prefetch shelf 0.28 GiB (286.9 MiB)`.

**Prefetch off:** the charge is 0 and the code path is as on int2, so plan
and slot counts are unchanged. The int2 shelf sat outside the plan, so int2
runs with prefetch on had no shelf in their plan.

**Q2:** no charge (no MXFP4 experts); the smoke and gate runs confirm it.

**Limits of the charge:**
- Legacy sizing and an explicit `--ssd-streaming-cache-experts` are not
  charged, as D4 also leaves them alone.
- If prefetch is on but D2 decode is off (`DS4_ROCM_D2_DECODE=0`), the shelf
  is charged but never allocated. That over-charges 287 MiB, which is safe.

Measured effect is in the Memory section.

## 3. Thermal limit for speculative reads (6b08ad5)

**Switch.** `DS4_ROCM_D5_TEMP_LIMIT_C`, default 84, 0 = no limit; hysteresis
3 C.

**Monitor.** On the first prefetch call, the runtime globs
`/sys/class/nvme/nvme*/hwmon*/temp1_input` (2 sensors here). It takes one
synchronous reading so the limit applies from the start, then starts a
detached thread. Once a second the thread reads the hottest sensor and
updates an atomic hold flag:
- set at >= limit;
- cleared at <= limit - 3.

**Hot path.** `ds4_gpu_d5_prefetch` only loads that flag. While it is set,
the call returns without queuing anything. The shelf stays allocated, and
demand reads are never limited. Holding back optional reads cannot change
tokens, and the gates confirm it.

**Stats.** One extra line after the D5 stats:
`D5 thermal: limit= monitor= held=<calls held> held_cands=<candidates in them> engaged=<transitions into hold> nvme_max=<hottest reading>`.

## 4. Model-free tests (`/data/ds4-bench/int3-tests.log`)

| test | result |
|---|---|
| test_mxfp4_stream_rows_rocm, test_mxfp4_overlap_rocm, test_mxfp4_rocm | EXIT=0, PASS |
| test_d0_verify_rows_rocm | EXIT=1, "D0 kernel rows: FAIL" (known; all EQUAL/DIFF lines identical to int2 and kumanix-int) |
| c1_exact | exit 0, "SOME DIFF"; lines identical to int2 (shipped kernels EQUAL) |
| test_d5_prefetch_rocm, default limit 84 | EXIT=0, PASS: 586 issued, 81+259 adopted, 0 held (sensors 56-57 C) |
| test_d5_prefetch_rocm, `TEMP_LIMIT_C=0` | EXIT=0, PASS: monitor off |
| test_d5_prefetch_rocm, `TEMP_LIMIT_C=1` | EXIT=1: 0 issued, 503 calls held (1132 candidates) |

About the D5 test:
- It is run as its make target's recipe, and includes the 2-row batch union
  loads (every fourth load).
- **The `TEMP_LIMIT_C=1` failure is expected.** Every per-step check
  (shelf bytes, union and LRU order) passed. Only the final sanity check
  failed, which requires the test to have exercised adoption at all
  ("expected adoptions and drops and no failures"). With every issue held,
  there is nothing to adopt. The run confirms the hold path is a no-op for
  the cache.

## 5. Gates

All plain runs use `DS4_TOKEN_TIMING=1`. A digest is the md5 of `"N ID"` over
the `perf token` lines, followed by the line count. The log is
`/data/ds4-bench/int3-queue.log`.

| tag | switches | digest | reference | |
|---|---|---|---|---|
| int3-smoke-mx-8 / -64 | defaults | b1cd778da121/7, 3098d05a1cef/63 | same | pass |
| int3-smoke-mx-pf-64 | prefetch | 3098d05a1cef/63 | same | pass |
| int3-gate-mx-code (logits ref) | defaults | 8290541d324b/511 | 8290541d324b/511 | pass |
| int3-gate-mx-prose | defaults | 88ed0e11adfd/511 | 88ed0e11adfd/511 | pass |
| int3-gate-mx-hash | defaults | 4c37f5fee667/410 | 4c37f5fee667/410 | pass |
| int3-gate-mx-long | defaults | c27a3b087fb5/255 | c27a3b087fb5/255 | pass |
| int3-gate-q2-code | defaults | f0a2cd53a247/511 | f0a2cd53a247/511 | pass |
| int3-gate-pf-mx-code | prefetch | 8290541d324b/511; logits vs int3-gate-mx-code bit-exact 511/511 | | pass |
| int3-gate-pf-mx-prose | prefetch | 88ed0e11adfd/511 | | pass |
| int3-gate-pf-mx-hash | prefetch | 4c37f5fee667/410 | | pass |
| int3-gate-pf-mx-long | prefetch | c27a3b087fb5/255 | | pass |
| int3-gate-pf-q2-code | prefetch | f0a2cd53a247/511 | | pass |
| int3-gate-pf-mx-code-hold | prefetch, `TEMP_LIMIT_C=1` | 8290541d324b/511; issued=0, held=23595 calls (141570 candidates), engaged=1 | | pass |

**DSpark `--dspark-verify exact`, prefetch on, MXFP4 code**, against
int3-gate-mx-code. These runs exercise the single-token prefetch under
DSpark (issued 16127 and 16114):

| tag | div_first_div | compared | tf_agree | logit maxabs | bitexact | rounds / accepted |
|---|---|---:|---:|---:|---:|---|
| int3-gate-pf-mx-code-exact (free) | none | 511 | 1.0 | 0.0 | 1.0 | 121 / 391 |
| int3-gate-pf-mx-code-exact-tf (teacher-forced) | none | 511 | 1.0 | 0.0 | 1.0 | 121 / 391 |

**DSpark fast verify, scheduler old, prefetch off (logits reference) against
on.**
- First pass, `gate.sh`, memory-sized cache:
  - Rounds, accepted drafts and logits were identical (bit-exact, 512/512)
    on code, json and translate.
  - Hit/miss counts differed by a few, for example code decode misses
    7606 / 7608 and json prefill misses 10327 / 10388.
  - The cause was the cache size, not prefetch. The memory-sized cache
    differed between the paired runs: dynamic slots 4252/4249, 4262/4245 and
    4255/4240. MemAvailable at open varies by about ±0.3 GiB, and the on run
    is charged the 287 MiB shelf.
  - D5b's own check ran at legacy sizing, where the size is fixed.
  - The gate script counted this as a failure and stopped the queue.
- Rerun, `gate2.sh`, at a fixed `--ssd-streaming-cache-experts 4240` for both
  runs. (A first attempt with 5000 was refused at admission in 2 s: N counts
  dynamic slots. No GPU work ran.)

| prompt | rounds / accepted | decode hits / misses | batch (prefill + verify) hits / misses | logits on vs off | |
|---|---|---|---|---|---|
| code | 172/340 both | 30550/7610 both | 65806/13156 both | bit-exact 512/512 | pass |
| json | 207/305 both | 14603/1237 both | 71558/10406 both | bit-exact 512/512 | pass |
| translate | 176/336 both | 4266/294 both | 77263/10871 both | bit-exact 512/512 | pass |

## 6. Measurements

Every number below is from `results.jsonl` (`scripts/int3/measure.sh`).
- MXFP4 runs use two drives.
- All runs: `DS4_TOKEN_TIMING=1`, `DS4_D8_STEP_TRACE=1`, no logits logs.
- DSpark runs: scheduler old (the default) plus phase and scheduler traces.
- Prefetch on means `DS4_ROCM_D5_PREFETCH=1`, with the thermal limit at its
  default of 84 C.
- Off and on runs were interleaved.
- D9's queue shared the lock throughout.

Decode tok/s, mean of two runs (r1, r2). NVMe max C is per run, r1 / r2.

| line | off | on | change | avail min MiB off | avail min MiB on | NVMe max C off | NVMe max C on | thermal hold (on) |
|---|---|---|---:|---|---|---|---|---|
| plain code | 9.135 (9.14, 9.13) | 9.335 (9.35, 9.32) | +2.2% | 13845, 13874 | 14025, 13753 | 73, 76 | 79, 79 | none |
| plain prose | 9.805 (9.80, 9.81) | 10.01 (10.04, 9.98) | +2.1% | 13801, 13757 | 13786, 13821 | 75, 76 | 79, 64 | none |
| plain hash | 9.26 (9.27, 9.25) | 9.48 (9.44, 9.52) | +2.4% | 13722, 13658 | 13926, 13810 | 75, 70 | 79, 78 | none |
| plain long (256) | 8.81 (8.79, 8.83) | 9.045 (9.10, 8.99) | +2.7% | 13514, 13267 | 13457, 13608 | 80, 79 | 80, 80 | none |
| DSpark code | 11.965 (11.97, 11.96) | 12.285 (12.28, 12.29) | +2.7% | 13300, 13407 | 13347, 13441 | 80, 80 | 68, **84** | r2: 77 calls held, engaged once |
| DSpark prose | 9.745 (9.75, 9.74) | 9.97 (9.98, 9.96) | +2.3% | 13422, 13473 | 13335, 13421 | 70, 77 | 75, 79 | none |
| DSpark json | 11.915 (11.94, 11.89) | 11.995 (12.02, 11.97) | +0.7% | 12965, 12999 | 12923, 12976 | 82, 83 | 83, **84** | r2: 1209 calls held (18523 candidates), engaged once |
| DSpark translate | 12.66 (12.68, 12.64) | 12.545 (12.55, 12.54) | -0.9% | 13071, 12963 | 13061, 12989 | 83, 83 | 82, 83 | r2: 861 calls held (10844 candidates), engaged once |
| DSpark list | 10.55 (10.58, 10.52) | 10.67 (10.70, 10.64) | +1.1% | 13449, 13456 | 13528, 13404 | 77, 78 | 79, 79 | none |
| Q2 code plain (single) | 12.11 | 12.10 | 0 | 13935 | 13735 | 73 | 72 | not engaged (no MXFP4) |
| Q2 code DSpark (single) | 14.58 | 14.57 | 0 | 13516 | 13470 | 73 | 73 | not engaged |

**DSpark round counts.** Rounds and accepted drafts are identical off and on
for every prompt: code 172/340, prose 435/77, json 207/305, translate
176/336, list 403/109, Q2 170/342.

**Speculative traffic per run (on):**

| line | issued | adopted (ready + in flight) | speculative GiB |
|---|---|---|---|
| plain code | 14.0k | 6.3k | 246 |
| plain prose | 12.3k | 5.7k | 215 |
| plain hash | 10.9k | 5.1k | 191 |
| plain long | 5.7k | 2.3k | 99 |
| DSpark code | 12.0k | 6.8k | 210 |
| DSpark prose | 13.2k | 6.4k | 231 |
| DSpark json | 9.8k / 7.9k | 4.3k / 3.4k | 171 / 138 |
| DSpark translate | 9.6k / 8.4k | 4.1k / 3.5k | 168 / 147 |
| DSpark list | 12.8k | 5.9k | 225 |

### Compared with int2 (prefetch with a 4-slot shelf, uncharged)

| line | int2 on/off | int3 on/off |
|---|---|---|
| plain code | +2.4% | +2.2% |
| plain prose | +2.3% | +2.1% |
| plain hash | +4.0% | +2.4% |
| plain long | +3.4% | +2.7% |
| DSpark code | +1.0% (single run) | +2.7% (the verify-round prefetch is new in D5b) |
| DSpark json | +1.7% (single run) | +0.7% |

Plain hash and long lost the most. The off speeds match int2: code 9.135 vs
9.15, prose 9.805 vs 9.81.

Possible causes of the smaller plain gain, as an estimate, not isolated:
- about 16 fewer cache slots on prefetch-on runs (+0.3 misses/token);
- the D5b runtime changes;
- noise.

### Memory

**Floor.** Lowest `avail_min_mb` over all int3 runs: **12923 MiB**
(int3-mx-json-dsp-on-r1). That is **635 MiB above the 12288 floor**. Lowest
with prefetch off: 12963 (translate DSpark r2). D4.md lists 12,964-13,136 MiB
for these prompts.

**Margin restored.** On int2, prefetch lowered `avail_min_mb` by 206 MiB on
average over the plain pairs, and int2's lowest run (12787) was a prefetch
run. On int3, the on-minus-off difference is +93 MiB on average over the
8 plain pairs and -8 MiB over the 10 DSpark pairs. Both are noise
(run-to-run spread is about ±200 MiB). The on runs carry about 16 fewer
dynamic slots instead (for example code 4703/4720 against 4731/4731).

### Temperature

- **Hottest NVMe:** 84 C, reached by:
  - three prefetch-on DSpark runs (code r2, json r2, translate r2), where the
    limit engaged and held issues back;
  - two prefetch-off gate runs (json and translate DSpark at 84);
  - two prefetch-on gate runs by the bench sampler (see below).
- **86 C or above:** no run reached it with the default limit, and none
  reached it at all.
- **How often the limit engaged:** in 3 of 20 measurement on-runs and in no
  gate run, except the forced hold. It engaged once per run, with no repeated
  flapping, and held 77, 1209 and 861 issue calls (1249, 18523 and 10844
  candidates).
- **Monitor and bench readings can differ.** In two gate on-runs
  (int3-gate-dsp-json-on and -translate-on) the bench sampler recorded 84 C,
  but the monitor's own maximum was 81 and 83 C, and the limit did not
  engage. The two read at different moments: the bench sampler every 2 s for
  the whole run, the monitor once a second, only from the first prefetch call
  until exit. Peaks outside the decode window, for example in the cooldown or
  at the start of prefill, are not seen by the monitor. Not traced further.
- **Effect when held:** fewer speculative reads (json r2 138 GiB against
  171; translate r2 147 against 168), with no measurable speed loss (json
  r2 11.97 against r1 12.02).
- **Without prefetch** the json and translate DSpark runs also reach
  82-84 C, so the limit cannot keep the drives below 84 by itself.

## 7. Recommendation: prefetch on by default (decision is the PM's)

I would turn it on.

**For:**
- Exactness is now gated on every path:
  - plain decode, bit-exact logits;
  - DSpark exact verify with prefetch on;
  - DSpark fast verify, off against on, at a fixed cache: rounds, drafts,
    hit/miss counts and logits identical;
  - the forced thermal hold.
- It no longer costs MemAvailable margin.
- The thermal limit keeps speculative reads from pushing the drives past
  84 C. The worst prefetch run reached 84, the same as the worst no-prefetch
  run.
- It gains +2.1% to +2.7% on plain decode and +2.3% to +2.7% on DSpark code
  and prose. It is about neutral on json and list (+0.7%, +1.1%) and does
  nothing on Q2.

**Against:**
- DSpark translate is -0.9% (both reps lower).
- About 100-250 GiB of extra SSD reads per 512-token run, roughly half of
  them dropped unused. Reads add little NAND wear (estimate), but they add
  drive heat and power.
- The gain is now 2-3%, not the 3-4% int2 showed.

If you want the cautious version, turn it on for plain decode only and leave
the verify-round part off (`DS4_ROCM_D5_VERIFY_K=0`). That loses the DSpark
code/prose gain but avoids the translate loss. I did not measure this
combination.

## Deviations

- **The DSpark off/on count gate failed at the memory-sized cache.** The
  cache-size difference explains it. I reran it at a fixed cache size
  (gate2), where everything is identical. This needed one extra script and
  two restarts of the chain; one attempt was refused at admission before any
  GPU work.
- **Q2:** off and on for both plain and DSpark (4 runs instead of 2).
- **D5 test:** built with the incremental strix-halo build and run with the
  make target's recipe, as in int2. It was also run with `TEMP_LIMIT_C=0`
  and `=1`.
- **Shelf charge:** applies to the memory-sized automatic cache only, not to
  legacy sizing or an explicit cache size.

## Not run or not verified

- The standard suite.
- The online scheduler.
- `DS4_ROCM_D5_VERIFY_K=0`, or other D5 knobs.
- Long sessions: thermal behaviour beyond 512 tokens.
- A real thermal hold during a plain run: the default limit never engaged on
  plain runs.
- The limit's hysteresis with a sensor that oscillates around the limit (only
  one engagement per run was seen).
- Read-only NVMe wear: not measured.
- C1 profiler and D5 probe/trace modes on this binary.
- Why D5b's stale shelf entries occur (inherited from D5.md).
