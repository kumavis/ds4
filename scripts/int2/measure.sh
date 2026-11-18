#!/usr/bin/env bash
# kumanix-int2 measurements, after the gate passed. Build defaults unless
# stated (memory-sized cache, C1 kernels on, prefetch off, DSpark scheduler
# old); MXFP4 with two drives (replica set); DS4_TOKEN_TIMING=1 and
# DS4_D8_STEP_TRACE=1 on every run; DSpark runs add DS4_ROCM_DSPARK_PHASE_TRACE=1
# and DS4_DSPARK_SCHED_TRACE=1 (as kumanix-int). No logits logs.
#  a. suite.sh, twice (labels int2, int2r2)
#  b. MXFP4 plain code, prose, hash, long: defaults and DS4_ROCM_D5_PREFETCH=1,
#     two reps, interleaved
#  c. MXFP4 DSpark code, prose, json, translate, list: defaults, two reps;
#     code and json with prefetch on, one run each
#  d. Q2 code: plain and DSpark, one run each
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) MEASURE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1"
m() { # TAG PROMPT [env...] [-- ds4 flags]
  local tag=$1 p=$2; shift 2; local envs=() fl=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift; fl=("$@")
  read -r -a A <<< "$(args $p)"
  ( [ ${#envs[@]} -gt 0 ] && export "${envs[@]}"
    DS4_NOTE="int2 $tag${envs[*]:+ (${envs[*]})}" run $tag $(ntok $p) "${A[@]}" "${fl[@]}" ) || exit 1; }
# a
mx
for lab in int2 int2r2; do
  echo "$(date -Is) start suite-$lab (4 runs)" >> $LOG
  /data/ds4-env/suite.sh $lab > /dev/null 2>&1; src=$?
  for t in suite-$lab-mxfp4-decode suite-$lab-mxfp4-long suite-$lab-q2-decode suite-$lab-q2-long; do
    d=$(rundir $t)
    echo "$(date -Is) end $t suite_rc=$src $(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1) decode_tps=$(field $t decode_tps) avail_min_mb=$(field $t avail_min_mb) nvme_max_c=$(field $t nvme_temp_max_c) digest=$(digest $t | tr ' ' /) dir=$d" >> $LOG
    av=$(field $t avail_min_mb); [ -n "$av" ] && [ "${av%.*}" -lt "$FLOOR_MB" ] && stop "after $t (MemAvailable)"
  done
  [ $src = 0 ] || stop "after suite-$lab (rc=$src)"
done
# b
mx
for rep in 1 2; do
  for p in code prose hash long; do
    m int2-mx-$p-plain-r$rep $p
    m int2-mx-$p-pf-r$rep $p DS4_ROCM_D5_PREFETCH=1
  done
done
# c
for rep in 1 2; do
  for p in code prose json translate list; do
    m int2-mx-$p-dsp-r$rep $p $DSPT -- --dspark --mtp-model $DSP
  done
done
m int2-mx-code-dsp-pf code DS4_ROCM_D5_PREFETCH=1 $DSPT -- --dspark --mtp-model $DSP
m int2-mx-json-dsp-pf json DS4_ROCM_D5_PREFETCH=1 $DSPT -- --dspark --mtp-model $DSP
# d
q2
m int2-q2-code-plain code
m int2-q2-code-dsp code $DSPT -- --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
