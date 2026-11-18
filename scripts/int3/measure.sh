#!/usr/bin/env bash
# kumanix-int3 measurements, after the gate passed. MXFP4, two drives
# (replica), DS4_TOKEN_TIMING=1 and DS4_D8_STEP_TRACE=1 on every run; DSpark
# runs add DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 and use the
# default scheduler (old). No logits logs. Prefetch off vs
# DS4_ROCM_D5_PREFETCH=1 (thermal limit at its default, 84 C), interleaved,
# two reps:
#  a. plain code, prose, hash, long
#  b. DSpark code, prose, json, translate, list
#  c. Q2 code: plain and DSpark, off and on (one run each)
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) MEASURE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1"
m() { # TAG PROMPT [env...] [-- ds4 flags]
  local tag=$1 p=$2; shift 2; local envs=() fl=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift; fl=("$@")
  read -r -a A <<< "$(args $p)"
  ( [ ${#envs[@]} -gt 0 ] && export "${envs[@]}"
    DS4_NOTE="int3 $tag${envs[*]:+ (${envs[*]})}" run $tag $(ntok $p) "${A[@]}" "${fl[@]}" ) || exit 1; }
mx
for rep in 1 2; do
  for p in code prose hash long; do
    m int3-mx-$p-plain-off-r$rep $p
    m int3-mx-$p-plain-on-r$rep $p DS4_ROCM_D5_PREFETCH=1
  done
  for p in code prose json translate list; do
    m int3-mx-$p-dsp-off-r$rep $p $DSPT -- --dspark --mtp-model $DSP
    m int3-mx-$p-dsp-on-r$rep $p DS4_ROCM_D5_PREFETCH=1 $DSPT -- --dspark --mtp-model $DSP
  done
done
q2
m int3-q2-code-plain-off code
m int3-q2-code-plain-on code DS4_ROCM_D5_PREFETCH=1
m int3-q2-code-dsp-off code $DSPT -- --dspark --mtp-model $DSP
m int3-q2-code-dsp-on code DS4_ROCM_D5_PREFETCH=1 $DSPT -- --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
