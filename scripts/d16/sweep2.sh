#!/usr/bin/env bash
# D16 reduced sweep (queue pace ~40-55 min per run): points that define the
# curve. MXFP4, two drives, fixed cache (plain $MXP / DSpark $MXD dynamic
# slots). Every run writes a logits log; lambda 0 runs are the references
# (DS4_D0_LOGITS=auto), lambda > 0 runs carry DS4_REF_TAG (free-run
# divergence) and DS4_OUTPUT_CLASS=approx-routing. keep 3 throughout.
#  a. code (--nothink, 512): lambda 0 (also the 8290541d324b/511 gate),
#     0.09, 0.05, 0.2
#  b. long (1.4k-token prompt, 256): lambda 0, 0.09
#  c. DSpark code (scheduler old): lambda 0 (also the 172/340 check)
# DSpark at two lambda > 0 follows in sweep3.sh once (a) is read.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) SWEEP2 QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
one() { # TAG PROMPT MODE LAMBDA REF
  local tag=$1 p=$2 mode=$3 lam=$4 ref=$5
  read -r -a A <<< "$(args $p)"
  local fl=(--ssd-streaming-cache-experts $MXP) envs=()
  [ $mode = dsp ] && { fl=(--dspark --mtp-model $DSP --ssd-streaming-cache-experts $MXD); envs+=($DSPT); }
  [ "$lam" != 0 ] && envs+=(DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=$lam DS4_ROCM_ROUTE_CACHE_KEEP=3)
  if [ "$ref" = self ]; then envs+=(DS4_D0_LOGITS=auto); else envs+=(DS4_REF_TAG=$ref); fi
  ( export "${envs[@]}"; DS4_NOTE="d16 $tag (lambda $lam keep 3 $mode)" run $tag $(ntok $p) "${A[@]}" "${fl[@]}" ) || exit 1; }
mx
one d16-mx-code-plain-l0 code plain 0 self
check d16-mx-code-plain-l0 8290541d324b 511 || stop "lambda 0 code digest"
one d16-mx-code-plain-l09 code plain 0.09 d16-mx-code-plain-l0
one d16-mx-long-plain-l0 long plain 0 self
check d16-mx-long-plain-l0 c27a3b087fb5 255 || stop "lambda 0 long digest"
one d16-mx-long-plain-l09 long plain 0.09 d16-mx-long-plain-l0
one d16-mx-code-plain-l05 code plain 0.05 d16-mx-code-plain-l0
one d16-mx-code-plain-l20 code plain 0.2 d16-mx-code-plain-l0
one d16-mx-code-dsp-l0 code dsp 0 self
echo "$(date -Is) QUEUE DONE (sweep2)" >> $LOG
