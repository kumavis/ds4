#!/usr/bin/env bash
# D16 sweep. MXFP4, two drives, fixed cache (--ssd-streaming-cache-experts:
# plain $MXP, DSpark $MXD dynamic slots). Settings: lambda 0 (x2), 0.09 (x2),
# 0.05, 0.1, 0.2 with keep 3; 0.1 with keep 2. Plain code (--nothink) and
# prose; DSpark (scheduler old) code, prose, json. Every speed run writes a
# logits log: lambda 0 r1 is the reference (DS4_D0_LOGITS=auto); every other
# run carries DS4_REF_TAG=<lambda 0 r1> (free-run divergence). One extra
# teacher-forced run (DS4_REF_FORCE=1) per lambda > 0 setting. lambda > 0
# runs are labelled DS4_OUTPUT_CLASS=approx-routing. Then the Q2 code pair
# (lambda 0 vs 0.09), plain and DSpark, fixed cache $Q2P / $Q2D.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) SWEEP QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
one() { # TAG PROMPT MODE(plain|dsp) LAMBDA KEEP REF FORCE
  local tag=$1 p=$2 mode=$3 lam=$4 keep=$5 ref=$6 force=$7
  read -r -a A <<< "$(args $p)"
  local fl=(--ssd-streaming-cache-experts $MXP) envs=()
  [ $mode = dsp ] && { fl=(--dspark --mtp-model $DSP --ssd-streaming-cache-experts $MXD); envs+=($DSPT); }
  if [ "$lam" != 0 ]; then envs+=(DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=$lam DS4_ROCM_ROUTE_CACHE_KEEP=$keep); fi
  if [ "$ref" = self ]; then envs+=(DS4_D0_LOGITS=auto); else envs+=(DS4_REF_TAG=$ref); fi
  [ "$force" = 1 ] && envs+=(DS4_REF_FORCE=1)
  ( export "${envs[@]}"; DS4_NOTE="d16 $tag (lambda $lam keep $keep $mode)" run $tag $(ntok $p) "${A[@]}" "${fl[@]}" ) || exit 1; }
mx
for mode in plain dsp; do
  prompts="code prose"; [ $mode = dsp ] && prompts="code prose json"
  for p in $prompts; do
    b=d16-mx-$p-$mode
    one $b-l0-r1 $p $mode 0 3 self 0
    one $b-l09-r1 $p $mode 0.09 3 $b-l0-r1 0
    one $b-l05 $p $mode 0.05 3 $b-l0-r1 0
    one $b-l10 $p $mode 0.1 3 $b-l0-r1 0
    one $b-l20 $p $mode 0.2 3 $b-l0-r1 0
    one $b-l10k2 $p $mode 0.1 2 $b-l0-r1 0
    one $b-l0-r2 $p $mode 0 3 $b-l0-r1 0
    one $b-l09-r2 $p $mode 0.09 3 $b-l0-r1 0
    for s in "l09 0.09 3" "l05 0.05 3" "l10 0.1 3" "l20 0.2 3" "l10k2 0.1 2"; do
      set -- $s; one $b-$1-tf $p $mode $2 $3 $b-l0-r1 1
    done
  done
done
q2
for mode in plain dsp; do
  b=d16-q2-code-$mode; read -r -a A <<< "$(args code)"
  if [ $mode = plain ]; then fl=(--ssd-streaming-cache-experts $Q2P); e2=(); else fl=(--dspark --mtp-model $DSP --ssd-streaming-cache-experts $Q2D); e2=($DSPT); fi
  ( export "${e2[@]}" DS4_D0_LOGITS=auto; DS4_NOTE="d16 $b-l0 (lambda 0 $mode)" run $b-l0 512 "${A[@]}" "${fl[@]}" ) || exit 1
  ( export "${e2[@]}" DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.09 DS4_ROCM_ROUTE_CACHE_KEEP=3 DS4_REF_TAG=$b-l0
    DS4_NOTE="d16 $b-l09 (lambda 0.09 keep 3 $mode)" run $b-l09 512 "${A[@]}" "${fl[@]}" ) || exit 1
done
echo "$(date -Is) QUEUE DONE" >> $LOG
