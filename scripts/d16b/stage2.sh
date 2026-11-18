#!/usr/bin/env bash
# D16b stage 2 on kumanix-int4 (d16b-cache-routing): build, smoke, lambda 0
# gates (int4 references: code 8290541d324b/511, long c27a3b087fb5/255,
# DSpark code 172/340), the lambda curve with the memory-sized cache (the
# staging cache lent, int4 default), then four agent sessions (D14 S1 to
# ~32k; plain and DSpark at lambda 0 and 0.09). Every lambda > 0 run carries
# DS4_OUTPUT_CLASS=approx-routing.
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
LOG=/data/ds4-bench/d16b-queue.log
echo "$(date -Is) STAGE2 BUILD start $(git rev-parse --short HEAD)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
mx; P=$E/p-code.txt
DS4_NOTE="d16b int4 smoke" run d16b-i4-smoke-mx-8 8 $P --nothink
check d16b-i4-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="d16b int4 smoke" run d16b-i4-smoke-mx-64 64 $P --nothink
check d16b-i4-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
one() { # TAG PROMPT MODE LAMBDA REF
  local tag=$1 p=$2 mode=$3 lam=$4 ref=$5 envs=() fl=()
  read -r -a A <<< "$(args $p)"
  [ $mode = dsp ] && { fl=(--dspark --mtp-model $DSP); envs+=($DSPT); }
  [ "$lam" != 0 ] && envs+=(DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=$lam DS4_ROCM_ROUTE_CACHE_KEEP=3)
  if [ "$ref" = self ]; then envs+=(DS4_D0_LOGITS=auto); else envs+=(DS4_REF_TAG=$ref); fi
  ( export "${envs[@]}"; DS4_NOTE="d16b $tag (lambda $lam keep 3 $mode, memory-sized cache)" run $tag $(ntok $p) "${A[@]}" "${fl[@]}" ) || exit 1; }
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o "[0-9.eE+-]*$"; }
# gates at lambda 0 (also the references)
one d16b-i4-code-plain-l0 code plain 0 self
check d16b-i4-code-plain-l0 8290541d324b 511 || stop "lambda 0 code digest"
one d16b-i4-long-plain-l0 long plain 0 self
check d16b-i4-long-plain-l0 c27a3b087fb5 255 || stop "lambda 0 long digest"
one d16b-i4-code-dsp-l0 code dsp 0 self
r="$(num d16b-i4-code-dsp-l0 dspark_cycles)/$(num d16b-i4-code-dsp-l0 dspark_accepted)"
echo "$(date -Is) ROUNDS d16b-i4-code-dsp-l0 $r ref 172/340 $([ "$r" = 172/340 ] && echo PASS || echo FAIL)" >> $LOG
[ "$r" = 172/340 ] || stop "lambda 0 DSpark rounds"
# lambda curve, memory-sized cache
one d16b-i4-code-plain-l09 code plain 0.09 d16b-i4-code-plain-l0
one d16b-i4-long-plain-l09 long plain 0.09 d16b-i4-long-plain-l0
one d16b-i4-code-dsp-l09 code dsp 0.09 d16b-i4-code-dsp-l0
one d16b-i4-code-dsp-l05 code dsp 0.05 d16b-i4-code-dsp-l0
one d16b-i4-code-plain-l09-r2 code plain 0.09 d16b-i4-code-plain-l0
one d16b-i4-code-dsp-l09-r2 code dsp 0.09 d16b-i4-code-dsp-l0
# agent sessions (D14 harness copy, S1 to ~32k)
H=/data/ds4-d16/scripts/d16b/harness/run.sh
cool() { local t0=$(date +%s) m
  while :; do m=0; for f in /sys/class/nvme/nvme*/hwmon*/temp1_input; do t=$(( $(cat $f) / 1000 )); [ $t -gt $m ] && m=$t; done
    [ $m -le 80 ] && break; [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 20; done
  echo "$(date -Is) cool: hottest NVMe ${m} C after $(( $(date +%s) - t0 )) s" >> $LOG; }
sess() { # TAG LAMBDA MODE
  local tag=$1 lam=$2 mode=$3 extra="" envs=()
  [ $mode = dsp ] && { extra="--mtp-model $DSP --dspark"; envs+=(DS4_DSPARK_SCHED=old); }
  [ "$lam" != 0 ] && envs+=(DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=$lam DS4_ROCM_ROUTE_CACHE_KEEP=3)
  cool; echo "$(date -Is) start $tag" >> $LOG
  ( [ ${#envs[@]} -gt 0 ] && export "${envs[@]}"
    DS4_DIR=/data/ds4-d16 DS4_ROCM_STREAM_REPLICA=$REPLICA D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json D14_BUDGET_MIN=70 \
    DS4_NOTE="D16b: S1 $mode to 32k, lambda $lam keep 3, int4 + d16" \
    $H $tag s1 32768 "$extra" grow:target=32768:label=grow ); rc=$?
  echo "$(date -Is) end $tag rc=$rc" >> $LOG; }
sess d14-d16b-i4-s1-plain-l0 0 plain
sess d14-d16b-i4-s1-plain-l09 0.09 plain
sess d14-d16b-i4-s1-dsp-l0 0 dsp
sess d14-d16b-i4-s1-dsp-l09 0.09 dsp
echo "$(date -Is) QUEUE DONE (stage2)" >> $LOG
