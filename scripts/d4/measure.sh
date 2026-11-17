#!/usr/bin/env bash
# D4 part 2: memory-sized automatic cache (DS4_ROCM_STREAM_CACHE_SIZING,
# default memory) against legacy sizing, same binary. DSpark runs use the new
# scheduler default (old; DS4_DSPARK_SCHED unset). Plain runs are also the
# gate: every plain run's token-ID digest is logged (DIGEST lines).
#  0b. the DSpark curve above auto (4219, top), see below
#  0. smoke of the new binary at the new size: plain/DSpark -n 8, -n 64; Q2
#     plain/DSpark -n 8, -n 64 (a new, larger cache size for each)
#  1. MXFP4 two reps, legacy then memory: plain code/prose/hash/long,
#     DSpark code/prose/json/translate/list
#  2. Q2 single runs: plain code and DSpark code, legacy and memory
#  3. long-prompt safety: p-ps-8k at ctx 16384, legacy -n 8 (measures the
#     ctx-16384 footprint), then, only if the prediction from it stays above
#     the floor, memory -n 8 and memory -n 64
# Memory guard: before the first run at a new size the MemAvailable minimum is
# predicted from the closest finished run (common.sh guard); runs predicted
# below 12 GiB are skipped and logged.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) MEASURE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD) floor=${FLOOR_MB}MiB" >> $LOG
C="$E/p-code.txt --nothink"
DSPF="--dspark --mtp-model $DSP"
dsp() { DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 run "$@"; }
args() { case $1 in code) echo "$E/p-code.txt --nothink";; *) echo "$E/p-$1.txt";; esac; }
ntok() { case $1 in long) echo 256;; *) echo 512;; esac; }
leg() { DS4_ROCM_STREAM_CACHE_SIZING=legacy "$@"; }
# dynamic slots a finished run chose (V4.1 SSD expert storage line)
dyn() { grep -o "dynamic ([0-9]* experts)" "$(rundir $1)/run.log" | head -1 | grep -o "[0-9]*"; }
memline() { grep -h "memory-sized SSD cache\|dynamic slots\|fixed = non-routed\|SSD expert storage\|planned$" "$(rundir $1)/run.log" | sed "s/^/    /" >> $LOG; }
# Predicted dynamic slots of the memory sizing right now, from a finished
# legacy run of the same model/ctx/drafter: fixed = planned - dynamic + 2 GiB.
pred_dyn() { local d=$(rundir $1) avail
  avail=$(awk '/MemAvailable/{print $2/1048576}' /proc/meminfo)
  grep -o "= [0-9.]* GiB planned" $d/run.log | head -1 | grep -o "[0-9.]*" | {
    read -r plan; dl=$(grep -o "+ [0-9.]* GiB dynamic" $d/run.log | head -1 | grep -o "[0-9.]*")
    slot=$(slotmb $1)
    awk -v a="$avail" -v p="$plan" -v dl="$dl" -v r=13 -v sl="$slot" 'BEGIN{printf "%d", (a-(p-dl+2)-r)*1024/sl}'; }; }
# MiB per expert slot of a finished run (two-layer staging = 768 slots)
slotmb() { grep -o "SSD expert storage [0-9.]* GiB = [0-9.]* GiB" "$(rundir $1)/run.log" | head -1 \
  | awk '{printf "%.3f", $(NF-1)*1024/768}'; }
# guard a memory-sized run against legacy reference REF (legacy slots from its log)
guard_mem() { local ref=$1 n; n=$(pred_dyn $ref)
  echo "$(date -Is) memory sizing predicted dynamic slots now: $n (from $ref: legacy $(dyn $ref) configured, $(res $ref) resident at its peak)" >> $LOG
  SLOT_MB=$2 guard $ref "$(res $ref)" "$n"; }
# resident peak slots of a finished run (a short run may not fill the cache)
res() { slots $1 | cut -d' ' -f1; }

mx
# 0. smoke (legacy first: today's size, safe; then the new size)
leg run d4-m-smoke-plain-leg-8 8 $C
guard_mem d4-m-smoke-plain-leg-8 17.94 || { echo "$(date -Is) QUEUE DONE (memory guard)" >> $LOG; exit 1; }
# check against the curve's top point too (plain 4780 slots, two runs)
guard d4-c-pc-4780-r2 4780 "$(pred_dyn d4-m-smoke-plain-leg-8)" || { echo "$(date -Is) QUEUE DONE (memory guard vs curve top)" >> $LOG; exit 1; }
DS4_NOTE="d4 smoke memory sizing" run d4-m-smoke-plain-8 8 $C; memline d4-m-smoke-plain-8
DS4_NOTE="d4 smoke memory sizing" run d4-m-smoke-plain-64 64 $C
# 0b. finish the DSpark curve above auto. The kumanix-int binary refused
#     4219 at start-up (admission double-charges the drafter, see D4.md); this
#     binary fixes that under the memory sizing, so these runs set it and pass
#     the slot count by hand (manual sizes are not re-sized). Scheduler old.
cref=d4-c-dc-3969; crefn=3969
dcurve() { local n=$1 reps=$2 r sfx
  guard $cref $crefn $n || return 1
  DS4_DSPARK_SCHED=old DS4_NOTE="d4 smoke dspark N=$n" dsp d4-c-d-$n-s8b 8 $C $DSPF --ssd-streaming-cache-experts $n
  cref=d4-c-d-$n-s8b crefn=$n; guard $cref $crefn $n || return 1
  DS4_DSPARK_SCHED=old DS4_NOTE="d4 smoke dspark N=$n" dsp d4-c-d-$n-s64b 64 $C $DSPF --ssd-streaming-cache-experts $n
  cref=d4-c-d-$n-s64b
  for r in $(seq 1 $reps); do
    sfx=; [ $reps -gt 1 ] && sfx=-r$r
    guard $cref $crefn $n || return 1
    DS4_DSPARK_SCHED=old DS4_NOTE="d4 curve dspark code N=$n (D4 binary)" dsp d4-c-dc-$n$sfx 512 $C $DSPF --ssd-streaming-cache-experts $n
    cref=d4-c-dc-$n$sfx
  done; return 0; }
dcurve 4219 1 && {
  top=$(maxn $cref $crefn); [ "$top" -gt 4469 ] && top=4469
  echo "$(date -Is) dspark top size $top (from $cref @ $crefn)" >> $LOG
  [ "$top" -gt 4219 ] && dcurve $top 2; }
leg dsp d4-m-smoke-dsp-leg-8 8 $C $DSPF
guard_mem d4-m-smoke-dsp-leg-8 17.94 || { echo "$(date -Is) QUEUE DONE (memory guard)" >> $LOG; exit 1; }
guard $cref $crefn "$(pred_dyn d4-m-smoke-dsp-leg-8)" || { echo "$(date -Is) QUEUE DONE (memory guard vs curve top)" >> $LOG; exit 1; }
DS4_NOTE="d4 smoke memory sizing" dsp d4-m-smoke-dsp-8 8 $C $DSPF; memline d4-m-smoke-dsp-8
DS4_NOTE="d4 smoke memory sizing" dsp d4-m-smoke-dsp-64 64 $C $DSPF
# 1. MXFP4, two reps, legacy then memory, interleaved
for rep in 1 2; do
  for p in code prose hash long; do
    read -r -a A <<< "$(args $p)"
    DS4_NOTE="d4 legacy sizing plain $p" leg run d4-mx-$p-plain-leg-r$rep $(ntok $p) "${A[@]}"
    DS4_NOTE="d4 memory sizing plain $p" run d4-mx-$p-plain-mem-r$rep $(ntok $p) "${A[@]}"
    echo "$(date -Is) DIGEST d4-mx-$p-plain-leg-r$rep $(digest d4-mx-$p-plain-leg-r$rep)" >> $LOG
    echo "$(date -Is) DIGEST d4-mx-$p-plain-mem-r$rep $(digest d4-mx-$p-plain-mem-r$rep)" >> $LOG
  done
  for p in code prose json translate list; do
    read -r -a A <<< "$(args $p)"
    DS4_NOTE="d4 legacy sizing DSpark $p" leg dsp d4-mx-$p-dsp-leg-r$rep 512 "${A[@]}" $DSPF
    DS4_NOTE="d4 memory sizing DSpark $p" dsp d4-mx-$p-dsp-mem-r$rep 512 "${A[@]}" $DSPF
  done
done
# 2. Q2
q2
leg run d4-q2-smoke-leg-8 8 $C
guard_mem d4-q2-smoke-leg-8 "$(slotmb d4-q2-smoke-leg-8)" || { echo "$(date -Is) QUEUE DONE (memory guard)" >> $LOG; exit 1; }
DS4_NOTE="d4 smoke memory sizing Q2" run d4-q2-smoke-8 8 $C; memline d4-q2-smoke-8
DS4_NOTE="d4 smoke memory sizing Q2" run d4-q2-smoke-64 64 $C
DS4_NOTE="d4 smoke memory sizing Q2 DSpark" dsp d4-q2-smoke-dsp-8 8 $C $DSPF; memline d4-q2-smoke-dsp-8
DS4_NOTE="d4 smoke memory sizing Q2 DSpark" dsp d4-q2-smoke-dsp-64 64 $C $DSPF
DS4_NOTE="d4 legacy sizing Q2 plain code" leg run d4-q2-code-plain-leg 512 $C
DS4_NOTE="d4 memory sizing Q2 plain code" run d4-q2-code-plain-mem 512 $C
echo "$(date -Is) DIGEST d4-q2-code-plain-leg $(digest d4-q2-code-plain-leg)" >> $LOG
echo "$(date -Is) DIGEST d4-q2-code-plain-mem $(digest d4-q2-code-plain-mem)" >> $LOG
DS4_NOTE="d4 legacy sizing Q2 DSpark code" leg dsp d4-q2-code-dsp-leg 512 $C $DSPF
DS4_NOTE="d4 memory sizing Q2 DSpark code" dsp d4-q2-code-dsp-mem 512 $C $DSPF
# 3. long-prompt safety, ctx 16384
mx; export DS4_CTX=16384
DS4_NOTE="d4 safety legacy ps-8k ctx16384" leg run d4-ps8k-leg-8 8 $E/p-ps-8k.txt
memline d4-ps8k-leg-8
if guard_mem d4-ps8k-leg-8 17.94; then
  DS4_NOTE="d4 safety memory ps-8k ctx16384" run d4-ps8k-mem-8 8 $E/p-ps-8k.txt; memline d4-ps8k-mem-8
  guard d4-ps8k-mem-8 "$(res d4-ps8k-mem-8)" "$(dyn d4-ps8k-mem-8)" && \
    DS4_NOTE="d4 safety memory ps-8k ctx16384" run d4-ps8k-mem-64 64 $E/p-ps-8k.txt
fi
unset DS4_CTX
echo "$(date -Is) QUEUE DONE" >> $LOG
