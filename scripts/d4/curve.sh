#!/usr/bin/env bash
# D4 part 1: decode speed and misses against dynamic expert-cache slots
# (--ssd-streaming-cache-experts N = dynamic slots; the two-layer prefill
# staging, 768 slots / 13.45 GiB, is allocated on top). kumanix-int binary.
# MXFP4, two drives, ctx 4096, 512 tokens. Plain code/prose and DSpark code
# (DS4_DSPARK_SCHED=old). Auto sizes: plain 4179, DSpark 3719 dynamic slots.
# Sizes above auto: -n 8, then -n 64, then the measurement; each guarded by
# the predicted MemAvailable minimum (common.sh guard, floor 12 GiB).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) CURVE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD) floor=${FLOOR_MB}MiB" >> $LOG
mx
C="$E/p-code.txt --nothink"; P="$E/p-prose.txt"
DSPF="--dspark --mtp-model $DSP"
PA=4179; DA=3719
dsp() { DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old run "$@"; }
# new binary smoke at the automatic size
DS4_NOTE="d4 smoke plain auto" run d4-smoke-plain-8 8 $C
DS4_NOTE="d4 smoke dspark auto" dsp d4-smoke-dsp-8 8 $C $DSPF
DS4_NOTE="d4 smoke plain auto" run d4-smoke-plain-64 64 $C
DS4_NOTE="d4 smoke dspark auto" dsp d4-smoke-dsp-64 64 $C $DSPF
# below auto (less memory than today: safe)
for n in 3179 3679; do
  DS4_NOTE="d4 curve plain code N=$n" run d4-c-pc-$n 512 $C --ssd-streaming-cache-experts $n
  DS4_NOTE="d4 curve plain prose N=$n" run d4-c-pp-$n 512 $P --ssd-streaming-cache-experts $n
done
for n in 2719 3219; do
  DS4_NOTE="d4 curve dspark code N=$n" dsp d4-c-dc-$n 512 $C $DSPF --ssd-streaming-cache-experts $n
done
# automatic size, two runs each (no flag)
for r in 1 2; do
  DS4_NOTE="d4 curve plain code auto" run d4-c-pc-auto-r$r 512 $C
  DS4_NOTE="d4 curve plain prose auto" run d4-c-pp-auto-r$r 512 $P
  DS4_NOTE="d4 curve dspark code auto" dsp d4-c-dc-auto-r$r 512 $C $DSPF
done
# above auto, plain: 4429, 4679, then the largest the floor allows (two runs)
ref=d4-c-pp-auto-r2; refn=$PA   # prose and code have the same footprint; take the lower
[ "$(field d4-c-pc-auto-r2 avail_min_mb)" -lt "$(field $ref avail_min_mb)" ] && ref=d4-c-pc-auto-r2
up_plain() { local n=$1 reps=$2
  guard $ref $refn $n || return 1
  DS4_NOTE="d4 smoke plain N=$n" run d4-c-p-$n-s8 8 $C --ssd-streaming-cache-experts $n
  ref=d4-c-p-$n-s8 refn=$n; guard $ref $refn $n || return 1
  DS4_NOTE="d4 smoke plain N=$n" run d4-c-p-$n-s64 64 $C --ssd-streaming-cache-experts $n
  ref=d4-c-p-$n-s64
  for r in $(seq 1 $reps); do
    sfx=; [ $reps -gt 1 ] && sfx=-r$r
    guard $ref $refn $n || return 1
    DS4_NOTE="d4 curve plain code N=$n" run d4-c-pc-$n$sfx 512 $C --ssd-streaming-cache-experts $n
    ref=d4-c-pc-$n$sfx
    DS4_NOTE="d4 curve plain prose N=$n" run d4-c-pp-$n$sfx 512 $P --ssd-streaming-cache-experts $n
    [ "$(field d4-c-pp-$n$sfx avail_min_mb)" -lt "$(field $ref avail_min_mb)" ] && ref=d4-c-pp-$n$sfx
  done; return 0; }
up_plain 4429 1 && up_plain 4679 1 && {
  top=$(maxn $ref $refn); [ "$top" -gt 4929 ] && top=4929
  echo "$(date -Is) plain top size $top (from $ref @ $refn)" >> $LOG
  [ "$top" -gt 4679 ] && up_plain $top 2; }
# above auto, DSpark code
ref=d4-c-dc-auto-r2; refn=$DA
up_dsp() { local n=$1 reps=$2
  guard $ref $refn $n || return 1
  DS4_NOTE="d4 smoke dspark N=$n" dsp d4-c-d-$n-s8 8 $C $DSPF --ssd-streaming-cache-experts $n
  ref=d4-c-d-$n-s8 refn=$n; guard $ref $refn $n || return 1
  DS4_NOTE="d4 smoke dspark N=$n" dsp d4-c-d-$n-s64 64 $C $DSPF --ssd-streaming-cache-experts $n
  ref=d4-c-d-$n-s64
  for r in $(seq 1 $reps); do
    sfx=; [ $reps -gt 1 ] && sfx=-r$r
    guard $ref $refn $n || return 1
    DS4_NOTE="d4 curve dspark code N=$n" dsp d4-c-dc-$n$sfx 512 $C $DSPF --ssd-streaming-cache-experts $n
    ref=d4-c-dc-$n$sfx
  done; return 0; }
up_dsp 3969 1 && up_dsp 4219 1 && {
  top=$(maxn $ref $refn); [ "$top" -gt 4469 ] && top=4469
  echo "$(date -Is) dspark top size $top (from $ref @ $refn)" >> $LOG
  [ "$top" -gt 4219 ] && up_dsp $top 2; }
echo "$(date -Is) QUEUE DONE" >> $LOG
