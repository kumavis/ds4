#!/usr/bin/env bash
# kumanix-int3 gate part e, rerun at a fixed cache size. In gate.sh the
# memory-sized cache differed between the off and on runs (MemAvailable at
# open varies run to run, and on charges the 287 MiB shelf), so hit/miss
# counts differed by a few even though rounds, drafts and logits were equal.
# Here both runs use --ssd-streaming-cache-experts 4240 dynamic slots (the
# lowest automatic DSpark count in gate.sh; 5000 was refused at admission,
# N counts dynamic slots): DSpark fast verify, scheduler old,
# prefetch off (logits reference) vs on (DS4_REF_TAG = off): code, json,
# translate; identical rounds, accepted drafts, hit/miss counts and
# bit-exact logits.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) GATE2 QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
fail=0
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o '[0-9.eE+-]*$'; }
divcheck() {
  local a b m c v; a=$(num $1 div_tf_agree); b=$(num $1 div_bitexact_frac); m=$(num $1 div_logit_maxabs); c=$(num $1 div_compared)
  v=$(awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN{print (a == 1 && b == 1) ? "PASS" : "FAIL"}'); [ $v = PASS ] || fail=1
  echo "$(date -Is) EXACT $1 vs $(rec $1 | grep -o '"ref_tag": *"[^"]*"' | cut -d'"' -f4) compared=$c tf_agree=$a bitexact_frac=$b logit_maxabs=$m $v" >> $LOG; }
mx
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
FIX="--ssd-streaming-cache-experts 4240"
for p in code json translate; do
  read -r -a A <<< "$(args $p)"
  ( export $DSPT DS4_D0_LOGITS=auto; DS4_NOTE="int3 gate2 DSpark fast old, fixed cache 4240, prefetch off $p (reference)" \
      run int3-gate2-dsp-$p-off 512 "${A[@]}" --dspark --mtp-model $DSP $FIX ) || exit 1
  ( export $DSPT DS4_ROCM_D5_PREFETCH=1 DS4_REF_TAG=int3-gate2-dsp-$p-off; DS4_NOTE="int3 gate2 DSpark fast old, fixed cache 4240, prefetch on $p" \
      run int3-gate2-dsp-$p-on 512 "${A[@]}" --dspark --mtp-model $DSP $FIX ) || exit 1
  divcheck int3-gate2-dsp-$p-on
  same=PASS; line=""
  for f in dspark_cycles dspark_accepted decode_hits decode_misses prefill_hits prefill_misses cache_experts; do
    a=$(num int3-gate2-dsp-$p-off $f); b=$(num int3-gate2-dsp-$p-on $f); line="$line $f=$a/$b"
    [ "$a" = "$b" ] || same=FAIL
  done
  [ $same = PASS ] || fail=1
  echo "$(date -Is) COUNTS int3-gate2-dsp-$p off/on$line $same" >> $LOG
done
echo "$(date -Is) GATE2 QUEUE DONE fail=$fail" >> $LOG
exit $fail
