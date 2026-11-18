#!/usr/bin/env bash
# kumanix-int3 correctness gate.
#  a. build defaults (prefetch off): MXFP4 code (--nothink, DS4_D0_LOGITS=auto:
#     the logits reference), prose, hash, long (256); Q2 code (--nothink)
#  b. the same five with DS4_ROCM_D5_PREFETCH=1; code also compares logits
#     against a (DS4_REF_TAG)
#  c. code, prefetch on, DS4_ROCM_D5_TEMP_LIMIT_C=1: digest equal and every
#     speculative issue held (issued=0, held>0)
#  d. DSpark --dspark-verify exact, prefetch on, MXFP4 code vs a: free run
#     and teacher-forced (DS4_REF_FORCE=1)
#  e. DSpark fast verify, scheduler old, prefetch off (logits reference) vs on
#     (DS4_REF_TAG = the off run): code, json, translate; identical rounds,
#     accepted drafts, hit/miss counts and bit-exact logits
# Exit 0 only when everything passes.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) GATE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
fail=0
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o '[0-9.eE+-]*$'; }
gate() { # TAG KIND(code|prose|hash|long|q2code) [env...]
  local tag=$1 k=$2; shift 2; local p=$k; [ $k = q2code ] && p=code
  read -r -a A <<< "$(args $p)"
  if [ $k = q2code ]; then q2; else mx; fi
  ( [ $# -gt 0 ] && export "$@"; DS4_NOTE="int3 gate $tag${*:+ ($*)}" run $tag $(ntok $p) "${A[@]}" ) || exit 1
  read -r r n <<< "$(ref $k)"; check $tag $r $n || fail=1; }
divcheck() { # TAG: div_tf_agree == 1 and div_bitexact_frac == 1
  local a b m c v; a=$(num $1 div_tf_agree); b=$(num $1 div_bitexact_frac); m=$(num $1 div_logit_maxabs); c=$(num $1 div_compared)
  v=$(awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN{print (a == 1 && b == 1) ? "PASS" : "FAIL"}'); [ $v = PASS ] || fail=1
  echo "$(date -Is) EXACT $1 vs $(rec $1 | grep -o '"ref_tag": *"[^"]*"' | cut -d'"' -f4) compared=$c tf_agree=$a bitexact_frac=$b logit_maxabs=$m $v" >> $LOG; }
gate int3-gate-mx-code code DS4_D0_LOGITS=auto
gate int3-gate-mx-prose prose
gate int3-gate-mx-hash hash
gate int3-gate-mx-long long
gate int3-gate-q2-code q2code
gate int3-gate-pf-mx-code code DS4_ROCM_D5_PREFETCH=1 DS4_REF_TAG=int3-gate-mx-code
divcheck int3-gate-pf-mx-code
gate int3-gate-pf-mx-prose prose DS4_ROCM_D5_PREFETCH=1
gate int3-gate-pf-mx-hash hash DS4_ROCM_D5_PREFETCH=1
gate int3-gate-pf-mx-long long DS4_ROCM_D5_PREFETCH=1
gate int3-gate-pf-q2-code q2code DS4_ROCM_D5_PREFETCH=1
gate int3-gate-pf-mx-code-hold code DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_TEMP_LIMIT_C=1
d=$(rundir int3-gate-pf-mx-code-hold)
iss=$(grep -o "D5 prefetch: diag.*issued=[0-9]*" $d/run.log | tail -1 | grep -o "issued=[0-9]*" | cut -d= -f2)
held=$(grep -o "D5 thermal: .*held=[0-9]*" $d/run.log | tail -1 | grep -o " held=[0-9]*" | cut -d= -f2)
v=FAIL; [ "${iss:-x}" = 0 ] && [ "${held:-0}" -gt 0 ] && v=PASS; [ $v = PASS ] || fail=1
echo "$(date -Is) HOLD int3-gate-pf-mx-code-hold issued=${iss:-?} held=${held:-?} $v" >> $LOG
mx; P=$E/p-code.txt
DS4_ROCM_D5_PREFETCH=1 DS4_REF_TAG=int3-gate-mx-code DS4_NOTE="int3 gate DSpark verify exact, prefetch on, MXFP4 code (free)" \
  run int3-gate-pf-mx-code-exact 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact
divcheck int3-gate-pf-mx-code-exact
DS4_ROCM_D5_PREFETCH=1 DS4_REF_TAG=int3-gate-mx-code DS4_REF_FORCE=1 DS4_NOTE="int3 gate DSpark verify exact, prefetch on, MXFP4 code (teacher-forced)" \
  run int3-gate-pf-mx-code-exact-tf 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact
divcheck int3-gate-pf-mx-code-exact-tf
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
for p in code json translate; do
  read -r -a A <<< "$(args $p)"
  ( export $DSPT DS4_D0_LOGITS=auto; DS4_NOTE="int3 gate DSpark fast old, prefetch off $p (reference)" \
      run int3-gate-dsp-$p-off 512 "${A[@]}" --dspark --mtp-model $DSP ) || exit 1
  ( export $DSPT DS4_ROCM_D5_PREFETCH=1 DS4_REF_TAG=int3-gate-dsp-$p-off; DS4_NOTE="int3 gate DSpark fast old, prefetch on $p" \
      run int3-gate-dsp-$p-on 512 "${A[@]}" --dspark --mtp-model $DSP ) || exit 1
  divcheck int3-gate-dsp-$p-on
  same=PASS; line=""
  for f in dspark_cycles dspark_accepted decode_hits decode_misses prefill_hits prefill_misses; do
    a=$(num int3-gate-dsp-$p-off $f); b=$(num int3-gate-dsp-$p-on $f); line="$line $f=$a/$b"
    [ "$a" = "$b" ] || same=FAIL
  done
  [ $same = PASS ] || fail=1
  echo "$(date -Is) COUNTS int3-gate-dsp-$p off/on$line $same" >> $LOG
done
echo "$(date -Is) GATE QUEUE DONE fail=$fail" >> $LOG
exit $fail
