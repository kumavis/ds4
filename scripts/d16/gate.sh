#!/usr/bin/env bash
# D16 gates at lambda 0 (build defaults, memory-sized cache as int3): the five
# plain digests; code also with DS4_ROCM_ROUTE_CACHE_PRIOR=0 set explicitly;
# DSpark fast, scheduler old, code: 172 rounds / 340 accepted as int3;
# --dspark-verify exact on code: bit-exact against the plain code run.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) GATE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
fail=0
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o '[0-9.eE+-]*$'; }
gate() { local tag=$1 k=$2; shift 2; local p=$k; [ $k = q2code ] && p=code
  read -r -a A <<< "$(args $p)"
  if [ $k = q2code ]; then q2; else mx; fi
  ( [ $# -gt 0 ] && export "$@"; DS4_NOTE="d16 gate $tag${*:+ ($*)}" run $tag $(ntok $p) "${A[@]}" ) || exit 1
  read -r r n <<< "$(ref $k)"; check $tag $r $n || fail=1; }
gate d16-gate-mx-code code DS4_D0_LOGITS=auto
gate d16-gate-mx-prose prose
gate d16-gate-mx-hash hash
gate d16-gate-mx-long long
gate d16-gate-q2-code q2code
gate d16-gate-mx-code-l0 code DS4_ROCM_ROUTE_CACHE_PRIOR=0
mx; P=$E/p-code.txt
DS4_DSPARK_SCHED=old DS4_NOTE="d16 gate DSpark fast old code lambda 0" run d16-gate-dsp-code 512 $P --nothink --dspark --mtp-model $DSP
c=$(num d16-gate-dsp-code dspark_cycles); a=$(num d16-gate-dsp-code dspark_accepted)
v=FAIL; [ "$c/$a" = "172/340" ] && v=PASS; [ $v = PASS ] || fail=1
echo "$(date -Is) ROUNDS d16-gate-dsp-code $c/$a ref 172/340 $v" >> $LOG
DS4_REF_TAG=d16-gate-mx-code DS4_NOTE="d16 gate DSpark verify exact code lambda 0" \
  run d16-gate-dsp-code-exact 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact
a=$(num d16-gate-dsp-code-exact div_tf_agree); b=$(num d16-gate-dsp-code-exact div_bitexact_frac); m=$(num d16-gate-dsp-code-exact div_logit_maxabs)
v=$(awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN{print (a == 1 && b == 1) ? "PASS" : "FAIL"}'); [ $v = PASS ] || fail=1
echo "$(date -Is) EXACT d16-gate-dsp-code-exact tf_agree=$a bitexact_frac=$b logit_maxabs=$m $v" >> $LOG
echo "$(date -Is) GATE QUEUE DONE fail=$fail" >> $LOG
exit $fail
