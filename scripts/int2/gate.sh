#!/usr/bin/env bash
# kumanix-int2 correctness gate (plain greedy token-ID digests):
#  a. build defaults: MXFP4 code (--nothink, DS4_D0_LOGITS=auto: the exactness
#     reference), prose, hash, long (256); Q2 code (--nothink)
#  b. the same five with DS4_ROCM_D5_PREFETCH=1
#  c. everything reverted (DS4_ROCM_STREAM_CACHE_SIZING=legacy
#     DS4_ROCM_C1_NORM=0 DS4_ROCM_C1_HC_PROJ=0, prefetch off): MXFP4 code, prose
#     (also the kumanix-int speed check: code 8.45-8.48, prose 9.12-9.17)
#  d. DSpark --dspark-verify exact on MXFP4 code against a: free run
#     (DS4_REF_TAG, as kumanix-int) and teacher-forced (DS4_REF_FORCE=1)
# Exit 0 only when every digest passes.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) GATE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
fail=0
gate() { # TAG KIND(code|prose|hash|long|q2code) [env...]
  local tag=$1 k=$2; shift 2; local p=$k; [ $k = q2code ] && p=code
  read -r -a A <<< "$(args $p)"
  if [ $k = q2code ]; then q2; else mx; fi
  ( [ $# -gt 0 ] && export "$@"; DS4_NOTE="int2 gate $tag${*:+ ($*)}" run $tag $(ntok $p) "${A[@]}" ) || exit 1
  read -r r n <<< "$(ref $k)"; check $tag $r $n || fail=1; }
gate int2-gate-mx-code code DS4_D0_LOGITS=auto
gate int2-gate-mx-prose prose
gate int2-gate-mx-hash hash
gate int2-gate-mx-long long
gate int2-gate-q2-code q2code
gate int2-gate-pf-mx-code code DS4_ROCM_D5_PREFETCH=1
gate int2-gate-pf-mx-prose prose DS4_ROCM_D5_PREFETCH=1
gate int2-gate-pf-mx-hash hash DS4_ROCM_D5_PREFETCH=1
gate int2-gate-pf-mx-long long DS4_ROCM_D5_PREFETCH=1
gate int2-gate-pf-q2-code q2code DS4_ROCM_D5_PREFETCH=1
REV="DS4_ROCM_STREAM_CACHE_SIZING=legacy DS4_ROCM_C1_NORM=0 DS4_ROCM_C1_HC_PROJ=0"
gate int2-gate-rev-mx-code code $REV
gate int2-gate-rev-mx-prose prose $REV
mx
DS4_REF_TAG=int2-gate-mx-code DS4_NOTE="int2 gate DSpark verify exact MXFP4 code (free)" \
  run int2-gate-mx-code-exact 512 $E/p-code.txt --nothink --dspark --mtp-model $DSP --dspark-verify exact
DS4_REF_TAG=int2-gate-mx-code DS4_REF_FORCE=1 DS4_NOTE="int2 gate DSpark verify exact MXFP4 code (teacher-forced)" \
  run int2-gate-mx-code-exact-tf 512 $E/p-code.txt --nothink --dspark --mtp-model $DSP --dspark-verify exact
for t in int2-gate-mx-code-exact int2-gate-mx-code-exact-tf; do
  r=$(rec $t)
  a=$(echo "$r" | grep -o '"div_tf_agree": *[0-9.]*' | grep -o '[0-9.]*$')
  b=$(echo "$r" | grep -o '"div_bitexact_frac": *[0-9.]*' | grep -o '[0-9.]*$')
  m=$(echo "$r" | grep -o '"div_logit_maxabs": *[0-9.e-]*' | grep -o '[0-9.e-]*$')
  v=$(awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN{print (a == 1 && b == 1) ? "PASS" : "FAIL"}'); [ $v = PASS ] || fail=1
  echo "$(date -Is) EXACT $t tf_agree=$a bitexact_frac=$b logit_maxabs=$m $v" >> $LOG
done
echo "$(date -Is) GATE QUEUE DONE fail=$fail" >> $LOG
exit $fail
