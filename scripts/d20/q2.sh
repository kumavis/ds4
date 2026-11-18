#!/usr/bin/env bash
# D20 steps 1 and 2. Build; model-free bitwise test of the 9-block-staged down
# kernel against the original (decides the output class of every stage-9 run:
# PASS -> no declaration, FAIL -> DS4_OUTPUT_CLASS=fp-order); profiled
# prefills (DS4_ROCM_D20_PROF=1) with DS4_ROCM_D20_DOWN_STAGE9=1 at about
# 2,048 + 842, 1,097 and 325 rows; stage-9 gates (code 8290541d324b/511, long
# c27a3b087fb5/255, DSpark code/json/translate rounds 172/340, 207/305,
# 176/336, exact verify on code bit-exact against the plain code run); then
# the existing off-by-default variants, profiled at 2,048 + 842 rows.
set -u
cd /data/ds4-d20 || exit 1
. scripts/d20/common.sh
echo "$(date -Is) Q2 BUILD start $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo && scripts/d0/build-incremental.sh tests/test_d20_down_stage9_rocm" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "./tests/test_d20_down_stage9_rocm" > /data/ds4-bench/d20-test-stage9.log 2>&1 < /dev/null
trc=$?
echo "$(date -Is) TEST stage9 rc=$trc $(tail -2 /data/ds4-bench/d20-test-stage9.log | tr "\n" " ")" >> $LOG
CLS=(); [ $trc = 0 ] || CLS=(DS4_OUTPUT_CLASS=fp-order)
S9=(DS4_ROCM_D20_DOWN_STAGE9=1 "${CLS[@]}")
mx; P=$E/p-code.txt
( export "${S9[@]}"; DS4_NOTE="d20 stage9 smoke" run d20-s9-smoke-8 8 $P --nothink ) || exit 1
( export "${S9[@]}"; DS4_NOTE="d20 stage9 smoke" run d20-s9-smoke-64 64 $P --nothink ) || exit 1
for f in p2890 p1150 p300; do
  ( export "${S9[@]}" DS4_ROCM_D20_PROF=1; DS4_NOTE="d20 stage9 profile $f" run d20-s9-prof-$f 8 /data/ds4-bench/d20-$f.txt ) || exit 1
done
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o "[0-9.eE+-]*$"; }
DS4_D0_LOGITS=auto DS4_NOTE="d20 reference plain code (off)" run d20-ref-code 512 $P --nothink
check d20-ref-code 8290541d324b 511
( export "${S9[@]}"; DS4_NOTE="d20 stage9 gate code" run d20-s9-code 512 $P --nothink ); check d20-s9-code 8290541d324b 511
( export "${S9[@]}"; DS4_NOTE="d20 stage9 gate long" run d20-s9-long 256 $E/p-long.txt ); check d20-s9-long c27a3b087fb5 255
for p in code json translate; do
  read -r -a A <<< "$(args $p)"
  ( export "${S9[@]}" DS4_DSPARK_SCHED=old; DS4_NOTE="d20 stage9 gate DSpark $p" run d20-s9-dsp-$p 512 "${A[@]}" --dspark --mtp-model $DSP )
  r="$(num d20-s9-dsp-$p dspark_cycles)/$(num d20-s9-dsp-$p dspark_accepted)"
  case $p in code) w=172/340;; json) w=207/305;; translate) w=176/336;; esac
  echo "$(date -Is) ROUNDS d20-s9-dsp-$p $r ref $w $([ "$r" = "$w" ] && echo PASS || echo FAIL)" >> $LOG
done
( export "${S9[@]}" DS4_REF_TAG=d20-ref-code; DS4_NOTE="d20 stage9 gate DSpark exact code" run d20-s9-dsp-code-exact 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact )
echo "$(date -Is) EXACT d20-s9-dsp-code-exact tf_agree=$(num d20-s9-dsp-code-exact div_tf_agree) bitexact=$(num d20-s9-dsp-code-exact div_bitexact_frac) maxabs=$(num d20-s9-dsp-code-exact div_logit_maxabs)" >> $LOG
# step 2: existing variants (documented bit-exact), profiled at 2,048 + 842 rows
for v in DS4_ROCM_ENABLE_MXFP4_TILE4=1 DS4_ROCM_ENABLE_MXFP4_ROW64=1 DS4_ROCM_ENABLE_MXFP4_TILE32=1 DS4_ROCM_ENABLE_MXFP4_LDSB=1; do
  t=d20-var-$(echo ${v%%=*} | sed "s/DS4_ROCM_ENABLE_MXFP4_//" | tr A-Z a-z)
  ( export $v DS4_ROCM_D20_PROF=1; DS4_NOTE="d20 variant $v profile p2890" run $t 8 /data/ds4-bench/d20-p2890.txt )
done
( export "${S9[@]}" DS4_ROCM_MXFP4_DOWN_RGROUP=2 DS4_ROCM_D20_PROF=1; DS4_NOTE="d20 stage9 + down rgroup 2 profile p2890" run d20-var-s9-rgroup2 8 /data/ds4-bench/d20-p2890.txt )
echo "$(date -Is) QUEUE DONE (q2)" >> $LOG
