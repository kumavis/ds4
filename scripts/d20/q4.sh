#!/usr/bin/env bash
# D20 q4 (worktree /data/ds4-d20w, branch d20-wmma): step 3, int8 WMMA kernels
# for experts with at least DS4_ROCM_D20_WMMA_MIN (16) pairs.
#  1. build; model-free test tests/test_d20_wmma_rocm.c (bitwise against the
#     tile8 kernels at 1..2040 pairs, gate/up and down, plus per-launch times
#     for the crossover); stop here if any bit differs;
#  2. in-model with DS4_ROCM_D20_WMMA=1: smoke, profiles at about 2,048 + 842,
#     1,097 and 325 rows, gates (code 8290541d324b/511, long c27a3b087fb5/255,
#     DSpark rounds code/json/translate 172/340, 207/305, 176/336, exact
#     verify on code bit-exact against the plain code run).
set -u
cd /data/ds4-d20w || exit 1
. scripts/d20/common.sh
export DS4_DIR=/data/ds4-d20w
echo "$(date -Is) Q4 BUILD start d20w $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo && scripts/d0/build-incremental.sh tests/test_d20_wmma_rocm && ./tests/test_d20_wmma_rocm > /data/ds4-bench/d20-test-wmma.log 2>&1; echo WMMA_TEST_RC=\$? >> /data/ds4-bench/d20-test-wmma.log" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) BUILD_EXIT=$rc wmma test: $(grep -c "|" /data/ds4-bench/d20-test-wmma.log) rows, $(tail -2 /data/ds4-bench/d20-test-wmma.log | tr "\n" " ")" >> $LOG
[ $rc = 0 ] || stop "build failed"
grep -q "WMMA_TEST_RC=0" /data/ds4-bench/d20-test-wmma.log || stop "wmma test not bit-exact (see d20-test-wmma.log)"
W=(DS4_ROCM_D20_WMMA=1)
mx; P=$E/p-code.txt
( export "${W[@]}"; DS4_NOTE="d20w wmma smoke" run d20w-smoke-8 8 $P --nothink ) || exit 1
check d20w-smoke-8 b1cd778da121 7 || stop "smoke digest"
( export "${W[@]}"; DS4_NOTE="d20w wmma smoke" run d20w-smoke-64 64 $P --nothink ) || exit 1
check d20w-smoke-64 3098d05a1cef 63 || stop "smoke digest"
for f in p2890 p1150 p300; do
  ( export "${W[@]}" DS4_ROCM_D20_PROF=1; DS4_NOTE="d20w wmma profile $f" run d20w-prof-$f 8 /data/ds4-bench/d20-$f.txt ) || exit 1
done
num() { rec "$1" | grep -o "\"$2\": *[0-9.eE+-]*" | head -1 | grep -o "[0-9.eE+-]*$"; }
DS4_D0_LOGITS=auto DS4_NOTE="d20w reference plain code (wmma off)" run d20w-ref-code 512 $P --nothink
check d20w-ref-code 8290541d324b 511
( export "${W[@]}"; DS4_NOTE="d20w wmma gate code" run d20w-code 512 $P --nothink ); check d20w-code 8290541d324b 511
( export "${W[@]}"; DS4_NOTE="d20w wmma gate long" run d20w-long 256 $E/p-long.txt ); check d20w-long c27a3b087fb5 255
for p in code json translate; do
  read -r -a A <<< "$(args $p)"
  ( export "${W[@]}" DS4_DSPARK_SCHED=old; DS4_NOTE="d20w wmma gate DSpark $p" run d20w-dsp-$p 512 "${A[@]}" --dspark --mtp-model $DSP )
  r="$(num d20w-dsp-$p dspark_cycles)/$(num d20w-dsp-$p dspark_accepted)"
  case $p in code) w=172/340;; json) w=207/305;; translate) w=176/336;; esac
  echo "$(date -Is) ROUNDS d20w-dsp-$p $r ref $w $([ "$r" = "$w" ] && echo PASS || echo FAIL)" >> $LOG
done
( export "${W[@]}" DS4_REF_TAG=d20w-ref-code; DS4_NOTE="d20w wmma gate DSpark exact code" run d20w-dsp-code-exact 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact )
echo "$(date -Is) EXACT d20w-dsp-code-exact tf_agree=$(num d20w-dsp-code-exact div_tf_agree) bitexact=$(num d20w-dsp-code-exact div_bitexact_frac) maxabs=$(num d20w-dsp-code-exact div_logit_maxabs)" >> $LOG

# threshold sweep (rows per expert at which the WMMA kernels take over),
# profiled, at 2,048 + 842 and 325 rows
for mn in 2 64; do
  for f in p2890 p300; do
    ( export "${W[@]}" DS4_ROCM_D20_WMMA_MIN=$mn DS4_ROCM_D20_PROF=1; DS4_NOTE="d20w wmma min $mn profile $f" run d20w-min$mn-prof-$f 8 /data/ds4-bench/d20-$f.txt ) || exit 1
  done
done
echo "$(date -Is) QUEUE DONE (q4)" >> $LOG
