#!/usr/bin/env bash
# D20 q3: build (adds the measurement-only dense Q8_0 matmul mark under
# DS4_ROCM_D20_PROF); WMMA layout test (model-free, tests/test_d20_wmma_layout_rocm.cpp);
# smoke; profiles of the routed MoE and the dense Q8_0 matmuls with stage 9 on
# at about 2,048 + 842, 1,097 and 325 rows; then the stage-9 session twin:
# D14 S1 plain to about 32k, DS4_ROCM_D20_DOWN_STAGE9 off and on, same binary.
set -u
cd /data/ds4-d20 || exit 1
. scripts/d20/common.sh
echo "$(date -Is) Q3 BUILD start $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo && hipcc -O3 --offload-arch=gfx1151 tests/test_d20_wmma_layout_rocm.cpp -o tests/test_d20_wmma_layout_rocm && ./tests/test_d20_wmma_layout_rocm > /data/ds4-bench/d20-test-wmma-layout.log 2>&1; echo LAYOUT_RC=\$? >> /data/ds4-bench/d20-test-wmma-layout.log" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; echo "$(date -Is) BUILD_EXIT=$rc layout: $(tail -3 /data/ds4-bench/d20-test-wmma-layout.log | tr "\n" " ")" >> $LOG
[ $rc = 0 ] || stop "build failed"
mx; P=$E/p-code.txt
DS4_NOTE="d20 q3 smoke" run d20-q3-smoke-8 8 $P --nothink
check d20-q3-smoke-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="d20 q3 smoke" run d20-q3-smoke-64 64 $P --nothink
check d20-q3-smoke-64 3098d05a1cef 63 || stop "smoke digest"
for f in p2890 p1150 p300; do
  DS4_ROCM_D20_DOWN_STAGE9=1 DS4_ROCM_D20_PROF=1 DS4_NOTE="d20 q3 profile (routed + dense q8_0) stage9 $f" run d20-q3-prof-$f 8 /data/ds4-bench/d20-$f.txt
done
H=/data/ds4-d20/scripts/d20/harness/run.sh
cool() { local t0=$(date +%s) m t
  while :; do m=0; for f in /sys/class/nvme/nvme*/hwmon*/temp1_input; do t=$(( $(cat $f) / 1000 )); [ $t -gt $m ] && m=$t; done
    [ $m -le 80 ] && break; [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 20; done
  echo "$(date -Is) cool: hottest NVMe ${m} C after $(( $(date +%s) - t0 )) s" >> $LOG; }
sess() { # TAG [env...]
  local tag=$1; shift
  cool; echo "$(date -Is) start $tag" >> $LOG
  ( [ $# -gt 0 ] && export "$@"
    DS4_DIR=/data/ds4-d20 DS4_ROCM_STREAM_REPLICA=$REPLICA D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json D14_BUDGET_MIN=70 \
    DS4_NOTE="D20: S1 plain to 32k, ${*:-stage9 off}" \
    $H $tag s1 32768 "" grow:target=32768:label=grow ); rc=$?
  echo "$(date -Is) end $tag rc=$rc" >> $LOG; }
sess d14-d20-s1-plain-off
sess d14-d20-s1-plain-s9 DS4_ROCM_D20_DOWN_STAGE9=1
echo "$(date -Is) QUEUE DONE (q3)" >> $LOG
