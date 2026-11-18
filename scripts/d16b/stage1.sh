#!/usr/bin/env bash
# D16b stage 1: build the trace binary, smoke (-n 8, -n 64), the long-prompt
# gate with the trace off and on (c27a3b087fb5/255 both; the trace must not
# change tokens), then one D14 agent session (S1, plain, MXFP4, two drives)
# to ~32k fill with DS4_ROCM_ROUTE_TRACE on.
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
LOG=/data/ds4-bench/d16b-queue.log
TR=/data/ds4-bench/d16b-trace-s1.bin
echo "$(date -Is) BUILD start $(git rev-parse --short HEAD)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
mx; P=$E/p-code.txt
DS4_NOTE="d16b smoke" run d16b-smoke-mx-8 8 $P --nothink
check d16b-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="d16b smoke" run d16b-smoke-mx-64 64 $P --nothink
check d16b-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
DS4_NOTE="d16b gate long, trace off" run d16b-gate-long 256 $E/p-long.txt
check d16b-gate-long c27a3b087fb5 255 || stop "gate digest"
rm -f /data/ds4-bench/d16b-trace-gate.bin
DS4_ROCM_ROUTE_TRACE=/data/ds4-bench/d16b-trace-gate.bin DS4_NOTE="d16b gate long, trace on" run d16b-gate-long-trace 256 $E/p-long.txt
check d16b-gate-long-trace c27a3b087fb5 255 || stop "gate digest (trace on)"
echo "$(date -Is) trace gate file $(stat -c %s /data/ds4-bench/d16b-trace-gate.bin 2>/dev/null) bytes" >> $LOG
rm -f $TR
echo "$(date -Is) start d14-d16b-s1-trace-32k" >> $LOG
DS4_DIR=/data/ds4-d16 DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_ROCM_ROUTE_TRACE=$TR \
  D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json D14_BUDGET_MIN=40 \
  DS4_NOTE="D16b stage 1: S1 plain to 32k with the prefill routing trace" \
  /data/ds4-d14/scripts/d14/run.sh d14-d16b-s1-trace-32k s1 32768 "" grow:target=32768:label=grow
echo "$(date -Is) end d14-d16b-s1-trace-32k rc=$? trace $(stat -c %s $TR 2>/dev/null) bytes" >> $LOG
echo "$(date -Is) QUEUE DONE (stage1)" >> $LOG
