#!/usr/bin/env bash
# D16 step 1: build (make strix-halo under the GPU lock), then the smoke runs
# of the new binary (-n 8, -n 64 at lambda 0; -n 64 at lambda 0.09, plain and
# DSpark, fixed cache). Stops after the smoke.
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
echo "$(date -Is) BUILD start $(git rev-parse --short HEAD)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
echo "$(date -Is) SMOKE QUEUE start $(git rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="d16 smoke plain lambda 0" run d16-smoke-mx-8 8 $P --nothink
check d16-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="d16 smoke plain lambda 0" run d16-smoke-mx-64 64 $P --nothink
check d16-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.09 DS4_NOTE="d16 smoke plain lambda 0.09 fixed cache" \
  run d16-smoke-mx-l09-64 64 $P --nothink --ssd-streaming-cache-experts $MXP
DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.09 DS4_NOTE="d16 smoke DSpark lambda 0.09 fixed cache" \
  run d16-smoke-mx-dsp-l09-64 64 $P --nothink --dspark --mtp-model $DSP --ssd-streaming-cache-experts $MXD
echo "$(date -Is) QUEUE DONE (smoke)" >> $LOG
