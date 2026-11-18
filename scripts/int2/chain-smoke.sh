#!/usr/bin/env bash
# kumanix-int2 step 1: build (make strix-halo under the GPU lock), model-free
# tests, then the smoke runs of the new binary (-n 8, then -n 64). Stops after
# the smoke so the memory floor can be checked before the gate queue.
set -u
cd /data/ds4-int2 || exit 1
. scripts/int2/common.sh
echo "$(date -Is) BUILD start $(git rev-parse --short HEAD)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
scripts/int2/tests.sh
echo "$(date -Is) tests done: $(grep "^TEST\|^C1BUILD\|^TESTBUILD" /data/ds4-bench/int2-tests.log | tr '\n' ' ')" >> $LOG
exec scripts/int2/smoke.sh
