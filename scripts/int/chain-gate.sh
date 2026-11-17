#!/usr/bin/env bash
# Wait for the strix-halo build, then the model-free tests and the gate queue.
set -u
L=/data/ds4-bench/int-queue.log
until grep -q BUILD_EXIT /data/ds4-int/build.log 2>/dev/null; do sleep 30; done
grep -q BUILD_EXIT=0 /data/ds4-int/build.log || { echo "$(date -Is) STOP build failed" >> $L; exit 1; }
echo "$(date -Is) build ok; tests" >> $L
/data/ds4-int/scripts/int/tests.sh
echo "$(date -Is) tests done: $(grep -c "EXIT=0" /data/ds4-bench/int-tests.log) of $(grep -c "EXIT=" /data/ds4-bench/int-tests.log) exit 0" >> $L
exec /data/ds4-int/scripts/int/gate.sh
