#!/usr/bin/env bash
# After d17-q-a ends: rebuild (under the lock), server unit tests, then q_a2.
set -u
while systemctl --user is-active --quiet d17-q-a; do sleep 60; done
cd /data/ds4-d17
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo > build-a2.log 2>&1 && scripts/d0/build-incremental.sh ds4_test >> build-a2.log 2>&1 && ./ds4_test --server >> build-a2.log 2>&1" < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> build-a2.log
[ $rc -eq 0 ] || exit 1
exec /data/ds4-d17/scripts/d17/q_a2.sh
