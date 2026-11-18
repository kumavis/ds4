#!/usr/bin/env bash
# Build kumanix-int4 under the GPU lock, then run the given command (e.g. tests, a queue).
set -u
cd /data/ds4-int4
L=/data/ds4-bench/int4-queue.log
echo "$(date -Is) build start $(git rev-parse --short HEAD)" >> $L
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) build done rc=$rc" >> $L
[ $rc = 0 ] || { echo "$(date -Is) QUEUE DONE (build failed)" >> $L; exit 1; }
for c in "$@"; do "$c" || { echo "$(date -Is) chain stopped after $c" >> $L; exit 1; }; done
