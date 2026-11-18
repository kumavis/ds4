#!/usr/bin/env bash
set -u
cd /data/ds4-d11
L=/data/ds4-bench/d11-queue-b.log
echo "$(date -Is) build start $(git rev-parse --short HEAD)" >> $L
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) build done rc=$rc" >> $L
[ $rc = 0 ] || { echo "$(date -Is) QUEUE DONE (build failed)" >> $L; exit 1; }
exec /data/ds4-d11/scripts/d11/qb.sh
