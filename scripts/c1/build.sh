#!/usr/bin/env bash
# Build ds4 in /data/ds4-c1 under the GPU lock; log to build.log with BUILD_EXIT=.
cd /data/ds4-c1 || exit 1
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; exit $rc
