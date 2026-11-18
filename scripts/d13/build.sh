#!/usr/bin/env bash
# Build the d13 worktree under the GPU lock; optional queue script to run after.
set -u
cd /data/ds4-d13
L=/data/ds4-bench/d13-queue.log
echo "$(date -Is) build start $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $L
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) build done rc=$rc" >> $L
[ $rc = 0 ] || { echo "$(date -Is) QUEUE DONE (build failed)" >> $L; exit 1; }
[ $# -gt 0 ] && exec "$@"
exit 0
