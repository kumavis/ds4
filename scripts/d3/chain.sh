#!/usr/bin/env bash
# Wait for the build, then smoke (-n 8, -n 64) and the measurement queue.
set -u
until grep -q BUILD_EXIT /data/ds4-d3/build.log 2>/dev/null; do sleep 30; done
grep -q BUILD_EXIT=0 /data/ds4-d3/build.log || { echo "$(date -Is) STOP build failed" >> /data/ds4-bench/d3-queue.log; exit 1; }
/data/ds4-d3/scripts/d3/smoke.sh || exit 1
/data/ds4-d3/scripts/d3/measure.sh
