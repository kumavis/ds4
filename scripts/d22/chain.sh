#!/usr/bin/env bash
# D22 chain: wait for the build unit, then queue 1 (bench gates), then queue 2 (sessions).
while systemctl --user is-active --quiet d22-build-1.service; do sleep 30; done
grep -q "BUILD_EXIT=0" /data/ds4-d22/build.log || { echo "$(date -Is) chain: build not ok" >> /data/ds4-bench/d22-queue.log; exit 1; }
/data/ds4-d22/scripts/d22/q1.sh && /data/ds4-d22/scripts/d22/q2.sh
