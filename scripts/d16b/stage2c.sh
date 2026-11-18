#!/usr/bin/env bash
# D16b stage 2c: plain code at lambda 0.05 on the memory-sized cache (the
# DSpark curve suggests the knee moved down on int4: 0.05 gives 18.30 against
# 18.92 tok/s at 0.09). Runs after stage 2b.
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
LOG=/data/ds4-bench/d16b-queue.log
while systemctl --user is-active --quiet d16b-stage2b; do sleep 30; done
mx; read -r -a A <<< "$(args code)"
DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.05 DS4_ROCM_ROUTE_CACHE_KEEP=3 DS4_REF_TAG=d16b-i4-code-plain-l0 \
  DS4_NOTE="d16b d16b-i4-code-plain-l05 (lambda 0.05 keep 3 plain, memory-sized cache)" \
  run d16b-i4-code-plain-l05 512 "${A[@]}"
echo "$(date -Is) QUEUE DONE (stage2c)" >> $LOG
