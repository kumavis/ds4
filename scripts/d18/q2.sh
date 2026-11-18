#!/usr/bin/env bash
# D18 item 4: which decode stage grows with context. C1 events profiler on the
# d18 build, plain MXFP4, 64 tokens after a 2k / 8k / 30k prompt (measurement only).
set -u
LOG=/data/ds4-bench/d18-queue.log
export DS4_DIR=/data/ds4-d18
. /data/ds4-d12/scripts/d12/common.sh
LOG=/data/ds4-bench/d18-queue.log
for p in 2k 8k 30k; do
  erun DS4_CTX=32768 DS4_ROCM_C1_PROF=events DS4_NOTE="d18 C1 events profile, decode after a $p prompt" \
    d18-c1ev-ps-$p 64 $E/p-ps-$p.txt
done
echo "$(date -Is) QUEUE DONE (q2)" >> $LOG
