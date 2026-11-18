#!/usr/bin/env bash
# kumanix-int4 queue 3: the standard suite twice (labels int4, int4r2), as
# INTEGRATION2.md did, after queue 2. Same env as int2 suite runs.
set -u
while systemctl --user is-active --quiet int4-q2.service; do sleep 120; done
LOG=/data/ds4-bench/int4-queue.log
echo "$(date -Is) QUEUE 3 (suite) start $(git -C /data/ds4-int4 rev-parse --short HEAD)" >> $LOG
export DS4_DIR=/data/ds4-int4 DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1 \
  DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
for l in int4 int4r2; do /data/ds4-env/suite.sh $l > /dev/null 2>&1; echo "$(date -Is) suite $l rc=$?" >> $LOG; done
echo "$(date -Is) QUEUE 3 DONE" >> $LOG
