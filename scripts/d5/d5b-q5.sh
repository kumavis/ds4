#!/usr/bin/env bash
# D5b queue 5 (binary /data/ds4-d5x, 80b832f): DSpark with verify budget 2
# and no layer-0 probe (the best code settings of queues 1/4), two runs each
# on code/json/translate, logits vs queue 1's off-r1.
set -u; LOG=/data/ds4-bench/d5b-q5.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_DSPARK_SCHED=old
P=/data/ds4-env
for r in 1 2; do
  for p in code json translate; do
    a="$P/p-$p.txt"; [ $p = code ] && a="$a --nothink"
    DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_VERIFY_K=2 DS4_ROCM_D5_VERIFY_L0=0 DS4_REF_TAG=d5b-b-$p-off-r1 \
      DS4_NOTE="D5b-f on (verify K 2, L0 off)" run d5bf-b-$p-k2l0-r$r 512 $a --dspark --mtp-model $DSP
  done
done
echo "$(date -Is) QUEUE DONE" >> $LOG
