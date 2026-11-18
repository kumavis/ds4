#!/usr/bin/env bash
# D13: second sequence gate (four turns, two layer sweeps), then queue B.
set -u
LOG=/data/ds4-bench/d13-queue.log
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
s() { echo "$(date -Is) start $1" >> $LOG; /data/ds4-d13/scripts/d13/seq.sh "$@"
  echo "$(date -Is) end $1 $(tr "\n" " " < /data/ds4-bench/d13-seq/$1/meta.txt)" >> $LOG; }
s d13-seq2-off off 96
s d13-seq2-on on 96
DS4_DSPARK_SCHED=old s d13-seq2-dspexact-on on 96 --dspark --mtp-model $DSP --dspark-verify exact
exec /data/ds4-d13/scripts/d13/qb.sh
