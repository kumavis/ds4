#!/usr/bin/env bash
# D13 queue D (after queue B): the D5 shelf hand-back path. Prefetch on in all
# runs; four-turn sequence (two sweeps) staging off vs on, and one plain code
# run with both switches on against the reference digest.
set -u
while systemctl --user is-active --quiet d13-qseq.service; do sleep 60; done
. /data/ds4-d13/scripts/d13/common.sh
s() { echo "$(date -Is) start $1" >> $LOG; /data/ds4-d13/scripts/d13/seq.sh "$@"
  echo "$(date -Is) end $1 $(tr "\n" " " < /data/ds4-bench/d13-seq/$1/meta.txt)" >> $LOG; }
export DS4_ROCM_D5_PREFETCH=1
s d13-seq2-d5-off off 96
s d13-seq2-d5-on on 96
mx
erun $ON d13-mx-code-plain-d5-on 512 $(args code)
echo "$(date -Is) QUEUE D DONE" >> $LOG
