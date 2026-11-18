#!/usr/bin/env bash
# D16b stage 2b: rerun of the DSpark lambda 0.09 session, which the harness
# memory guard (12,288 + 512 MiB) ended at about 17.9k fill (its memory-sized
# cache had 20 more slots than the lambda 0 twin).
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
LOG=/data/ds4-bench/d16b-queue.log
H=/data/ds4-d16/scripts/d16b/harness/run.sh
m=0; t0=$(date +%s)
while :; do m=0; for f in /sys/class/nvme/nvme*/hwmon*/temp1_input; do t=$(( $(cat $f) / 1000 )); [ $t -gt $m ] && m=$t; done
  [ $m -le 80 ] && break; [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 20; done
echo "$(date -Is) cool: hottest NVMe ${m} C; start d14-d16b-i4-s1-dsp-l09-r2" >> $LOG
DS4_DSPARK_SCHED=old DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.09 DS4_ROCM_ROUTE_CACHE_KEEP=3 \
DS4_DIR=/data/ds4-d16 DS4_ROCM_STREAM_REPLICA=$REPLICA D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json D14_BUDGET_MIN=70 \
DS4_NOTE="D16b: S1 dsp to 32k, lambda 0.09 keep 3, int4 + d16 (rerun after mem-guard stop)" \
  $H d14-d16b-i4-s1-dsp-l09-r2 s1 32768 "--mtp-model $DSP --dspark" grow:target=32768:label=grow
echo "$(date -Is) end d14-d16b-i4-s1-dsp-l09-r2 rc=$?" >> $LOG
echo "$(date -Is) QUEUE DONE (stage2b)" >> $LOG
