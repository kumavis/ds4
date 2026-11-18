#!/usr/bin/env bash
# D5b debug (binary /data/ds4-d5x: 4b91934 + busy cap + shelf dump): json DSpark -n 8, -n 64.
set -u; LOG=/data/ds4-bench/d5b-dbg.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_DSPARK_SCHED=old
P=/data/ds4-env
DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_DEBUG=1 DS4_NOTE="D5b debug shelf" run d5bx-dbg-json-8 8 $P/p-json.txt --dspark --mtp-model $DSP
DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_DEBUG=1 DS4_NOTE="D5b debug shelf" run d5bx-dbg-json-64 64 $P/p-json.txt --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
