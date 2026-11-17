#!/usr/bin/env bash
# D3 smoke: new binary, MXFP4 DSpark under SSD streaming with the online
# scheduler, -n 8 then -n 64, two drives.
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1
P=/data/ds4-env/p-code.txt
DS4_NOTE="D3 smoke: online scheduler" run d3-mx-smoke2-8 8 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="D3 smoke: online scheduler" run d3-mx-smoke2-64 64 $P --nothink --dspark --mtp-model $DSP
echo "$(date -Is) SMOKE DONE" >> $LOG
