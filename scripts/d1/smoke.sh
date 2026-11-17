#!/usr/bin/env bash
# D1 smoke: MXFP4 DSpark under SSD streaming, -n 8 then -n 64, replica on.
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1
P=/data/ds4-env/p-code.txt
DS4_NOTE="D1 smoke: MXFP4 SSD DSpark, auto cache, 2 drives" run d1-mx-smoke-8 8 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="D1 smoke: MXFP4 SSD DSpark, auto cache, 2 drives" run d1-mx-smoke-64 64 $P --nothink --dspark --mtp-model $DSP
echo "$(date -Is) SMOKE DONE" >> $LOG
