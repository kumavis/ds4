#!/usr/bin/env bash
# D2 smoke on MXFP4, two drives: plain then DSpark, -n 8 then -n 64.
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
P=/data/ds4-env/p-code.txt
DS4_NOTE="D2 smoke plain" run d2-smoke-plain-8 8 $P --nothink
DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_NOTE="D2 smoke DSpark" \
  run d2-smoke-dsp-8 8 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="D2 smoke plain" run d2-smoke-plain-64 64 $P --nothink
DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_NOTE="D2 smoke DSpark" \
  run d2-smoke-dsp-64 64 $P --nothink --dspark --mtp-model $DSP
echo "$(date -Is) SMOKE DONE" >> $LOG
