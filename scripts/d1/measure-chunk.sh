#!/usr/bin/env bash
# D1 headline two-drive runs with the chunk replica policy (default from
# cd3e7a3; set explicitly here too): code x2 reps and prose rep 1 (prose rep 2
# of measure.sh already ran with chunk). 512 tokens, interleaved arms.
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_ROCM_STREAM_REPLICA_POLICY=chunk
arms() { pr=$1 rep=$2 P=$3; shift 3
  DS4_NOTE="D1 plain MXFP4 $pr 2d chunk" run d1-mx-$pr-2dc-plain-r$rep 512 $P "$@"
  DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_NOTE="D1 DSpark default MXFP4 $pr 2d chunk" \
    run d1-mx-$pr-2dc-dsp-r$rep 512 $P "$@" --dspark --mtp-model $DSP
  DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING=1 \
    DS4_NOTE="D1 DSpark backoff off MXFP4 $pr 2d chunk" \
    run d1-mx-$pr-2dc-dspnb-r$rep 512 $P "$@" --dspark --mtp-model $DSP; }
arms code 1 /data/ds4-env/p-code.txt --nothink
arms prose 1 /data/ds4-env/p-prose.txt
arms code 2 /data/ds4-env/p-code.txt --nothink
echo "$(date -Is) CHUNK QUEUE DONE" >> $LOG
