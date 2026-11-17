#!/usr/bin/env bash
# D1 measurements: MXFP4 512 tokens, {code --nothink, prose} x {2 drives, 1 drive}
# x {plain, DSpark default, DSpark backoff off}, 2 reps interleaved; then a Q2
# plain/DSpark pair (one drive).
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX
for rep in 1 2; do
 for pr in code prose; do
  if [ $pr = code ]; then P=/data/ds4-env/p-code.txt; X=--nothink; else P=/data/ds4-env/p-prose.txt; X=; fi
  for dr in 2d 1d; do
   if [ $dr = 2d ]; then export DS4_ROCM_STREAM_REPLICA=$REPLICA; else unset DS4_ROCM_STREAM_REPLICA; fi
   DS4_NOTE="D1 plain MXFP4 $pr $dr" run d1-mx-$pr-$dr-plain-r$rep 512 $P $X
   DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_NOTE="D1 DSpark default MXFP4 $pr $dr" \
     run d1-mx-$pr-$dr-dsp-r$rep 512 $P $X --dspark --mtp-model $DSP
   DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING=1 \
     DS4_NOTE="D1 DSpark backoff off MXFP4 $pr $dr" \
     run d1-mx-$pr-$dr-dspnb-r$rep 512 $P $X --dspark --mtp-model $DSP
  done
 done
done
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
P=/data/ds4-env/p-code.txt
DS4_NOTE="D1 Q2 regression plain" run d1-q2-code-plain 512 $P --nothink
DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1 DS4_NOTE="D1 Q2 regression DSpark default" \
  run d1-q2-code-dsp 512 $P --nothink --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
