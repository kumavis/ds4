#!/usr/bin/env bash
# D16 sweep3: DSpark (scheduler old) code at the two lambda chosen from the
# plain code curve: 0.09 (recipe production point, the knee: +24% vs +26% at
# 0.2, first divergence 30 vs 11) and 0.05 (least drift). Fixed cache $MXD,
# keep 3, reference d16-mx-code-dsp-l0 (DS4_REF_TAG, free run).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) SWEEP3 QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
DSPT="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old"
mx; read -r -a A <<< "$(args code)"
for lam in 0.09 0.05; do
  t=d16-mx-code-dsp-l$(echo $lam | tr -d .| sed "s/^0//")
  ( export $DSPT DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=$lam DS4_ROCM_ROUTE_CACHE_KEEP=3 DS4_REF_TAG=d16-mx-code-dsp-l0
    DS4_NOTE="d16 $t (lambda $lam keep 3 dsp)" run $t 512 "${A[@]}" --dspark --mtp-model $DSP --ssd-streaming-cache-experts $MXD ) || exit 1
done
echo "$(date -Is) QUEUE DONE (sweep3)" >> $LOG
