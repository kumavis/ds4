#!/usr/bin/env bash
# D16 extras: (a) the exact-verify gate at lambda 0: DSpark --dspark-verify
# exact on code, logits against d16-mx-code-plain-l0 (must be bit-exact);
# (b) run-to-run repeatability at lambda 0.09: plain code again, digest
# must equal d16-mx-code-plain-l09 (983db0a4a5bb/511).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) EXTRA QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; read -r -a A <<< "$(args code)"
DS4_REF_TAG=d16-mx-code-plain-l0 DS4_NOTE="d16 gate DSpark verify exact code lambda 0" \
  run d16-gate-dsp-code-exact 512 "${A[@]}" --dspark --mtp-model $DSP --dspark-verify exact --ssd-streaming-cache-experts $MXD
r=$(rec d16-gate-dsp-code-exact)
echo "$(date -Is) EXACT d16-gate-dsp-code-exact $(echo "$r" | grep -o "\"div_tf_agree\": *[0-9.]*\|\"div_bitexact_frac\": *[0-9.]*\|\"div_logit_maxabs\": *[0-9.]*\|\"div_compared\": *[0-9]*" | tr -d " " | tr "\n" " ")" >> $LOG
DS4_OUTPUT_CLASS=approx-routing DS4_ROCM_ROUTE_CACHE_PRIOR=0.09 DS4_ROCM_ROUTE_CACHE_KEEP=3 DS4_NOTE="d16 repeat lambda 0.09 plain code" \
  run d16-mx-code-plain-l09-r2 512 "${A[@]}" --ssd-streaming-cache-experts $MXP
check d16-mx-code-plain-l09-r2 983db0a4a5bb 511
echo "$(date -Is) QUEUE DONE (extra)" >> $LOG
