#!/usr/bin/env bash
# D13 queue B: gates and measurements, switch off vs on, same binary,
# off and on interleaved per prompt. Plain "on" runs double as the token-ID
# gates; DSpark "on" runs as the rounds/accepted gates.
set -u
. /data/ds4-d13/scripts/d13/common.sh
echo "$(date -Is) QUEUE B start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
plain_round() { local r=$1; mx
  for p in code prose hash long; do
    erun $OFF d13-mx-$p-plain-off-r$r $(ntok $p) $(args $p)
    erun $ON  d13-mx-$p-plain-on-r$r  $(ntok $p) $(args $p)
  done; }
dsp_round() { local r=$1; mx
  for p in code prose json translate list; do
    erun $OFF $DSPE d13-mx-$p-dsp-off-r$r 512 $(args $p) $DSPF
    erun $ON  $DSPE d13-mx-$p-dsp-on-r$r  512 $(args $p) $DSPF
  done; }
plain_round 1
# Exact verify against plain decode of the same binary, switch on: code
# (token-major prompt) and json (layer-sweep prompt).
mx
for p in code json; do
  erun $ON DS4_D0_LOGITS=auto d13-gate-ref-mx-$p-on 512 $(args $p)
  erun $ON $DSPE DS4_REF_TAG=d13-gate-ref-mx-$p-on d13-gate-mx-$p-exact-on 512 $(args $p) $DSPF --dspark-verify exact
done
dsp_round 1
q2
erun $OFF d13-q2-code-plain-off 512 $(args code)
erun $ON  d13-q2-code-plain-on  512 $(args code)
erun $OFF $DSPE d13-q2-code-dsp-off 512 $(args code) $DSPF
erun $ON  $DSPE d13-q2-code-dsp-on  512 $(args code) $DSPF
plain_round 2
# D13: DSpark repeat 2 cut on the project manager's decision (2026-11-17 22:3x); edited in place while queue B ran.
echo "$(date -Is) dsp_round 2 skipped (PM decision: DSpark repeat 2 cut)" >> $LOG
echo "$(date -Is) QUEUE B DONE" >> $LOG
