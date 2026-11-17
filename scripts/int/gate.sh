#!/usr/bin/env bash
# Integration smoke (-n 8, -n 64) and correctness gate: plain greedy token-ID
# digests against the pre-integration references, the D2 revert, and
# DSpark --dspark-verify exact against plain (DS4_REF_TAG divergence fields).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) GATE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="int smoke plain" run int-smoke-mx-plain-8 8 $P --nothink
DS4_NOTE="int smoke DSpark" run int-smoke-mx-dsp-8 8 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="int smoke plain" run int-smoke-mx-plain-64 64 $P --nothink
DS4_NOTE="int smoke DSpark" run int-smoke-mx-dsp-64 64 $P --nothink --dspark --mtp-model $DSP
DS4_D0_LOGITS=auto DS4_NOTE="int gate plain MXFP4 code" run int-gate-mx-code 512 $P --nothink
DS4_NOTE="int gate plain MXFP4 prose" run int-gate-mx-prose 512 $E/p-prose.txt
DS4_NOTE="int gate plain MXFP4 hash" run int-gate-mx-hash 512 $E/p-hash.txt
DS4_NOTE="int gate plain MXFP4 long" run int-gate-mx-long 256 $E/p-long.txt
DS4_ROCM_D2_DECODE=0 DS4_ROCM_D2_VERIFY=0 DS4_NOTE="int gate plain MXFP4 code, D2 off" \
  run int-gate-mx-code-d2off 512 $P --nothink
DS4_REF_TAG=int-gate-mx-code DS4_NOTE="int gate DSpark verify exact MXFP4 code" \
  run int-gate-mx-code-exact 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact
q2
DS4_NOTE="int gate plain Q2 code" run int-gate-q2-code 512 $P --nothink
for t in int-gate-mx-code int-gate-mx-prose int-gate-mx-hash int-gate-mx-long \
         int-gate-mx-code-d2off int-gate-mx-code-exact int-gate-q2-code; do
  echo "$(date -Is) DIGEST $t $(digest $t)" >> $LOG
done
echo "$(date -Is) GATE QUEUE DONE" >> $LOG
