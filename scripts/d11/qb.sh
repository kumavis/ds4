#!/usr/bin/env bash
# D11 queue B (d11-drafter-experts build): smoke, gates, trace-off/on
# equality, drafter expert-selection traces on five prompts and Q2 code.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) QB start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
erun DS4_NOTE="d11 smoke plain" d11-smoke-plain-8 8 $P --nothink
erun $TR DS4_D11_DRAFT_TRACE=1 DS4_NOTE="d11 smoke DSpark trace" d11-smoke-dsp-8 8 $P --nothink --dspark --mtp-model $DSP
erun DS4_NOTE="d11 smoke plain" d11-smoke-plain-64 64 $P --nothink
erun $TR DS4_D11_DRAFT_TRACE=1 DS4_NOTE="d11 smoke DSpark trace" d11-smoke-dsp-64 64 $P --nothink --dspark --mtp-model $DSP
erun DS4_NOTE="d11 gate plain MXFP4 code" d11-gate-mx-code 512 $P --nothink
echo "$(date -Is) DIGEST d11-gate-mx-code $(digest d11-gate-mx-code) (want 8290541d324b/511)" >> $LOG
erun $TR DS4_NOTE="d11 gate DSpark old code, trace off" d11-gate-dsp-code-off 512 $P --nothink --dspark --mtp-model $DSP
erun $TR DS4_D11_DRAFT_TRACE=1 DS4_NOTE="d11 DSpark old code, drafter trace" d11-trace-mx-code 512 $P --nothink --dspark --mtp-model $DSP
for p in prose json translate list; do
  read -r -a A <<< "$(args $p)"
  erun $TR DS4_D11_DRAFT_TRACE=1 DS4_NOTE="d11 DSpark old $p, drafter trace" d11-trace-mx-$p 512 "${A[@]}" --dspark --mtp-model $DSP
done
q2
erun DS4_NOTE="d11 gate plain Q2 code" d11-gate-q2-code 512 $P --nothink
echo "$(date -Is) DIGEST d11-gate-q2-code $(digest d11-gate-q2-code) (want f0a2cd53a247/511)" >> $LOG
erun $TR DS4_D11_DRAFT_TRACE=1 DS4_NOTE="d11 DSpark old Q2 code, drafter trace" d11-trace-q2-code 512 $P --nothink --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
