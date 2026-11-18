#!/usr/bin/env bash
# D9 queue 2 (build 2, fitted model defaults): smoke, plain gates (MXFP4 and
# Q2 digests), exact-verify gates under the model scheduler (code, prose),
# measurement (old/model x2, online x1 per prompt; Q2 code old/model/online),
# and two teacher-forced probes along the traced old trajectories (replay
# validation).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) Q2 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
erun $TR DS4_DSPARK_SCHED=model DS4_NOTE="d9 smoke model b2" d9-b2-smoke-model-8 8 $P --nothink --dspark --mtp-model $DSP
erun $TR DS4_DSPARK_SCHED=model DS4_NOTE="d9 smoke model b2" d9-b2-smoke-model-64 64 $P --nothink --dspark --mtp-model $DSP
erun DS4_NOTE="d9 gate plain MXFP4 code" d9-b2-gate-mx-code 512 $P --nothink
echo "$(date -Is) DIGEST d9-b2-gate-mx-code $(digest d9-b2-gate-mx-code) (want 8290541d324b 511)" >> $LOG
erun $TR DS4_DSPARK_SCHED=model DS4_REF_TAG=int-gate-mx-code DS4_NOTE="d9 gate model exact code" \
  d9-gate-model-exact-code 512 $P --nothink --dspark --mtp-model $DSP --dspark-verify exact
erun $TR DS4_DSPARK_SCHED=model DS4_REF_TAG=int-mx-prose-plain-r1 DS4_NOTE="d9 gate model exact prose" \
  d9-gate-model-exact-prose 512 $E/p-prose.txt --dspark --mtp-model $DSP --dspark-verify exact
for rep in 1 2; do
  for p in code prose json translate list; do
    read -r -a A <<< "$(args $p)"
    erun $TR DS4_DSPARK_SCHED=old DS4_REF_TAG=int-mx-$p-plain-r1 DS4_NOTE="d9 measure old $p" \
      d9-mx-$p-old-r$rep 512 "${A[@]}" --dspark --mtp-model $DSP
    erun $TR DS4_DSPARK_SCHED=model DS4_REF_TAG=int-mx-$p-plain-r1 DS4_NOTE="d9 measure model $p" \
      d9-mx-$p-model-r$rep 512 "${A[@]}" --dspark --mtp-model $DSP
  done
done
for p in code prose json translate list; do
  read -r -a A <<< "$(args $p)"
  erun $TR DS4_DSPARK_SCHED=online DS4_DSPARK_SCHED_TRACE=1 DS4_REF_TAG=int-mx-$p-plain-r1 DS4_NOTE="d9 measure online $p" \
    d9-mx-$p-online-r1 512 "${A[@]}" --dspark --mtp-model $DSP
done
read -r -a A <<< "$(args code)"
for p in code translate; do
  read -r -a A <<< "$(args $p)"
  erun $TR DS4_D9_PROBE=1 DS4_DSPARK_SCHED=old DS4_REF_TAG=d9-trace-old-mx-$p DS4_REF_FORCE=1 \
    DS4_NOTE="d9 probe forced along the traced old trajectory $p" d9-probe-old-traj-mx-$p 512 "${A[@]}" --dspark --mtp-model $DSP
done
q2; read -r -a A <<< "$(args code)"
erun DS4_NOTE="d9 gate plain Q2 code" d9-b2-gate-q2-code 512 "${A[@]}"
echo "$(date -Is) DIGEST d9-b2-gate-q2-code $(digest d9-b2-gate-q2-code) (want f0a2cd53a247 511)" >> $LOG
for s in old model online; do
  erun $TR DS4_DSPARK_SCHED=$s DS4_NOTE="d9 measure Q2 $s code" d9-q2-code-$s 512 "${A[@]}" --dspark --mtp-model $DSP
done
echo "$(date -Is) QUEUE DONE" >> $LOG
