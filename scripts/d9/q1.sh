#!/usr/bin/env bash
# D9 queue 1 (build 1). NOTE: its `env $TR ... run` lines never ran (env cannot call a shell function); q1b.sh is the version that ran.: smoke, plain gate, per-position probe runs (the
# drafter's full proposal at every position of the plain trajectory), traced
# old-scheduler runs, a drafter profile, Q2 probe/old.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) Q1 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="d9 smoke plain" run d9-smoke-plain-8 8 $P --nothink
env $TR DS4_D9_PROBE=1 DS4_DSPARK_SCHED=old DS4_NOTE="d9 smoke probe" run d9-smoke-probe-8 8 $P --nothink --dspark --mtp-model $DSP
env $TR DS4_DSPARK_SCHED=model DS4_NOTE="d9 smoke model" run d9-smoke-model-8 8 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="d9 smoke plain" run d9-smoke-plain-64 64 $P --nothink
env $TR DS4_D9_PROBE=1 DS4_DSPARK_SCHED=old DS4_NOTE="d9 smoke probe" run d9-smoke-probe-64 64 $P --nothink --dspark --mtp-model $DSP
env $TR DS4_DSPARK_SCHED=model DS4_NOTE="d9 smoke model" run d9-smoke-model-64 64 $P --nothink --dspark --mtp-model $DSP
DS4_NOTE="d9 gate plain MXFP4 code" run d9-gate-mx-code 512 $P --nothink
echo "$(date -Is) DIGEST d9-gate-mx-code $(digest d9-gate-mx-code) (want 8290541d324b 511)" >> $LOG
for p in code prose json translate list; do
  read -r -a A <<< "$(args $p)"
  env $TR DS4_D9_PROBE=1 DS4_DSPARK_SCHED=old DS4_REF_TAG=int-mx-$p-plain-r1 \
    DS4_NOTE="d9 probe $p (draft every position, plain step)" \
    run d9-probe-mx-$p 512 "${A[@]}" --dspark --mtp-model $DSP
done
for p in code prose json translate list; do
  read -r -a A <<< "$(args $p)"
  env $TR DS4_DSPARK_SCHED=old DS4_REF_TAG=int-mx-$p-plain-r1 \
    DS4_NOTE="d9 traced old $p" run d9-trace-old-mx-$p 512 "${A[@]}" --dspark --mtp-model $DSP
done
read -r -a A <<< "$(args code)"
env $TR DS4_DSPARK_SCHED=old DS4_ROCM_DSPARK_V41_PROPOSE_PROFILE=100000 DS4_D8_PROPOSE_DETAIL=1 \
  DS4_NOTE="d9 drafter profile code" run d9-drafter-prof-mx-code 512 "${A[@]}" --dspark --mtp-model $DSP
q2
env $TR DS4_D9_PROBE=1 DS4_DSPARK_SCHED=old DS4_NOTE="d9 probe Q2 code" \
  run d9-probe-q2-code 512 "${A[@]}" --dspark --mtp-model $DSP
env $TR DS4_DSPARK_SCHED=old DS4_NOTE="d9 traced old Q2 code" \
  run d9-trace-old-q2-code 512 "${A[@]}" --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
