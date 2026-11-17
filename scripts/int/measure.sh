#!/usr/bin/env bash
# Integration measurements (kumanix-int), after the gate passed. MXFP4 first,
# DS4_TOKEN_TIMING=1 and DS4_D8_STEP_TRACE=1 on every run (as d8 queue B),
# default (automatic) stream cache, two drives on MXFP4.
#  a. suite.sh int (replica set; Q2 refuses it harmlessly: size differs)
#  b. MXFP4 512 tokens, prompts code (--nothink), prose, json, translate,
#     list; per rep: plain (DS4_D0_LOGITS=auto, the references), DSpark with
#     DS4_DSPARK_SCHED=old, DSpark with =online; DSpark runs carry
#     DS4_REF_TAG=int-mx-<p>-plain-r1 (divergence fields). Reps interleaved.
#  c. one teacher-forced DSpark run per prompt (DS4_REF_FORCE=1, default
#     verify setting, scheduler online)
#  d. Q2 code: plain, DSpark old, DSpark online (single runs)
set -u; . "$(dirname "$0")/common.sh"
export DS4_D8_STEP_TRACE=1
echo "$(date -Is) MEASURE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
args() { case $1 in code) echo "$E/p-code.txt --nothink";; *) echo "$E/p-$1.txt";; esac; }
# a
mx
echo "$(date -Is) start suite-int (4 runs)" >> $LOG
/data/ds4-env/suite.sh int > /dev/null 2>&1; src=$?
for t in suite-int-mxfp4-decode suite-int-mxfp4-long suite-int-q2-decode suite-int-q2-long; do
  d=$(ls -dt /data/ds4-bench/runs/*-"$t" 2>/dev/null | head -1)
  echo "$(date -Is) end $t suite_rc=$src $(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1) decode_tps=$(field $t decode_tps) avail_min_mb=$(field $t avail_min_mb) ssd_max_c=$(field $t ssd_temp_max_c) dir=$d" >> $LOG
done
[ $src = 0 ] || { echo "$(date -Is) STOP after suite (rc=$src)" >> $LOG; exit 1; }
# b
mx
for rep in 1 2; do
  for p in code prose json translate list; do
    read -r -a A <<< "$(args $p)"
    DS4_D0_LOGITS=auto DS4_NOTE="int plain $p" run int-mx-$p-plain-r$rep 512 "${A[@]}"
    DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old \
      DS4_REF_TAG=int-mx-$p-plain-r1 DS4_NOTE="int DSpark sched old $p" \
      run int-mx-$p-old-r$rep 512 "${A[@]}" --dspark --mtp-model $DSP
    DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=online \
      DS4_REF_TAG=int-mx-$p-plain-r1 DS4_NOTE="int DSpark sched online $p" \
      run int-mx-$p-online-r$rep 512 "${A[@]}" --dspark --mtp-model $DSP
  done
done
# c
for p in code prose json translate list; do
  read -r -a A <<< "$(args $p)"
  DS4_DSPARK_SCHED=online DS4_REF_TAG=int-mx-$p-plain-r1 DS4_REF_FORCE=1 \
    DS4_NOTE="int teacher-forced DSpark online $p" run int-mx-$p-tf 512 "${A[@]}" --dspark --mtp-model $DSP
done
# d
q2; read -r -a A <<< "$(args code)"
DS4_NOTE="int Q2 plain code" run int-q2-code-plain 512 "${A[@]}"
DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=old DS4_NOTE="int Q2 DSpark sched old code" \
  run int-q2-code-old 512 "${A[@]}" --dspark --mtp-model $DSP
DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_DSPARK_SCHED=online DS4_NOTE="int Q2 DSpark sched online code" \
  run int-q2-code-online 512 "${A[@]}" --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
