#!/usr/bin/env bash
# D9 queue 3: teacher-forced (DS4_REF_FORCE=1 along the plain references)
# old vs model vs online, so every scheduler verifies against the same token
# trajectory (the replay's setting); free-running fast-verify runs drift to
# different texts per scheduler.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) Q3 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx
for p in code prose json translate list; do
  read -r -a A <<< "$(args $p)"
  for s in old model online; do
    erun $TR DS4_DSPARK_SCHED=$s DS4_REF_TAG=int-mx-$p-plain-r1 DS4_REF_FORCE=1 \
      DS4_NOTE="d9 teacher-forced $s $p" d9-tf-mx-$p-$s 512 "${A[@]}" --dspark --mtp-model $DSP
  done
done
echo "$(date -Is) QUEUE DONE" >> $LOG
