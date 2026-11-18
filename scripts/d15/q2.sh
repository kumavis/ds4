#!/usr/bin/env bash
# D15 part B, second queue: teacher-forced divergence of the Q8 pipe.
# The plain CLI decode ignores DS4_D0_FORCE (d15-q8-tf-* ran free), so the
# forced runs use DSpark with --dspark-verify exact (bit-identical to plain
# decode, D0/int2), as int2-gate-mx-code-exact-tf was run.
set -u
LOG=/data/ds4-bench/d15-queue.log
. /data/ds4-d12/scripts/d12/common.sh
ON="DS4_ROCM_C1_Q8_PIPE=1 DS4_OUTPUT_CLASS=fp-order"
echo "$(date -Is) D15 q2 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
erun $ON DS4_REF_TAG=d15-q8-ref-code DS4_REF_FORCE=1 DS4_NOTE="d15 Q8 pipe on code, teacher-forced (DSpark exact verify)" \
  d15-q8-tfx-code 512 $E/p-code.txt --nothink --dspark --mtp-model $DSP --dspark-verify exact
erun $ON DS4_REF_TAG=d15-q8-ref-prose DS4_REF_FORCE=1 DS4_NOTE="d15 Q8 pipe on prose, teacher-forced (DSpark exact verify)" \
  d15-q8-tfx-prose 512 $E/p-prose.txt --dspark --mtp-model $DSP --dspark-verify exact
echo "$(date -Is) QUEUE DONE (q2)" >> $LOG
