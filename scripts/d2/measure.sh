#!/usr/bin/env bash
# D2 measurements, MXFP4, two drives, same binary, old path (DS4_ROCM_D2_*=0,
# i.e. D1) vs new, 2 reps interleaved.  Plain: code (--nothink), prose, hash
# at 512 tokens, long at 256.  DSpark default and backoff off on code and
# prose.  Then a Q2 plain pair (one drive).
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
arm() { if [ $1 = old ]; then export DS4_ROCM_D2_DECODE=0 DS4_ROCM_D2_VERIFY=0;
        else unset DS4_ROCM_D2_DECODE DS4_ROCM_D2_VERIFY; fi; }
for rep in 1 2; do
  for pr in code prose hash long; do
    X=; [ $pr = code ] && X=--nothink; n=512; [ $pr = long ] && n=256
    for a in old new; do
      arm $a; DS4_NOTE="D2 plain $a $pr" run d2-mx-$pr-plain-$a-r$rep $n /data/ds4-env/p-$pr.txt $X
    done
  done
  export DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SPEC_LOG=1
  for pr in code prose; do
    X=; [ $pr = code ] && X=--nothink
    for bo in dsp dspnb; do
      if [ $bo = dspnb ]; then export DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING=1;
      else unset DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING; fi
      for a in old new; do
        arm $a; DS4_NOTE="D2 $bo $a $pr" run d2-mx-$pr-$bo-$a-r$rep 512 /data/ds4-env/p-$pr.txt $X \
          --dspark --mtp-model $DSP
      done
    done
  done
  unset DS4_ROCM_DSPARK_PHASE_TRACE DS4_DSPARK_SPEC_LOG DS4_DSPARK_DISABLE_ADAPTIVE_DRAFTING
done
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
for a in old new; do
  arm $a; DS4_NOTE="D2 Q2 regression $a" run d2-q2-code-plain-$a 512 /data/ds4-env/p-code.txt --nothink
done
echo "$(date -Is) MEASURE DONE" >> $LOG
