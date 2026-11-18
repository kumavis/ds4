#!/usr/bin/env bash
# kumanix-int4 queue 2 (studies, after queue 1):
#  1. reclaim-with-shelf gate, longer decode: the 96-token sequence never put a
#     staging buffer in the D5 shelf (LRU victims were still the ~4,680 entries
#     prefill seeded). 320 tokens per turn gives ~12k misses before sweep 2.
#  2. model scheduler with the int4 refit (refit.py over int4-mx-*-dsp-r{1,2}:
#     P=98 VA=81 VB=38) on the five prompts, one run each (old = headline runs).
#  3. D5 prefetch on top of the staging cache: plain code, long, DSpark code.
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) QUEUE 2 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
P=$E/p-code.txt
DSPF="--dspark --mtp-model $DSP"
DSPE="DS4_DSPARK_SCHED_TRACE=1 DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_D9_ROUND_TRACE=1"
erun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  ( export "${e[@]}"; run "$@" ) || exit 1; }
for m in off on; do
  t=int4-seq-pf320-$m; echo "$(date -Is) start $t" >> $LOG
  DS4_ROCM_D5_PREFETCH=1 "$(dirname "$0")/seq.sh" $t $m 320
  o=/data/ds4-bench/int4-seq/$t
  echo "$(date -Is) end $t $(tr '\n' ' ' < $o/meta.txt) steps=$(grep -c D8_STEP $o/stderr.txt) tokmd5=$(grep -o 'D8_STEP pos=[0-9]* tok=[0-9]*' $o/stderr.txt | md5sum | cut -c1-12) shelf_swap_lines=$(grep -c 'D5 shelf buffer' $o/stderr.txt) reclaims=$(grep -c 'staging reclaimed' $o/stderr.txt)" >> $LOG
done
mx
MOD="DS4_DSPARK_SCHED=model DS4_DSPARK_MODEL_P=98 DS4_DSPARK_MODEL_VA=81 DS4_DSPARK_MODEL_VB=38"
for p in code prose json translate list; do erun $MOD $DSPE int4-mx-$p-dsp-model $(ntok $p) $(args $p) $DSPF; done
erun DS4_ROCM_D5_PREFETCH=1 int4-mx-code-plain-pf 512 $P --nothink; check int4-mx-code-plain-pf $(ref code) || stop digest
erun DS4_ROCM_D5_PREFETCH=1 int4-mx-long-plain-pf 256 $E/p-long.txt; check int4-mx-long-plain-pf $(ref long) || stop digest
erun DS4_ROCM_D5_PREFETCH=1 DS4_DSPARK_SCHED=old $DSPE int4-mx-code-dsp-pf 512 $P --nothink $DSPF
echo "$(date -Is) ROUNDS int4-mx-code-dsp-pf $(field int4-mx-code-dsp-pf dspark_cycles)/$(field int4-mx-code-dsp-pf dspark_accepted) ref 172/340" >> $LOG
echo "$(date -Is) QUEUE 2 DONE" >> $LOG
