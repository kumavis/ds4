#!/usr/bin/env bash
# D15 part B: DS4_ROCM_C1_Q8_PIPE=1 (load-pipelined Q8 matvec, fp-order) on the
# d12 build (bdf9396 binary). Smoke on, then off/on pairs (two rounds,
# interleaved), divergence (teacher-forced and free-run) and the C1 profiler.
set -u
LOG=/data/ds4-bench/d15-queue.log
. /data/ds4-d12/scripts/d12/common.sh
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
ON="DS4_ROCM_C1_Q8_PIPE=1 DS4_OUTPUT_CLASS=fp-order"
note() { echo "$(date -Is) $*" >> $LOG; }
dig() { note "DIGEST $1 $(digest $1)"; }
note "D15 start binary $(git -C $DS4_DIR rev-parse --short HEAD) (built from bdf9396)"
P=$E/p-code.txt
# smoke, switch on
erun $ON DS4_NOTE="d15 smoke Q8 pipe on" d15-q8-smoke-on-8 8 $P --nothink;   dig d15-q8-smoke-on-8
erun $ON DS4_NOTE="d15 smoke Q8 pipe on" d15-q8-smoke-on-64 64 $P --nothink; dig d15-q8-smoke-on-64
# pairs: off then on, two rounds
pair() { # name refdigest args...
  local n=$1 want=$2; shift 2
  erun DS4_NOTE="d15 Q8 pipe off $n" d15-q8-$n-off-r$r "$@"
  [ "$want" = - ] && dig d15-q8-$n-off-r$r || gate d15-q8-$n-off-r$r $want
  erun $ON DS4_NOTE="d15 Q8 pipe on $n" d15-q8-$n-on-r$r "$@"; dig d15-q8-$n-on-r$r; }
for r in 1 2; do
  pair mx-code 8290541d324b/511 512 $P --nothink
  pair mx-prose 88ed0e11adfd/511 512 $E/p-prose.txt
  pair mx-hash 4c37f5fee667/410 512 $E/p-hash.txt
  pair mx-long c27a3b087fb5/255 256 $E/p-long.txt
  ( export $TR; pair dsp-code - 512 $P --nothink --dspark --mtp-model $DSP ) || exit 1
  ( export $TR; pair dsp-json - 512 $E/p-json.txt --dspark --mtp-model $DSP ) || exit 1
  ( export DS4_MODEL=$Q2; unset DS4_ROCM_STREAM_REPLICA; pair q2-code f0a2cd53a247/511 512 $P --nothink ) || exit 1
done
# divergence: reference = off with logits; on teacher-forced along it; on free-run
for p in code prose; do
  case $p in code) A="$P --nothink";; *) A="$E/p-prose.txt";; esac
  erun DS4_D0_LOGITS=auto DS4_NOTE="d15 reference (Q8 pipe off) $p, logits" d15-q8-ref-$p 512 $A
  erun $ON DS4_REF_TAG=d15-q8-ref-$p DS4_REF_FORCE=1 DS4_NOTE="d15 Q8 pipe on $p, teacher-forced" d15-q8-tf-$p 512 $A
  erun $ON DS4_REF_TAG=d15-q8-ref-$p DS4_NOTE="d15 Q8 pipe on $p, free-run vs reference" d15-q8-fr-$p 512 $A
done
# C1 profiler, coarse, one each way
erun DS4_ROCM_C1_PROF=coarse DS4_NOTE="d15 C1 profile, Q8 pipe off" d15-q8-prof-off 128 $P --nothink
erun $ON DS4_ROCM_C1_PROF=coarse DS4_NOTE="d15 C1 profile, Q8 pipe on" d15-q8-prof-on 128 $P --nothink
note "QUEUE DONE"
