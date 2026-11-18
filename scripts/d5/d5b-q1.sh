#!/usr/bin/env bash
# D5b queue 1 (binary: /data/ds4-d5 at the D5b commit): model-free test,
# smokes, Part A interference experiments (plain code, one run each) and
# Part B verify-round prefetch (DSpark, DS4_DSPARK_SCHED=old) on code/json/
# translate, two off and two on runs each with logits compared against the
# first off run, an exact-verify check and a budget sweep on code.
set -u; LOG=/data/ds4-bench/d5b-q1.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm" >> $LOG
( cd /data/ds4-d5 && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) >> /data/ds4-bench/d5b-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test|step " /data/ds4-bench/d5b-tests.log | tail -3 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5b on${NX:+ ($NX)}" run "$t" "$@"; }
off() { t=$1; shift; DS4_NOTE="D5b off${NX:+ ($NX)}" run "$t" "$@"; }
DSPF="--dspark --mtp-model $DSP"
on d5b-smoke-8 8 $P/p-code.txt --nothink
on d5b-smoke-64 64 $P/p-code.txt --nothink
DS4_DSPARK_SCHED=old on d5b-smoke-dsp-8 8 $P/p-code.txt --nothink $DSPF
DS4_DSPARK_SCHED=old on d5b-smoke-dsp-64 64 $P/p-code.txt --nothink $DSPF
C="$P/p-code.txt --nothink"
# Part A (plain code, 512)
off d5b-a-off 512 $C
NX=noadopt DS4_ROCM_D5_DIAG=noadopt on d5b-a-noadopt 512 $C
NX="noadopt drive0" DS4_ROCM_D5_DIAG=noadopt DS4_ROCM_D5_DRIVE=0 on d5b-a-noadopt-drive0 512 $C
NX="noadopt drive1" DS4_ROCM_D5_DIAG=noadopt DS4_ROCM_D5_DRIVE=1 on d5b-a-noadopt-drive1 512 $C
NX=noupload DS4_ROCM_D5_DIAG=noupload on d5b-a-noupload 512 $C
NX=noread DS4_ROCM_D5_DIAG=noread on d5b-a-noread 512 $C
NX="noadopt strict" DS4_ROCM_D5_DIAG=noadopt DS4_ROCM_D5_STRICT=1 on d5b-a-noadopt-strict 512 $C
NX="noadopt earlydrop0" DS4_ROCM_D5_DIAG=noadopt DS4_ROCM_D5_EARLYDROP=0 on d5b-a-noadopt-ed0 512 $C
NX=probe DS4_ROCM_D5_DIAG=probe on d5b-a-probe 512 $C
on d5b-a-on 512 $C
NX=earlydrop0 DS4_ROCM_D5_EARLYDROP=0 on d5b-a-on-ed0 512 $C
NX=drive0 DS4_ROCM_D5_DRIVE=0 on d5b-a-on-drive0 512 $C
NX=strict DS4_ROCM_D5_STRICT=1 on d5b-a-on-strict 512 $C
NX=chunk256 DS4_ROCM_D5_CHUNK_KB=256 on d5b-a-on-chunk256 512 $C
NX="workers 18" DS4_ROCM_STREAM_READ_WORKERS=18 on d5b-a-on-w18 512 $C
NX="workers 18" DS4_ROCM_STREAM_READ_WORKERS=18 off d5b-a-off-w18 512 $C
# plain reference with logits, and logits checks of the prefetch
NX="plain ref, logits" DS4_D0_LOGITS=auto off d5b-ref-code 512 $C
NX="plain, logits vs ref" DS4_REF_TAG=d5b-ref-code on d5b-ref-code-on 512 $C
NX="dspark exact verify vs plain ref" DS4_REF_TAG=d5b-ref-code DS4_DSPARK_SCHED=old on d5b-dsp-exact-code-on 512 $C $DSPF --dspark-verify exact
# Part B: DSpark verify-round prefetch, old scheduler
export DS4_DSPARK_SCHED=old
for p in code json translate; do
  a="$P/p-$p.txt"; [ $p = code ] && a="$a --nothink"
  NX="dsp ref, logits" DS4_D0_LOGITS=auto off d5b-b-$p-off-r1 512 $a $DSPF
  NX="dsp vs off-r1" DS4_REF_TAG=d5b-b-$p-off-r1 on d5b-b-$p-on-r1 512 $a $DSPF
  NX="dsp vs off-r1" DS4_REF_TAG=d5b-b-$p-off-r1 off d5b-b-$p-off-r2 512 $a $DSPF
  NX="dsp vs off-r1" DS4_REF_TAG=d5b-b-$p-off-r1 on d5b-b-$p-on-r2 512 $a $DSPF
done
NX="verify K 2" DS4_ROCM_D5_VERIFY_K=2 on d5b-b-code-k2 512 $C $DSPF
NX="verify K 8" DS4_ROCM_D5_VERIFY_K=8 on d5b-b-code-k8 512 $C $DSPF
NX="verify K 16" DS4_ROCM_D5_VERIFY_K=16 on d5b-b-code-k16 512 $C $DSPF
NX="verify L0 off" DS4_ROCM_D5_VERIFY_L0=0 on d5b-b-code-l0off 512 $C $DSPF
NX="verify off (K 0), single-token on" DS4_ROCM_D5_VERIFY_K=0 on d5b-b-code-k0 512 $C $DSPF
echo "$(date -Is) QUEUE DONE" >> $LOG
