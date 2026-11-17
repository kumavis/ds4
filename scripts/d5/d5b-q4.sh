#!/usr/bin/env bash
# D5b queue 4 (binary: /data/ds4-d5x at 80b832f: stale-entry fix + busy cap).
# Test, smokes, DSpark (old scheduler) on/off on code/json/translate with
# logits compared against queue 1's off-r1 runs, a budget sweep, and plain
# gates with everything on.
set -u; LOG=/data/ds4-bench/d5b-q4.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm" >> $LOG
( cd /data/ds4-d5x && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) > /data/ds4-bench/d5b-q4-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test|step " /data/ds4-bench/d5b-q4-tests.log | tail -3 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5b-f on${NX:+ ($NX)}" run "$t" "$@"; }
off() { t=$1; shift; DS4_NOTE="D5b-f off${NX:+ ($NX)}" run "$t" "$@"; }
DSPF="--dspark --mtp-model $DSP"
C="$P/p-code.txt --nothink"
on d5bf-smoke-8 8 $C
on d5bf-smoke-64 64 $C
NX=debug DS4_ROCM_D5_DEBUG=1 DS4_DSPARK_SCHED=old on d5bf-smoke-dsp-json-8 8 $P/p-json.txt $DSPF
NX=debug DS4_ROCM_D5_DEBUG=1 DS4_DSPARK_SCHED=old on d5bf-smoke-dsp-json-64 64 $P/p-json.txt $DSPF
export DS4_DSPARK_SCHED=old
for r in 1 2; do
  for p in code json translate; do
    a="$P/p-$p.txt"; [ $p = code ] && a="$a --nothink"
    [ $r = 1 ] && NX="dsp control" off d5bf-b-$p-off 512 $a $DSPF
    NX="dsp vs d5b-b-$p-off-r1" DS4_REF_TAG=d5b-b-$p-off-r1 on d5bf-b-$p-on-r$r 512 $a $DSPF
  done
done
NX="verify K 2" DS4_ROCM_D5_VERIFY_K=2 on d5bf-b-code-k2 512 $C $DSPF
NX="verify K 8" DS4_ROCM_D5_VERIFY_K=8 on d5bf-b-code-k8 512 $C $DSPF
NX="verify K 8" DS4_ROCM_D5_VERIFY_K=8 on d5bf-b-json-k8 512 $P/p-json.txt $DSPF
NX="verify L0 off" DS4_ROCM_D5_VERIFY_L0=0 on d5bf-b-code-l0off 512 $C $DSPF
NX="verify off (K 0)" DS4_ROCM_D5_VERIFY_K=0 on d5bf-b-code-k0 512 $C $DSPF
unset DS4_DSPARK_SCHED
NX="plain vs d5b-ref-code" DS4_REF_TAG=d5b-ref-code on d5bf-g-code-on 512 $C
off d5bf-g-code-off 512 $C
on d5bf-g-prose-on 512 $P/p-prose.txt
on d5bf-g-hash-on 512 $P/p-hash.txt
on d5bf-g-long-on 256 $P/p-long.txt
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
on d5bf-g-q2-code-on 512 $C
echo "$(date -Is) QUEUE DONE" >> $LOG
