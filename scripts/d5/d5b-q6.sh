#!/usr/bin/env bash
# D5b queue 6 (binary /data/ds4-d5 at 188fb61, the branch tip): test, smokes
# and one confirmation run each of the defaults (plain code with logits vs
# d5b-ref-code; DSpark code with logits vs d5b-b-code-off-r1).
set -u; LOG=/data/ds4-bench/d5b-q6.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm" >> $LOG
( cd /data/ds4-d5 && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) > /data/ds4-bench/d5b-q6-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test|step " /data/ds4-bench/d5b-q6-tests.log | tail -3 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5b tip on${NX:+ ($NX)}" run "$t" "$@"; }
C="$P/p-code.txt --nothink"
DSPF="--dspark --mtp-model $DSP"
on d5bt-smoke-8 8 $C
on d5bt-smoke-64 64 $C
DS4_DSPARK_SCHED=old on d5bt-smoke-dsp-8 8 $C $DSPF
DS4_DSPARK_SCHED=old on d5bt-smoke-dsp-64 64 $C $DSPF
NX="plain vs d5b-ref-code" DS4_REF_TAG=d5b-ref-code on d5bt-code-on 512 $C
NX="dsp vs d5b-b-code-off-r1" DS4_REF_TAG=d5b-b-code-off-r1 DS4_DSPARK_SCHED=old on d5bt-dsp-code-on 512 $C $DSPF
echo "$(date -Is) QUEUE DONE" >> $LOG
