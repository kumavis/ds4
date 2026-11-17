#!/usr/bin/env bash
# D5b queue 2x (binary: /data/ds4-d5x, 4b91934 + DS4_ROCM_D5_INFLIGHT_BUSY):
# test, smokes, then plain code with the speculative cap while a demand set
# is active at 2 / 4 / 8, and an off control.
set -u; LOG=/data/ds4-bench/d5b-q2x.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm" >> $LOG
( cd /data/ds4-d5x && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) > /data/ds4-bench/d5b-q2x-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test|step " /data/ds4-bench/d5b-q2x-tests.log | tail -3 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5b-x on${NX:+ ($NX)}" run "$t" "$@"; }
off() { t=$1; shift; DS4_NOTE="D5b-x off${NX:+ ($NX)}" run "$t" "$@"; }
C="$P/p-code.txt --nothink"
NX="busy 2" DS4_ROCM_D5_INFLIGHT_BUSY=2 on d5bx-smoke-8 8 $C
NX="busy 2" DS4_ROCM_D5_INFLIGHT_BUSY=2 on d5bx-smoke-64 64 $C
off d5bx-a-off 512 $C
NX="busy 2" DS4_ROCM_D5_INFLIGHT_BUSY=2 on d5bx-a-busy2 512 $C
NX="busy 4" DS4_ROCM_D5_INFLIGHT_BUSY=4 on d5bx-a-busy4 512 $C
NX="busy 8" DS4_ROCM_D5_INFLIGHT_BUSY=8 on d5bx-a-busy8 512 $C
NX="busy 2, noadopt" DS4_ROCM_D5_INFLIGHT_BUSY=2 DS4_ROCM_D5_DIAG=noadopt on d5bx-a-busy2-noadopt 512 $C
echo "$(date -Is) QUEUE DONE" >> $LOG
