#!/usr/bin/env bash
# D5 diagnostics on the d5-prefetch-x build (buffer-swap adoption, DS4_ROCM_D5_DIAG):
# model-free test, then code runs: off, probe only, reads without adoption,
# prefetch on (swap), prefetch on with 16 speculative chunks in flight.
set -u; LOG=/data/ds4-bench/d5x-queue.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm (d5x)" >> $LOG
( cd /data/ds4-d5x && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) >> /data/ds4-bench/d5x-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test" /data/ds4-bench/d5x-tests.log | tail -2 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
DS4_NOTE="D5x smoke swap" DS4_ROCM_D5_PREFETCH=1 run d5x-smoke-8 8 $P/p-code.txt --nothink
DS4_NOTE="D5x smoke swap" DS4_ROCM_D5_PREFETCH=1 run d5x-smoke-64 64 $P/p-code.txt --nothink
DS4_NOTE="D5x off" run d5x-code-off 512 $P/p-code.txt --nothink
DS4_NOTE="D5x diag probe only" DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_DIAG=probe run d5x-code-diag-probe 512 $P/p-code.txt --nothink
DS4_NOTE="D5x diag reads, no adoption" DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_DIAG=noadopt run d5x-code-diag-noadopt 512 $P/p-code.txt --nothink
DS4_NOTE="D5x on (swap)" DS4_ROCM_D5_PREFETCH=1 run d5x-code-on 512 $P/p-code.txt --nothink
DS4_NOTE="D5x on (swap), inflight 16" DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_D5_INFLIGHT=16 run d5x-code-on-inflight16 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
