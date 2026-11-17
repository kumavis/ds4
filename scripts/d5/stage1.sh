#!/usr/bin/env bash
# D5 stage 1: smoke the trace build (-n 8, -n 64), one untraced gate run, then
# traces (selected ids + miss masks + early-router probes) on code/prose/long/hash.
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
P=/data/ds4-env
DS4_NOTE="D5 trace smoke" DS4_ROCM_D5_TRACE=$TR/smoke-8.trace DS4_ROCM_D5_PROBE=1 run d5-trace-smoke-8 8 $P/p-code.txt --nothink
DS4_NOTE="D5 trace smoke" DS4_ROCM_D5_TRACE=$TR/smoke-64.trace DS4_ROCM_D5_PROBE=1 run d5-trace-smoke-64 64 $P/p-code.txt --nothink
DS4_NOTE="D5 build, trace off (gate + zero-cost check)" run d5-off-code 512 $P/p-code.txt --nothink
DS4_NOTE="D5 trace+probe" DS4_ROCM_D5_TRACE=$TR/code.trace DS4_ROCM_D5_PROBE=1 run d5-trace-code 512 $P/p-code.txt --nothink
DS4_NOTE="D5 trace+probe" DS4_ROCM_D5_TRACE=$TR/prose.trace DS4_ROCM_D5_PROBE=1 run d5-trace-prose 512 $P/p-prose.txt
DS4_NOTE="D5 trace+probe" DS4_ROCM_D5_TRACE=$TR/long.trace DS4_ROCM_D5_PROBE=1 run d5-trace-long 256 $P/p-long.txt
DS4_NOTE="D5 trace+probe" DS4_ROCM_D5_TRACE=$TR/hash.trace DS4_ROCM_D5_PROBE=1 run d5-trace-hash 512 $P/p-hash.txt
DS4_NOTE="D5 trace only (no probe)" DS4_ROCM_D5_TRACE=$TR/code-noprobe.trace run d5-trace-code-noprobe 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
