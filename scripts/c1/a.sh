#!/usr/bin/env bash
# C1 stage A: build, smoke, profile modes on MXFP4 code (+ Q2), zero-cost pair.
set -u; . "$(dirname "$0")/common.sh"
P=/data/ds4-env
echo "$(date -Is) A start build" >> $LOG
"$(dirname "$0")/build.sh" || { echo "$(date -Is) STOP build failed" >> $LOG; exit 1; }
echo "$(date -Is) build ok $(git -C /data/ds4-c1 rev-parse --short HEAD)" >> $LOG
mx
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_TOKENS=2 run c1-smoke-8 8 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events run c1-smoke-64 64 $P/p-code.txt --nothink
run c1-off-code-r1 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=coarse DS4_ROCM_C1_PROF_LAYERS=1 run c1-coarse-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1-events-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=sync DS4_ROCM_C1_PROF_LAYERS=1 run c1-sync-code 512 $P/p-code.txt --nothink
run c1-off-code-r2 512 $P/p-code.txt --nothink
q2
run c1-q2-off-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1-q2-events-code 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
