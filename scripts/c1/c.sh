#!/usr/bin/env bash
# C1 stage C: adopted bitwise kernels (DS4_ROCM_C1_NORM, DS4_ROCM_C1_HC_PROJ
# default on; DS4_ROCM_C1_Q8_PIPE default off): gates, on/off pairs, profile,
# DSpark and Q2.  OFF = the old kernels.
set -u; . "$(dirname "$0")/common.sh"
P=/data/ds4-env
offrun() { DS4_ROCM_C1_NORM=0 DS4_ROCM_C1_HC_PROJ=0 run "$@"; }
echo "$(date -Is) C wait for B" >> $LOG
until [ "$(grep -c "QUEUE DONE\|STOP" $LOG)" -ge 2 ]; do sleep 30; done
grep -q STOP $LOG && { echo "$(date -Is) C not started (STOP seen)" >> $LOG; exit 1; }
cp -r /data/ds4-c1/.c1-stage/. /data/ds4-c1/ && cd /data/ds4-c1 && git add rocm/ds4_rocm_v41.cuh rocm/ds4_rocm_norm_rope.cuh tools/c1/c1_exact.cpp scripts/c1/c.sh && \
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -qm "c1: bitwise norm preload (DS4_ROCM_C1_NORM) and HC projection step 128 on by default; Q8 load pipelining off (not bitwise); kernel exactness test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || { echo "$(date -Is) STOP commit failed" >> $LOG; exit 1; }
echo "$(date -Is) C start build" >> $LOG
"$(dirname "$0")/build.sh" || { echo "$(date -Is) STOP build failed" >> $LOG; exit 1; }
echo "$(date -Is) build ok $(git -C /data/ds4-c1 rev-parse --short HEAD)" >> $LOG
mx
DS4_ROCM_V41_ENGRAM_E4M3_TRACE=1 run c1c-smoke-8 8 $P/p-code.txt --nothink
run c1c-smoke-64 64 $P/p-code.txt --nothink
run c1c-on-code-r1 512 $P/p-code.txt --nothink
offrun c1c-off-code-r1 512 $P/p-code.txt --nothink
run c1c-on-prose 512 $P/p-prose.txt
run c1c-on-hash 512 $P/p-hash.txt
run c1c-on-long 256 $P/p-long.txt
run c1c-on-code-r2 512 $P/p-code.txt --nothink
offrun c1c-off-code-r2 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1c-prof-events-on 512 $P/p-code.txt --nothink
offrun c1c-off-prose 512 $P/p-prose.txt
export DS4_DSPARK_SCHED=old
run c1c-dsp-on 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
offrun c1c-dsp-off 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
unset DS4_DSPARK_SCHED
q2
run c1c-q2-on-code 512 $P/p-code.txt --nothink
offrun c1c-q2-off-code 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE C" >> $LOG
