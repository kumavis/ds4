#!/usr/bin/env bash
# C1 stage B: rebuild (decode-only profile, finer marks, HC projection and Q8
# matvec load pipelining), gates with the new kernels (default) and with the
# old path, clean profiles, speed pairs, Q2 and DSpark.
set -u; . "$(dirname "$0")/common.sh"
P=/data/ds4-env
echo "$(date -Is) B start build" >> $LOG
"$(dirname "$0")/build.sh" || { echo "$(date -Is) STOP build failed" >> $LOG; exit 1; }
echo "$(date -Is) build ok $(git -C /data/ds4-c1 rev-parse --short HEAD)" >> $LOG
mx
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_TOKENS=2 run c1b-smoke-8 8 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events run c1b-smoke-64 64 $P/p-code.txt --nothink
run c1b-on-code-r1 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 run c1b-off-code-r1 512 $P/p-code.txt --nothink
run c1b-on-prose 512 $P/p-prose.txt
run c1b-on-hash 512 $P/p-hash.txt
run c1b-on-long 256 $P/p-long.txt
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1b-prof-events-off 512 $P/p-code.txt --nothink
DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1b-prof-events-on 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 DS4_ROCM_C1_PROF=sync DS4_ROCM_C1_PROF_LAYERS=1 run c1b-prof-sync-off 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 DS4_ROCM_C1_PROF=coarse DS4_ROCM_C1_PROF_LAYERS=1 run c1b-prof-coarse-off 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 run c1b-off-code-r2 512 $P/p-code.txt --nothink
run c1b-on-code-r2 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 run c1b-q8only-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_Q8_PIPE=0 run c1b-hconly-code 512 $P/p-code.txt --nothink
export DS4_DSPARK_SCHED=old
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 run c1b-dsp-off 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
run c1b-dsp-on 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 DS4_ROCM_DSPARK_V41_VERIFY_PROFILE=100000 DS4_D8_STEP_TRACE=1 DS4_ROCM_DSPARK_PHASE_TRACE=1 run c1b-dsp-prof-off 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
unset DS4_DSPARK_SCHED
q2
run c1b-q2-on-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 run c1b-q2-off-code 512 $P/p-code.txt --nothink
DS4_ROCM_C1_HC_PROJ=0 DS4_ROCM_C1_Q8_PIPE=0 DS4_ROCM_C1_PROF=events DS4_ROCM_C1_PROF_LAYERS=1 run c1b-q2-prof-events-off 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
