#!/usr/bin/env bash
# D11 Part A: one-drive runs on the int2 binary (same commands as the
# int2-mx-code-plain/dsp records, without DS4_ROCM_STREAM_REPLICA).
set -u
B=/data/ds4-env/bench.sh; E=/data/ds4-env
LOG=/data/ds4-bench/d11-queue.log
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
export DS4_DIR=/data/ds4-int2 DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1
export DS4_MODEL=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
unset DS4_ROCM_STREAM_REPLICA
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" | head -1); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" | tail -1)
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  dg="$(printf '%s\n' "$l" | md5sum | cut -c1-12)/$(printf '%s\n' "$l" | grep -c .)"
  echo "$(date -Is) end $tag rc=$rc $ex tps=$(field $tag decode_tps) cycles=$(field $tag dspark_cycles) accepted=$(field $tag dspark_accepted) misses=$(field $tag decode_misses) nvme_max_c=$(field $tag nvme_temp_max_c) avail_min_mb=$(field $tag avail_min_mb) digest=$dg" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP; QUEUE DONE" >> $LOG; exit 1; }; }
echo "$(date -Is) D11 QA start int2 $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
P="$E/p-code.txt --nothink"
DS4_NOTE="d11 one drive plain code smoke" run d11-1d-smoke-plain-8 8 $P
for r in 1 2; do
  DS4_NOTE="d11 one drive plain code" run d11-1d-plain-r$r 512 $P
  DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1 DS4_NOTE="d11 one drive DSpark old code" \
    run d11-1d-dsp-r$r 512 $P --dspark --mtp-model $DSP
done
echo "$(date -Is) QUEUE DONE" >> $LOG
