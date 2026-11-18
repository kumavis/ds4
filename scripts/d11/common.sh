# D11 queue helpers (sourced). Sequential runs through bench.sh (which takes
# the GPU lock per run). Stops on a failed run, MemAvailable below 12 GB or an
# NVMe at 89 C.
B=/data/ds4-env/bench.sh
E=/data/ds4-env
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=${DS4_DIR:-/data/ds4-d11} DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1
LOG=${LOG:-/data/ds4-bench/d11-queue-b.log}
mx() { export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA; }
q2() { export DS4_MODEL=$Q2; unset DS4_ROCM_STREAM_REPLICA; }
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
digest() { d=$(ls -dt /data/ds4-bench/runs/*-"$1" | head -1)
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  echo "$(printf '%s\n' "$l" | md5sum | cut -c1-12)/$(printf '%s\n' "$l" | grep -c .)"; }
batchm() { d=$(ls -dt /data/ds4-bench/runs/*-"$1" | head -1); grep -o "batch calls=[0-9]* unique=[0-9]* hits=[0-9]* misses=[0-9]*" "$d/run.log" | grep -o "misses=[0-9]*"; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" 2>/dev/null | head -1); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  av=$(field "$tag" avail_min_mb); st=$(field "$tag" nvme_temp_max_c)
  echo "$(date -Is) end $tag rc=$rc $ex tps=$(field $tag decode_tps) cycles=$(field $tag dspark_cycles) accepted=$(field $tag dspark_accepted) misses=$(field $tag decode_misses) batch_$(batchm $tag) digest=$(digest $tag) avail_min_mb=$av nvme_max_c=$st" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP after $tag (exit); QUEUE DONE" >> $LOG; exit 1; }
  [ -n "$av" ] && [ "${av%.*}" -lt 12000 ] && { echo "$(date -Is) STOP after $tag (memory); QUEUE DONE" >> $LOG; exit 1; }
  [ -n "$st" ] && [ "${st%.*}" -ge 89 ] && { echo "$(date -Is) STOP after $tag (ssd temperature); QUEUE DONE" >> $LOG; exit 1; }
  return 0; }
erun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  ( export "${e[@]}"; run "$@" ) || exit 1; }
args() { case $1 in code) echo "$E/p-code.txt --nothink";; *) echo "$E/p-$1.txt";; esac; }
TR="DS4_D9_ROUND_TRACE=1 DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1"
