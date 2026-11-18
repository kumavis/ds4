# D12 queue helpers (sourced). Sequential runs through bench.sh (which takes
# the GPU lock per run). Stops on a failed run, MemAvailable below 12 GB or an
# NVMe at 89 C. Based on scripts/d11/common.sh.
B=/data/ds4-env/bench.sh
E=/data/ds4-env
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
T=/data/ds4-bench/d12-traces
mkdir -p $T
export DS4_DIR=${DS4_DIR:-/data/ds4-d12} DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA DS4_DSPARK_SCHED=old
LOG=${LOG:-/data/ds4-bench/d12-queue.log}
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
digest() { d=$(ls -dt /data/ds4-bench/runs/*-"$1" | head -1)
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  echo "$(printf '%s\n' "$l" | md5sum | cut -c1-12)/$(printf '%s\n' "$l" | grep -c .)"; }
batchm() { d=$(ls -dt /data/ds4-bench/runs/*-"$1" | head -1); grep -o "batch calls=[0-9]* unique=[0-9]* hits=[0-9]* misses=[0-9]*" "$d/run.log" | grep -o "misses=[0-9]*"; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" 2>/dev/null | head -1); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  av=$(field "$tag" avail_min_mb); st=$(field "$tag" nvme_temp_max_c)
  echo "$(date -Is) end $tag rc=$rc $ex tps=$(field $tag decode_tps) cycles=$(field $tag dspark_cycles) accepted=$(field $tag dspark_accepted) misses=$(field $tag decode_misses) batch_$(batchm $tag) evictions=$(field $tag evictions) digest=$(digest $tag) avail_min_mb=$av nvme_max_c=$st" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP after $tag (exit); QUEUE DONE" >> $LOG; exit 1; }
  [ -n "$av" ] && [ "${av%.*}" -lt 12000 ] && { echo "$(date -Is) STOP after $tag (memory); QUEUE DONE" >> $LOG; exit 1; }
  [ -n "$st" ] && [ "${st%.*}" -ge 89 ] && { echo "$(date -Is) STOP after $tag (ssd temperature); QUEUE DONE" >> $LOG; exit 1; }
  return 0; }
erun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  ( export "${e[@]}"; run "$@" ) || exit 1; }
# trace run: erun with DS4_D12_CACHE_TRACE=$T/<tag>.trace
trun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  local tag=$1; erun "${e[@]}" DS4_D12_CACHE_TRACE=$T/$tag.trace "$@"; gzip -f $T/$tag.trace; }
# gate: stop the queue unless the tag's token-ID digest equals $2
gate() { local g; g=$(digest "$1"); echo "$(date -Is) DIGEST $1 $g (want $2)" >> $LOG
  [ "$g" = "$2" ] || { echo "$(date -Is) STOP: digest mismatch on $1; QUEUE DONE" >> $LOG; exit 1; }; }
# DSpark gate: rounds/accepted must equal $2/$3
dgate() { local c a; c=$(field "$1" dspark_cycles); a=$(field "$1" dspark_accepted)
  echo "$(date -Is) DSPARK $1 $c/$a (want $2/$3)" >> $LOG
  [ "$c/$a" = "$2/$3" ] || { echo "$(date -Is) STOP: DSpark mismatch on $1; QUEUE DONE" >> $LOG; exit 1; }; }
args() { case $1 in code) echo "$E/p-code.txt --nothink";; *) echo "$E/p-$1.txt";; esac; }
TR="DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1"
