# C1 queue helpers (from scripts/int) (sourced). Sequential runs through
# bench.sh (which takes the GPU lock per run). Logs "start TAG" / "end TAG rc"
# lines; stops the queue on a failed run, MemAvailable below 8 GB, or the data
# SSD reaching 89 C.
B=/data/ds4-env/bench.sh
E=/data/ds4-env
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=/data/ds4-c1 DS4_TOKEN_TIMING=1
LOG=${LOG:-/data/ds4-bench/c1-queue.log}
mx() { export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA; }
q2() { export DS4_MODEL=$Q2; unset DS4_ROCM_STREAM_REPLICA; }
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" 2>/dev/null | head -1)
  ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  av=$(field "$tag" avail_min_mb); st=$(field "$tag" ssd_temp_max_c); tps=$(field "$tag" decode_tps)
  echo "$(date -Is) end $tag rc=$rc $ex decode_tps=${tps:-?} avail_min_mb=${av:-?} ssd_max_c=${st:-?} dir=$d" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP after $tag (exit)" >> $LOG; exit 1; }
  [ -n "$av" ] && [ "${av%.*}" -lt 12000 ] && { echo "$(date -Is) STOP after $tag (memory)" >> $LOG; exit 1; }
  [ -n "$st" ] && [ "${st%.*}" -ge 89 ] && { echo "$(date -Is) STOP after $tag (ssd temperature)" >> $LOG; exit 1; }
  return 0; }
# Token-ID digest of a run (the gate): md5 of "N ID" over the perf token lines.
digest() { d=$(ls -dt /data/ds4-bench/runs/*-"$1" | head -1)
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  echo "$(printf '%s\n' "$l" | md5sum | cut -c1-12) $(printf '%s\n' "$l" | grep -c .)"; }
