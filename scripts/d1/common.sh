# D1 queue helpers (sourced). Sequential runs through bench.sh; stops on the
# first failed run or when MemAvailable dipped below 10 GB.
B=/data/ds4-env/bench.sh
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=/data/ds4-d1 DS4_TOKEN_TIMING=1
LOG=${LOG:-/data/ds4-bench/d1-queue.log}
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@"; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" | head -1); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" | tail -1)
  av=$(grep -F "\"tag\":\"$tag\"" /data/ds4-bench/results.jsonl | tail -1 | grep -o '"avail_min_mb": *[0-9]*' | grep -o '[0-9]*$')
  echo "$(date -Is) end $tag rc=$rc $ex avail_min_mb=${av:-?} dir=$d" >> $LOG
  [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP after $tag (exit)" >> $LOG; exit 1; }
  [ -n "$av" ] && [ "$av" -lt 10000 ] && { echo "$(date -Is) STOP after $tag (memory)" >> $LOG; exit 1; }
  return 0; }
