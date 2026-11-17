# D5 queue helpers (sourced). Sequential runs through bench.sh; stop on the
# first failed run, MemAvailable < 8 GB or data SSD >= 89 C.  Each run logs its
# token-ID digest (perf token ... id N lines, as in the brief).
B=/data/ds4-env/bench.sh
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=/data/ds4-d5 DS4_TOKEN_TIMING=1
TR=/data/ds4-bench/d5
LOG=${LOG:-/data/ds4-bench/d5-queue.log}
digest() { grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$1" | awk "{print \$3, \$NF}" | md5sum | cut -c1-12; }
lines() { grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$1" | wc -l; }
run() { tag=$1; shift
  av0=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
  [ "$av0" -lt 8000 ] && { echo "$(date -Is) STOP before $tag (MemAvailable ${av0} MB)" >> $LOG; exit 1; }
  echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@"; rc=$?
  d=$(ls -dt /data/ds4-bench/runs/*-"$tag" | head -1); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" | tail -1)
  rec=$(grep -F "\"tag\":\"$tag\"" /data/ds4-bench/results.jsonl | tail -1)
  av=$(echo "$rec" | grep -o "\"avail_min_mb\": *[0-9]*" | grep -o "[0-9]*$")
  tc=$(echo "$rec" | grep -o "\"ssd_temp_max_c\": *[0-9]*" | grep -o "[0-9]*$")
  tps=$(echo "$rec" | grep -o "\"decode_tps\": *[0-9.]*" | grep -o "[0-9.]*$")
  echo "$(date -Is) end $tag rc=$rc $ex tps=${tps:-?} digest=$(digest $d/run.log)/$(lines $d/run.log) avail_min_mb=${av:-?} ssd_max=${tc:-?} dir=$d" >> $LOG
  [ "$ex" = "EXIT=0" ] || { echo "$(date -Is) STOP after $tag (exit)" >> $LOG; exit 1; }
  [ -n "$av" ] && [ "$av" -lt 8000 ] && { echo "$(date -Is) STOP after $tag (memory)" >> $LOG; exit 1; }
  [ -n "$tc" ] && [ "$tc" -ge 89 ] && { echo "$(date -Is) STOP after $tag (ssd temp)" >> $LOG; exit 1; }
  return 0; }
