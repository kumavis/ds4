# D4 queue helpers (sourced). Sequential runs through bench.sh (which takes
# the GPU lock per run). Logs "start TAG" / "end TAG ..." lines to $LOG.
# Stops the queue on a failed run, on MemAvailable below the floor, or on the
# data SSD reaching 89 C. Memory rule: before a run, predict its MemAvailable
# minimum from the last finished run of the same kind and refuse it when the
# prediction is below FLOOR_MB.
B=/data/ds4-env/bench.sh
E=/data/ds4-env
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=/data/ds4-d4 DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1
LOG=${LOG:-/data/ds4-bench/d4-queue.log}
FLOOR_MB=${FLOOR_MB:-12288}   # 12 GiB: never queue a run predicted below this
JITTER_MB=${JITTER_MB:-600}   # allowance for run-to-run MemAvailable noise
SLOT_MB=17.94                 # one MXFP4 expert slot (13.45 GiB / 768)
mx() { export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA; }
q2() { export DS4_MODEL=$Q2; unset DS4_ROCM_STREAM_REPLICA; }
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
rundir() { ls -dt /data/ds4-bench/runs/*-"$1" 2>/dev/null | head -1; }
# resident max slots and budget of a finished run (stream cache stats line)
slots() { grep -o "resident current=[0-9]*/[0-9.]* GiB max=[0-9]*/[0-9.]* GiB budget=[0-9]*" "$(rundir $1)/run.log" | tail -1 \
  | sed 's/.*max=\([0-9]*\)\/.*budget=\([0-9]*\)/\1 \2/'; }
stop() { echo "$(date -Is) STOP: $*" >> $LOG; echo "$(date -Is) QUEUE DONE (stopped)" >> $LOG; exit 1; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(rundir "$tag")
  ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  av=$(field "$tag" avail_min_mb); st=$(field "$tag" ssd_temp_max_c); tps=$(field "$tag" decode_tps)
  echo "$(date -Is) end $tag rc=$rc $ex decode_tps=${tps:-?} avail_min_mb=${av:-?} ssd_max_c=${st:-?} slots(max,budget)=$(slots $tag | tr ' ' ,) dir=$d" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || stop "after $tag (exit)"
  [ -n "$av" ] || stop "after $tag (no avail_min_mb)"
  [ "${av%.*}" -lt "$FLOOR_MB" ] && stop "after $tag (MemAvailable ${av} MiB below ${FLOOR_MB})"
  [ -n "$st" ] && [ "${st%.*}" -ge 89 ] && stop "after $tag (ssd temperature)"
  grep -h "ssd_c=" "$d/sys.log" | grep -o "ssd_c=[0-9]*" | cut -d= -f2 | sort -n | tail -1 | \
    { read -r m; [ -n "$m" ] && [ "$m" -ge 89 ] && stop "after $tag (sys.log ssd_c $m)"; true; }
  return 0; }
# predict MemAvailable minimum (MiB) at N dynamic slots from a finished
# reference run REF_TAG that held REF_N slots
predict() { local ref=$1 refn=$2 n=$3 av; av=$(field "$ref" avail_min_mb)
  awk -v a="$av" -v r="$refn" -v n="$n" -v s=$SLOT_MB -v j=$JITTER_MB 'BEGIN{printf "%d", a-(n-r)*s-j}'; }
# guard REF_TAG REF_N N: refuse (return 1, logged) when the prediction is below the floor
guard() { local p; p=$(predict "$1" "$2" "$3")
  [ -n "$p" ] || { echo "$(date -Is) SKIP N=$3: no reference $1" >> $LOG; return 1; }
  if [ "$p" -lt "$FLOOR_MB" ]; then echo "$(date -Is) SKIP N=$3: predicted avail_min ${p} MiB < ${FLOOR_MB} (ref $1 @ $2)" >> $LOG; return 1; fi
  echo "$(date -Is) guard N=$3: predicted avail_min ${p} MiB (ref $1 @ $2, jitter $JITTER_MB) ok" >> $LOG; return 0; }
# largest N (multiple of 10) with prediction >= floor
maxn() { local av; av=$(field "$1" avail_min_mb)
  awk -v a="$av" -v r="$2" -v s=$SLOT_MB -v j=$JITTER_MB -v f=$FLOOR_MB 'BEGIN{n=r+(a-j-f)/s; printf "%d", int(n/10)*10}'; }
digest() { d=$(rundir "$1")
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  echo "$(printf '%s\n' "$l" | md5sum | cut -c1-12) $(printf '%s\n' "$l" | grep -c .)"; }
