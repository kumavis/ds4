# D20 queue helpers (sourced; from scripts/d16).
# Sequential runs through bench.sh (which takes the GPU lock per run). Logs
# "start TAG" / "end TAG ..." lines to $LOG. Stops the queue on a failed run,
# on a missing or too low MemAvailable minimum (FLOOR_MB, 12 GiB), or on an
# NVMe reaching 89 C. Every run: DS4_TOKEN_TIMING=1, DS4_D8_STEP_TRACE=1 (as
# kumanix-int and D4 measured).
B=/data/ds4-env/bench.sh
E=/data/ds4-env
Q2=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
MX=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
DSP=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_DIR=/data/ds4-d20 DS4_TOKEN_TIMING=1 DS4_D8_STEP_TRACE=1
LOG=${LOG:-/data/ds4-bench/d20-queue.log}
FLOOR_MB=${FLOOR_MB:-12288}
mx() { export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA; }
q2() { export DS4_MODEL=$Q2; unset DS4_ROCM_STREAM_REPLICA; }
rec() { grep -F "\"tag\":\"$1\"" /data/ds4-bench/results.jsonl | tail -1; }
field() { rec "$1" | grep -o "\"$2\": *[0-9.-]*" | grep -o '[0-9.-]*$'; }
rundir() { ls -dt /data/ds4-bench/runs/*-"$1" 2>/dev/null | head -1; }
slots() { local d; d=$(rundir $1)/run.log
  { grep -o "dynamic slots [0-9]*" "$d" 2>/dev/null || grep -o "dynamic [0-9]* -> [0-9]* slots\|+ [0-9.]* GiB dynamic ([0-9]* experts)" "$d" 2>/dev/null; } | tail -1 | grep -o "[0-9]*" | tail -1; }
digest() { d=$(rundir "$1")
  l=$(grep -o "perf token [0-9]* pos [0-9]* .* id [0-9]*" "$d/run.log" | awk '{print $3, $NF}')
  echo "$(printf '%s\n' "$l" | md5sum | cut -c1-12) $(printf '%s\n' "$l" | grep -c .)"; }
stop() { echo "$(date -Is) STOP: $*" >> $LOG; echo "$(date -Is) QUEUE DONE (stopped)" >> $LOG; exit 1; }
run() { tag=$1; shift; echo "$(date -Is) start $tag" >> $LOG; "$B" "$tag" "$@" > /dev/null 2>&1; rc=$?
  d=$(rundir "$tag")
  ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  av=$(field "$tag" avail_min_mb); st=$(field "$tag" ssd_temp_max_c); nt=$(field "$tag" nvme_temp_max_c)
  tps=$(field "$tag" decode_tps)
  pf=$(grep -o "D16 route prior decode: .*swapped_row_layers=[0-9]* ([0-9.]*)" "$d/run.log" 2>/dev/null | tail -1 | grep -o "swapped_row_layers=.*" | tr -d " ")
  pv=$(grep -o "D16 route prior verify (committed rows): .*swapped_row_layers=[0-9]* ([0-9.]*)" "$d/run.log" 2>/dev/null | tail -1 | grep -o "swapped_row_layers=.*" | tr -d " ")
  pf="${pf:+dec_$pf }${pv:+ver_$pv}"
  echo "$(date -Is) end $tag rc=$rc $ex decode_tps=${tps:-?} avail_min_mb=${av:-?} ssd_max_c=${st:-?} nvme_max_c=${nt:-?} dyn_slots=$(slots $tag) digest=$(digest $tag | tr ' ' /) ${pf:-} dir=$d" >> $LOG
  [ "$rc" = 0 ] && [ "$ex" = "EXIT=0" ] || stop "after $tag (exit)"
  [ -n "$av" ] || stop "after $tag (no avail_min_mb)"
  [ "${av%.*}" -lt "$FLOOR_MB" ] && stop "after $tag (MemAvailable ${av} MiB below ${FLOOR_MB})"
  [ -n "$nt" ] && [ "${nt%.*}" -ge 89 ] && stop "after $tag (nvme temperature)"
  [ -n "$st" ] && [ "${st%.*}" -ge 89 ] && stop "after $tag (ssd temperature)"
  return 0; }
# check TAG DIGEST LINES: logs PASS/FAIL against the reference
check() { got=$(digest "$1"); if [ "$got" = "$2 $3" ]; then r=PASS; else r=FAIL; fi
  echo "$(date -Is) DIGEST $1 $got ref $2 $3 $r" >> $LOG; [ $r = PASS ]; }
args() { case $1 in code) echo "$E/p-code.txt --nothink";; *) echo "$E/p-$1.txt";; esac; }
ntok() { case $1 in long) echo 256;; *) echo 512;; esac; }
ref() { case $1 in code) echo "8290541d324b 511";; prose) echo "88ed0e11adfd 511";;
  hash) echo "4c37f5fee667 410";; long) echo "c27a3b087fb5 255";; q2code) echo "f0a2cd53a247 511";; esac; }
# fixed cache sizes (dynamic slots, --ssd-streaming-cache-experts N): the
# memory-sized cache's counts on int3 (plain ~4703-4743, DSpark ~4240-4273;
# Q2 plain ~9605-9640, DSpark ~8736-8751)
MXP=4730; MXD=4260; Q2P=9610; Q2D=8740
