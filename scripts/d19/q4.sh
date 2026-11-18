#!/usr/bin/env bash
# D19 queue 4: S3-FIFO measured (binary from q2, d79081f). Single prompts off
# (lru) / on (s3fifo); 32k sessions s3fifo and s3fifo + seed cold (the lru twins
# are q3's d19-s1-*-32k-always/-cold, same code path); traced session to ~100k
# with s3fifo + seed cold.
# Resume at an unused tag with: q4.sh --from TAG. Earlier completed or
# independent stages are skipped. A resumed treatment or restore checks its prerequisite.
set -eu
start_at=
if [[ $# -ne 0 ]]; then
  [[ $# == 2 && $1 == --from && -n $2 ]] || { echo "usage: q4.sh [--from TAG]" >&2; exit 2; }
  start_at=$2
fi
selected() {
  if [[ $1 == "$start_at" ]]; then start_at=; fi
  [[ -z $start_at ]]
}
[[ -n ${INVOCATION_ID:-} ]] || { echo "q4 requires a user systemd unit" >&2; exit 1; }
export DS4_DIR=/data/ds4-d19
LOG=/data/ds4-bench/d19-queue.log
. "$DS4_DIR/scripts/d12/common.sh"
LOG=/data/ds4-bench/d19-queue.log
T=/data/ds4-bench/d19-traces; mkdir -p $T /data/ds4-bench/d19-kv/s3
note() { echo "$(date -Is) $*" >> $LOG; }
qcheck() {
  local cmd
  printf -v cmd '%q ' python3 "$DS4_DIR/scripts/d19/q4_check.py" "$@"
  nix-shell /data/ds4-env/shell.nix --run "$cmd"
}
before() {
  [[ ! -e /data/ds4-bench/PAUSED ]] || { note "STOP before $1: PAUSED"; exit 1; }
  qcheck fresh "$1"
}
# Retain the historic D12 digest/count helpers, but verify the raw run and the
# structured telemetry immediately after each benchmark (including absent fields).
run() {
  local tag=$1
  before "$tag" || return 1
  note "start $tag"
  "$B" "$@" || { note "STOP after $tag: bench failed"; return 1; }
  qcheck check "$tag" || { note "STOP after $tag: record check failed"; return 1; }
  note "end $tag"
}
exec 8>>/data/ds4-bench/d19-q4.lock
flock -n 8 || { note "q4 already active"; exit 1; }
S3="DS4_ROCM_STREAM_EVICT_POLICY=s3fifo"
P=$E/p-code.txt
note "D19 q4 start $(git -C $DS4_DIR rev-parse --short HEAD) binary $(stat -c %y $DS4_DIR/ds4-server)"
for p in code prose long; do
  case $p in
    code) A="512 $P --nothink"; want=8290541d324b/511;;
    prose) A="512 $E/p-prose.txt"; want=88ed0e11adfd/511;;
    long) A="256 $E/p-long.txt"; want=c27a3b087fb5/255;;
  esac
  off=d19-m3-$p-lru
  on=d19-m3-$p-s3
  if selected "$off"; then
    erun DS4_ROCM_STREAM_EVICT_POLICY=lru DS4_NOTE="d19 measure evict lru" "$off" $A
    gate "$off" "$want"
  elif [[ $start_at == "$on" ]]; then
    qcheck check "$off"; gate "$off" "$want"
  fi
  if selected "$on"; then
    erun $S3 DS4_NOTE="d19 measure evict s3fifo" "$on" $A
    gate "$on" "$want"
  fi
done
for p in code json; do
  if [ $p = code ]; then A="$P --nothink"; cycles=172; accepted=340; else A="$E/p-json.txt"; cycles=207; accepted=305; fi
  off=d19-m3-dsp-$p-lru
  on=d19-m3-dsp-$p-s3
  if selected "$off"; then
    erun $TR DS4_ROCM_STREAM_EVICT_POLICY=lru DS4_NOTE="d19 measure DSpark old evict lru" "$off" 512 $A --dspark --mtp-model "$DSP"
    dgate "$off" "$cycles" "$accepted"
  elif [[ $start_at == "$on" ]]; then
    qcheck check "$off"; dgate "$off" "$cycles" "$accepted"
  fi
  if selected "$on"; then
    erun $TR $S3 DS4_NOTE="d19 measure DSpark old evict s3fifo" "$on" 512 $A --dspark --mtp-model "$DSP"
    dgate "$on" "$cycles" "$accepted"
  fi
done
R=$DS4_DIR/scripts/d19/run.sh
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
DRAFT="--mtp-model $DSP --dspark"
sess() {
  selected "$1" || return 0
  before "$1" || return 1
  note "start session $1"
  "$R" "$@" || { note "STOP after $1: session failed"; return 1; }
  verified_dir=$(qcheck check "$1" "$5") || { note "STOP after $1: session record failed"; return 1; }
  note "end session $1: target and telemetry verified"
}
D14_BUDGET_MIN=50 DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_NOTE="D19 plain session to 32k, s3fifo" \
  sess d19-s1-plain-32k-s3 s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 plain session to 32k, s3fifo + seed cold" \
  sess d19-s1-plain-32k-s3cold s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_DSPARK_SCHED=old DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 DSpark session to 32k, s3fifo + seed cold" \
  sess d19-s1-dsp-32k-s3cold s1 163840 "$DRAFT" "grow:target=32768:label=grow"
if [[ -z $start_at || $start_at == d19-s1-plain-s3cold-trace-a ]] && [[ -n $(ls -A /data/ds4-bench/d19-kv/s3) ]]; then
  note "STOP before trace-a: existing checkpoints in d19-kv/s3"; exit 1
fi
KV="--kv-disk-dir /data/ds4-bench/d19-kv/s3 --kv-disk-space-mb 32768"
DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-s3cold-trace-a.trace D14_BUDGET_MIN=85 \
  DS4_NOTE="D19 traced plain session 0->66k, s3fifo + seed cold, disk KV saves" \
  sess d19-s1-plain-s3cold-trace-a s1-long 163840 "$KV" "grow:target=66000:label=grow"
case $start_at in
  "") A1=$verified_dir ;;
  d19-s1-plain-s3cold-trace-b)
    A1=$(qcheck check d19-s1-plain-s3cold-trace-a "grow:target=66000:label=grow")
    [[ -n $(ls -A /data/ds4-bench/d19-kv/s3) ]] || { note "STOP: missing restore checkpoints"; exit 1; } ;;
  *) echo "unknown q4 tag: $start_at" >&2; exit 2 ;;
esac
DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-s3cold-trace-b.trace D14_BUDGET_MIN=88 \
  DS4_NOTE="D19 traced plain session restored at ~66k, s3fifo + seed cold, ->100k" \
  sess d19-s1-plain-s3cold-trace-b s1-long 163840 "$KV" "grow:target=100000:from=$A1:label=grow"
gzip -- "$T/d19-s1-plain-s3cold-trace-a.trace" "$T/d19-s1-plain-s3cold-trace-b.trace"
note "QUEUE DONE (q4)"
