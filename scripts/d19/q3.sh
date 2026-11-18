#!/usr/bin/env bash
# D19 queue 3 (refill change measured; binary from q1 = int4 6a48b38 + the
# seed-policy switch + D18 trace marks): single prompts off/on, plain and DSpark
# 32k sessions off/on, then a traced plain session to ~100k with cold on.
set -u
export DS4_DIR=/data/ds4-d19
LOG=/data/ds4-bench/d19-queue.log
. /data/ds4-d12/scripts/d12/common.sh
LOG=/data/ds4-bench/d19-queue.log
T=/data/ds4-bench/d19-traces; mkdir -p $T /data/ds4-bench/d19-kv/plain
note() { echo "$(date -Is) $*" >> $LOG; }
COLD="DS4_ROCM_STREAM_SEED_POLICY=cold"
P=$E/p-code.txt
note "D19 q3 start $(git -C $DS4_DIR rev-parse --short HEAD) binary $(stat -c %y $DS4_DIR/ds4-server)"
erun DS4_NOTE="d19 measure, seed always" d19-m-code-always 512 $P --nothink; gate d19-m-code-always 8290541d324b/511
erun $COLD DS4_NOTE="d19 measure, seed cold" d19-m-code-cold 512 $P --nothink; gate d19-m-code-cold 8290541d324b/511
erun DS4_NOTE="d19 measure, seed always" d19-m-long-always 256 $E/p-long.txt; gate d19-m-long-always c27a3b087fb5/255
erun $COLD DS4_NOTE="d19 measure, seed cold" d19-m-long-cold 256 $E/p-long.txt; gate d19-m-long-cold c27a3b087fb5/255
R=$DS4_DIR/scripts/d19/run.sh
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
DRAFT="--mtp-model $DSP --dspark"
sess() { note "start session $1"; "$R" "$@"; note "end session $1 rc=$?"; }
D14_BUDGET_MIN=50 DS4_NOTE="D19 plain session to 32k, seed always (int4 default)" \
  sess d19-s1-plain-32k-always s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 plain session to 32k, seed cold" \
  sess d19-s1-plain-32k-cold s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_DSPARK_SCHED=old DS4_NOTE="D19 DSpark session to 32k, seed always" \
  sess d19-s1-dsp-32k-always s1 163840 "$DRAFT" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_DSPARK_SCHED=old DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 DSpark session to 32k, seed cold" \
  sess d19-s1-dsp-32k-cold s1 163840 "$DRAFT" "grow:target=32768:label=grow"
KV="--kv-disk-dir /data/ds4-bench/d19-kv/plain --kv-disk-space-mb 32768"
DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-cold-trace-a.trace D14_BUDGET_MIN=85 \
  DS4_NOTE="D19 traced plain session 0->66k, seed cold, disk KV saves" \
  sess d19-s1-plain-cold-trace-a s1-long 163840 "$KV" "grow:target=66000:label=grow"
A1=$(ls -d /data/ds4-bench/runs/*-d19-s1-plain-cold-trace-a | tail -1)
DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-cold-trace-b.trace D14_BUDGET_MIN=88 \
  DS4_NOTE="D19 traced plain session restored at ~66k, seed cold, ->100k" \
  sess d19-s1-plain-cold-trace-b s1-long 163840 "$KV" "grow:target=100000:from=$A1:label=grow"
gzip -f $T/d19-s1-plain-cold-trace-*.trace
note "QUEUE DONE (q3)"
