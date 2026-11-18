#!/usr/bin/env bash
# D18 queue: build (trace-only change: Q/q prompt-sync marks), gates, then one
# traced plain MXFP4 agent session to ~100k through ds4-server in two server
# sessions (0 -> 66k with disk KV saves; restart, restore, -> 100k), as D14 part A.
set -u
LOG=/data/ds4-bench/d18-queue.log
export DS4_DIR=/data/ds4-d18
. /data/ds4-d12/scripts/d12/common.sh       # bench helpers (run/erun/gate/digest)
LOG=/data/ds4-bench/d18-queue.log
T=/data/ds4-bench/d18-traces; mkdir -p $T /data/ds4-bench/d18-kv/plain
note() { echo "$(date -Is) $*" >> $LOG; }
cd $DS4_DIR
note "D18 build start $(git rev-parse --short HEAD)"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; note "build done rc=$rc"
[ $rc = 0 ] || { note "QUEUE DONE (build failed)"; exit 1; }
P=$E/p-code.txt
erun DS4_NOTE="d18 gate, trace off" d18-gate-code-8 8 $P --nothink;     gate d18-gate-code-8 b1cd778da121/7
erun DS4_NOTE="d18 gate, trace off" d18-gate-code-64 64 $P --nothink;   gate d18-gate-code-64 3098d05a1cef/63
erun DS4_D12_CACHE_TRACE=$T/d18-gate-code-tr-64.trace DS4_NOTE="d18 gate, trace on" d18-gate-code-tr-64 64 $P --nothink
gate d18-gate-code-tr-64 3098d05a1cef/63
n=$(grep -c "^M [Qq] " $T/d18-gate-code-tr-64.trace); note "trace Q/q marks in gate trace: $n"
[ "$n" -ge 2 ] || { note "STOP: no Q/q marks; QUEUE DONE"; exit 1; }
erun DS4_NOTE="d18 gate, trace off" d18-gate-code-512 512 $P --nothink; gate d18-gate-code-512 8290541d324b/511
# traced session
R=$DS4_DIR/scripts/d18/run.sh
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
KV="--kv-disk-dir /data/ds4-bench/d18-kv/plain --kv-disk-space-mb 32768"
note "session a start"
DS4_D12_CACHE_TRACE=$T/d18-s1-plain-trace-a.trace D14_BUDGET_MIN=85 \
  DS4_NOTE="D18: traced plain session 0->66k (DS4_D12_CACHE_TRACE), live reuse, disk KV saves" \
  $R d18-s1-plain-trace-a s1-long 163840 "$KV" "grow:target=66000:label=grow"
note "session a done rc=$?"
A1=$(ls -d /data/ds4-bench/runs/*-d18-s1-plain-trace-a | tail -1)
DS4_D12_CACHE_TRACE=$T/d18-s1-plain-trace-b.trace D14_BUDGET_MIN=88 \
  DS4_NOTE="D18: traced plain session restored from disk at ~64k after a restart, then ->100k" \
  $R d18-s1-plain-trace-b s1-long 163840 "$KV" "grow:target=100000:from=$A1:label=grow"
note "session b done rc=$?"
gzip -f $T/d18-s1-plain-trace-a.trace $T/d18-s1-plain-trace-b.trace $T/d18-gate-code-tr-64.trace
note "QUEUE DONE"
