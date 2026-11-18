#!/usr/bin/env bash
# D12 queue: build, smoke (trace off and on), gates, cache-access traces.
set -u; . "$(dirname "$0")/common.sh"
cd $DS4_DIR
echo "$(date -Is) D12 build start $(git rev-parse --short HEAD)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log
echo "$(date -Is) build done rc=$rc" >> $LOG
[ $rc = 0 ] || { echo "$(date -Is) QUEUE DONE (build failed)" >> $LOG; exit 1; }
P=$E/p-code.txt
# smoke: -n 8 then -n 64, trace off and on
erun DS4_NOTE="d12 smoke plain, trace off" d12-smoke-plain-8 8 $P --nothink;      gate d12-smoke-plain-8 b1cd778da121/7
trun DS4_NOTE="d12 smoke plain, trace on" d12-smoke-plain-tr-8 8 $P --nothink;    gate d12-smoke-plain-tr-8 b1cd778da121/7
erun DS4_NOTE="d12 smoke plain, trace off" d12-smoke-plain-64 64 $P --nothink;    gate d12-smoke-plain-64 3098d05a1cef/63
trun DS4_NOTE="d12 smoke plain, trace on" d12-smoke-plain-tr-64 64 $P --nothink;  gate d12-smoke-plain-tr-64 3098d05a1cef/63
trun $TR DS4_NOTE="d12 smoke DSpark old, trace on" d12-smoke-dsp-tr-64 64 $P --nothink --dspark --mtp-model $DSP
# gates at 512: trace off and on
erun DS4_NOTE="d12 gate plain code, trace off" d12-gate-plain-code 512 $P --nothink;  gate d12-gate-plain-code 8290541d324b/511
trun DS4_NOTE="d12 trace plain code" d12-tr-plain-code 512 $P --nothink;             gate d12-tr-plain-code 8290541d324b/511
erun $TR DS4_NOTE="d12 gate DSpark old code, trace off" d12-gate-dsp-code 512 $P --nothink --dspark --mtp-model $DSP; dgate d12-gate-dsp-code 172 340
trun $TR DS4_NOTE="d12 trace DSpark old code" d12-tr-dsp-code 512 $P --nothink --dspark --mtp-model $DSP;             dgate d12-tr-dsp-code 172 340
# plain traces
trun DS4_NOTE="d12 trace plain prose" d12-tr-plain-prose 512 $E/p-prose.txt; gate d12-tr-plain-prose 88ed0e11adfd/511
for p in json translate list; do
  trun DS4_NOTE="d12 trace plain $p" d12-tr-plain-$p 512 $E/p-$p.txt
done
trun DS4_NOTE="d12 trace plain after long prompt" d12-tr-plain-long 256 $E/p-long.txt
# DSpark traces
for p in prose json translate list; do
  read -r -a A <<< "$(args $p)"
  trun $TR DS4_NOTE="d12 trace DSpark old $p" d12-tr-dsp-$p 512 "${A[@]}" --dspark --mtp-model $DSP
done
# long plain traces (steady state)
trun DS4_CTX=4096 DS4_NOTE="d12 trace plain code 2048" d12-tr-plain-code-2k 2048 $P --nothink
trun DS4_CTX=4096 DS4_NOTE="d12 trace plain prose 2048" d12-tr-plain-prose-2k 2048 $E/p-prose.txt
echo "$(date -Is) QUEUE DONE" >> $LOG
