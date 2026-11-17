#!/usr/bin/env bash
# D5 stage 2: model-free test, smoke (-n 8, -n 64), then on/off pairs (two
# each, interleaved) on MXFP4 code/prose/hash/long, one Q2 code pair, one
# DSpark code pair, and three parameter probes on code.  Defaults of the "on"
# runs: DS4_ROCM_D5_PREFETCH=1, K=1, DEPTH=6, SHELF=4, INFLIGHT=4.
set -u; . "$(dirname "$0")/common.sh"
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm" >> $LOG
( cd /data/ds4-d5 && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) >> /data/ds4-bench/d5-tests.log 2>&1
rc=$?; echo "$(date -Is) end test-d5-prefetch-rocm rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test" /data/ds4-bench/d5-tests.log | tail -2 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5 prefetch on${2:+ ($2)}" run "$1" "${@:3}"; }
off() { DS4_NOTE="D5 build, prefetch off" run "$1" "${@:3}"; }
on d5-pf-smoke-8 "" 8 $P/p-code.txt --nothink
on d5-pf-smoke-64 "" 64 $P/p-code.txt --nothink
for r in 1 2; do
  off d5-mx-code-off-r$r "" 512 $P/p-code.txt --nothink
  on  d5-mx-code-on-r$r "" 512 $P/p-code.txt --nothink
  off d5-mx-prose-off-r$r "" 512 $P/p-prose.txt
  on  d5-mx-prose-on-r$r "" 512 $P/p-prose.txt
  off d5-mx-hash-off-r$r "" 512 $P/p-hash.txt
  on  d5-mx-hash-on-r$r "" 512 $P/p-hash.txt
  off d5-mx-long-off-r$r "" 256 $P/p-long.txt
  on  d5-mx-long-on-r$r "" 256 $P/p-long.txt
done
off d5-mx-dsp-code-off "" 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
on  d5-mx-dsp-code-on "" 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
DS4_ROCM_D5_DEPTH=16 on d5-mx-code-on-depth16 "depth 16" 512 $P/p-code.txt --nothink
DS4_ROCM_D5_INFLIGHT=8 on d5-mx-code-on-inflight8 "inflight 8" 512 $P/p-code.txt --nothink
DS4_ROCM_D5_K=2 on d5-mx-code-on-k2 "k 2" 512 $P/p-code.txt --nothink
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
off d5-q2-code-off "" 512 $P/p-code.txt --nothink
on  d5-q2-code-on "" 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
