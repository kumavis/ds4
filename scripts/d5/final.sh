#!/usr/bin/env bash
# D5 final measurements on the d5-prefetch-x build (c0e3826: buffer-swap
# adoption, 16 speculative chunks in flight): model-free test, smoke, on/off
# pairs (two each, interleaved) on MXFP4 code/prose/hash/long, a DSpark code
# pair, a Q2 code pair, and three single variants on code.
set -u; LOG=/data/ds4-bench/d5f-queue.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5x
P=/data/ds4-env
echo "$(date -Is) start test-d5-prefetch-rocm (c0e3826)" >> $LOG
( cd /data/ds4-d5x && flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make test-d5-prefetch-rocm" ) >> /data/ds4-bench/d5f-tests.log 2>&1
rc=$?; echo "$(date -Is) end test rc=$rc $(grep -E "^(PASS|FAIL)|d5 prefetch test" /data/ds4-bench/d5f-tests.log | tail -2 | tr "\n" " ")" >> $LOG
[ $rc -eq 0 ] || { echo "$(date -Is) STOP (test failed)" >> $LOG; exit 1; }
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5 final, prefetch on${NOTE_X:+ ($NOTE_X)}" run "$t" "$@"; }
off() { t=$1; shift; DS4_NOTE="D5 final build, prefetch off" run "$t" "$@"; }
on d5f-smoke-8 8 $P/p-code.txt --nothink
on d5f-smoke-64 64 $P/p-code.txt --nothink
for r in 1 2; do
  off d5f-mx-code-off-r$r 512 $P/p-code.txt --nothink
  on  d5f-mx-code-on-r$r 512 $P/p-code.txt --nothink
  off d5f-mx-prose-off-r$r 512 $P/p-prose.txt
  on  d5f-mx-prose-on-r$r 512 $P/p-prose.txt
  off d5f-mx-hash-off-r$r 512 $P/p-hash.txt
  on  d5f-mx-hash-on-r$r 512 $P/p-hash.txt
  off d5f-mx-long-off-r$r 256 $P/p-long.txt
  on  d5f-mx-long-on-r$r 256 $P/p-long.txt
done
off d5f-mx-dsp-code-off 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
on  d5f-mx-dsp-code-on 512 $P/p-code.txt --nothink --dspark --mtp-model $DSP
NOTE_X="k 2" DS4_ROCM_D5_K=2 on d5f-mx-code-on-k2 512 $P/p-code.txt --nothink
NOTE_X="depth 16" DS4_ROCM_D5_DEPTH=16 on d5f-mx-code-on-depth16 512 $P/p-code.txt --nothink
NOTE_X="inflight 8" DS4_ROCM_D5_INFLIGHT=8 on d5f-mx-code-on-inflight8 512 $P/p-code.txt --nothink
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
off d5f-q2-code-off 512 $P/p-code.txt --nothink
on  d5f-q2-code-on 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
