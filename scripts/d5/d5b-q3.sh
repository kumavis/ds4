#!/usr/bin/env bash
# D5b queue 3 (binary: /data/ds4-d5, 4b91934): plain digest gates with
# everything on (DS4_ROCM_D5_PREFETCH=1: single-token and verify prefetch
# at their defaults) and off on prose, hash, long and Q2 code.  Code was
# gated in queue 1 (d5b-a-off/d5b-ref-code, d5b-a-on/d5b-ref-code-on).
set -u; LOG=/data/ds4-bench/d5b-q3.log; . "$(dirname "$0")/common.sh"
export DS4_DIR=/data/ds4-d5
P=/data/ds4-env
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
on() { t=$1; shift; DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="D5b gate on" run "$t" "$@"; }
off() { t=$1; shift; DS4_NOTE="D5b gate off" run "$t" "$@"; }
off d5b-g-prose-off 512 $P/p-prose.txt
on  d5b-g-prose-on 512 $P/p-prose.txt
off d5b-g-hash-off 512 $P/p-hash.txt
on  d5b-g-hash-on 512 $P/p-hash.txt
off d5b-g-long-off 256 $P/p-long.txt
on  d5b-g-long-on 256 $P/p-long.txt
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
off d5b-g-q2-code-off 512 $P/p-code.txt --nothink
on  d5b-g-q2-code-on 512 $P/p-code.txt --nothink
echo "$(date -Is) QUEUE DONE" >> $LOG
