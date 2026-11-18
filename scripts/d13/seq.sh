#!/usr/bin/env bash
# D13 sequence gate: one ds4 process, three chat turns through the REPL (stdin
# is not a tty, so linenoise reads plain lines):
#   turn 1: p-long (1,445 tokens -> layer sweep), decode N tokens
#   turn 2: p-code (~100 new tokens -> token-major prefill), decode N tokens
#   turn 3: p-json (~280 new tokens -> second layer sweep; with the switch on
#           this reclaims lent slots that decode filled), decode N tokens
# Usage: seq.sh TAG on|off [N] [extra ds4 flags]. Output in /data/ds4-bench/d13-seq/TAG/.
# Runs under the GPU lock and refuses to start if another ds4 is running.
set -u
tag=$1; mode=$2; n=${3:-96}; shift 3 || shift $#
out=/data/ds4-bench/d13-seq/$tag
mkdir -p "$out"
E=/data/ds4-env
for p in long code json; do tr '\n' ' ' < $E/p-$p.txt | sed 's/  */ /g'; echo; done > "$out/input.txt"
case $mode in on) sc=1;; *) sc=0;; esac
cd /data/ds4-d13
exec 9>/data/ds4-bench/gpu.lock
flock 9
if pgrep -x 'ds4|ds4-server|ds4-bench|ds4-eval|ds4-agent' >/dev/null; then
  echo "another ds4 is running; not starting" > "$out/error"; exit 1
fi
{
  echo "start $(date -Is) commit=$(git rev-parse --short HEAD) mode=$mode n=$n extra=$*"
  env DS4_ROCM_STREAM_STAGING_CACHE=$sc \
      DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf \
      DS4_ROCM_STREAM_CACHE_STATS=1 DS4_ROCM_STREAM_READ_PROFILE=1 DS4_D8_STEP_TRACE=1 \
      nix-shell /data/ds4-env/shell.nix --run "./ds4 -m /data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf --rocm --ssd-streaming --temp 0 --ctx 4096 -n $n --nothink $*" \
      < "$out/input.txt" > "$out/stdout.txt" 2> "$out/stderr.txt"
  echo "EXIT=$? end $(date -Is)"
} > "$out/meta.txt"
md5sum < "$out/stdout.txt" | cut -c1-12 >> "$out/meta.txt"
