#!/usr/bin/env bash
# D14: S4 leftovers (ctx 8k rerun after the first-turn guard fix; plan for 8
# resident 128k sessions), then the long-session queue.
set -u
R=/data/ds4-d14/scripts/d14/run.sh
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
D14_BUDGET_MIN=20 DS4_NOTE="D14 S4: --ctx 8192: admission and plan; first turn (~7.7k fill); rerun after the guard fix" \
  $R d14-s4-mx-ctx8k-r2 s4 8192 "" "grow:target=8192:max_turns=1:label=grow"
D14_BUDGET_MIN=20 DS4_NOTE="D14 S3/S4: memory plan for 8 resident sessions at --ctx 131072 (one 8-token request)" \
  $R d14-s4-mx-ctx128k-b8 s4b 131072 "--batched-session 8" ping:n=1:max=8
exec /data/ds4-d14/scripts/d14/q_long.sh
