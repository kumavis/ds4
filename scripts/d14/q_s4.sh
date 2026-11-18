#!/usr/bin/env bash
# D14 S4 queue: --ctx admission and plan, and decode at ~7-12k fill, per context
# size; then the memory plan for N resident sessions at --ctx 131072.
# Each run.sh call is its own bounded server session (lock taken and released
# per call). Run as: systemd-run --user --collect -p SendSIGKILL=no
#   -p TimeoutStopSec=infinity --unit=d14-q-s4 /data/ds4-d14/scripts/d14/q_s4.sh
set -u
R=/data/ds4-d14/scripts/d14/run.sh
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_BUDGET_MIN=20
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
for c in ${D14_CTXS:-8192 32768 65536 131072 163840 204800 262144}; do
  t=$c; [ "$t" -gt 16384 ] && t=16384
  DS4_NOTE="D14 S4: --ctx $c: admission and memory plan; first two turns of the agent session (~7-12k fill)" \
    $R "d14-s4-mx-ctx$((c / 1024))k" s4 "$c" "" "grow:target=$t:max_turns=2:label=grow"
done
for n in ${D14_SESSIONS:-2 4}; do
  DS4_NOTE="D14 S3/S4: memory plan for $n resident sessions at --ctx 131072 (one 8-token request)" \
    $R "d14-s4-mx-ctx128k-b$n" s4b 131072 "--batched-session $n" ping:n=1:max=8
done
