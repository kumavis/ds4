#!/usr/bin/env bash
# D17 part B, 100k confirmation: the small-append probes at ~102k fill,
# restored from Part A's turn checkpoint of d17-a-on2k-100k-a, stock dispatch
# (twin) against the recommended settings (union sweep <= 512 rows, partial
# staging reclaim, token-major threshold 1). Same binary as q_b5.
set -u
while systemctl --user is-active --quiet d17-q-b5; do sleep 60; done
grep -q "BUILD_TEST_EXIT=0" /data/ds4-d17b/build-b5.log || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
R=/data/ds4-d17/scripts/d17/run.sh
SRC=$(ls -d /data/ds4-bench/runs/*-d17-a-on2k-100k-a | tail -1)
KV="--kv-disk-dir /data/ds4-bench/d17-kv/on2k-100k --kv-disk-space-mb 32768"
STEP="probe:from=$SRC:src_label=grow:sizes=32,64,96,128,192,256,384,512:max=16:label=probe"
sleep 120
D14_BUDGET_MIN=30 DS4_NOTE="D17 B 100k: appends 32-512 at ~102k, stock dispatch (twin)" \
  $R d17-b6-probe100k-default s2 163840 "$KV" "$STEP"
D17_TWIN_TAG=d17-b6-probe100k-default DS4_OUTPUT_CLASS=approx-prefill-dispatch \
DS4_ROCM_V41_UNION_SWEEP=512 DS4_ROCM_STREAM_STAGING_RECLAIM_UNION=1 DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B 100k: appends 32-512 at ~102k, union <= 512 + partial reclaim + token-major threshold 1" \
  $R d17-b6-probe100k-union s2 163840 "$KV" "$STEP"
