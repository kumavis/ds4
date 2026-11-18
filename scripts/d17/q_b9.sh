#!/usr/bin/env bash
# D17 part B, 100k confirmation (i): union sweep <= 512 rows alone (stock
# token-major threshold, so outputs stay stock's), same probes at ~102k as
# d17-b7-probe100k-*, same binary; twin d17-b7-probe100k-default.
set -u
while systemctl --user is-active --quiet d17-q-b8; do sleep 60; done
grep -q "BUILD_TEST_EXIT=0" /data/ds4-d17b/build-b7.log || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
SRC=$(ls -d /data/ds4-bench/runs/*-d17-a-on2k-100k-a | tail -1)
KV="--kv-disk-dir /data/ds4-bench/d17-kv/on2k-100k --kv-disk-space-mb 32768"
STEP="probe:from=$SRC:src_label=grow:sizes=32,64,96,128,192,256,384,512:max=16:label=probe"
D17_TWIN_TAG=d17-b7-probe100k-default DS4_ROCM_V41_UNION_SWEEP=512 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B 100k: appends 32-512 at ~102k, union sweep <= 512 rows alone (stock threshold)" \
  /data/ds4-d17/scripts/d17/run.sh d17-b9-probe100k-unionalone s2 163840 "$KV" "$STEP"
