#!/usr/bin/env bash
# D17 part B: withdraw the partial staging reclaim (wrong slot mapping, see
# D17.md), rebuild, then the 100k confirmation: small-append probes at ~102k
# fill restored from Part A's turn checkpoint, stock dispatch (twin) against the
# recommended settings (union sweep <= 512 rows, token-major threshold 1).
set -u
while systemctl --user is-active --quiet d17-q-b5 || pgrep -f "run.sh d17-b5-probe256" >/dev/null; do sleep 60; done
cd /data/ds4-d17b || exit 1
L=/data/ds4-d17b/build-b7.log
{
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com revert --no-edit 6aa474f &&
  git add scripts/d17 && git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -q \
    -m "chore(d17b): scripts for the 100k confirmation; partial reclaim withdrawn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
} > $L 2>&1
grep -q "DS4_ROCM_STREAM_STAGING_RECLAIM_UNION" ds4.c && { echo "REVERT_FAILED" >> $L; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix \
  --run "scripts/d0/build-incremental.sh ds4-server ds4_test && ./ds4_test --server" >> $L 2>&1 < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> $L
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
R=/data/ds4-d17/scripts/d17/run.sh
SRC=$(ls -d /data/ds4-bench/runs/*-d17-a-on2k-100k-a | tail -1)
KV="--kv-disk-dir /data/ds4-bench/d17-kv/on2k-100k --kv-disk-space-mb 32768"
STEP="probe:from=$SRC:src_label=grow:sizes=32,64,96,128,192,256,384,512:max=16:label=probe"
D14_BUDGET_MIN=30 DS4_NOTE="D17 B 100k: appends 32-512 at ~102k, stock dispatch (twin)" \
  $R d17-b7-probe100k-default s2 163840 "$KV" "$STEP"
sleep 120
D17_TWIN_TAG=d17-b7-probe100k-default DS4_OUTPUT_CLASS=approx-prefill-dispatch \
DS4_ROCM_V41_UNION_SWEEP=512 DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B 100k: appends 32-512 at ~102k, union sweep <= 512 rows + token-major threshold 1" \
  $R d17-b7-probe100k-union s2 163840 "$KV" "$STEP"
