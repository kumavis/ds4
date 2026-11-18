#!/usr/bin/env bash
# D17 part B item 4: partial staging reclaim for union sweeps
# (DS4_ROCM_STREAM_STAGING_RECLAIM_UNION=1). After q_b4: patch, commit,
# incremental build + server unit tests under the lock, then on one binary:
#   gate   d17-b5-grow-whole (union off) and d17-b5-grow-union-partial
#          (union <= 512 rows + partial reclaim): one session to ~31k each,
#          token identity per turn (every turn starts with a sweep after decode)
#   probe  six 256-token appends with 64-token replies at ~31k, union <= 512
#          rows, partial reclaim off (d17-b5-probe256-full) and on
#          (d17-b5-probe256-partial): union reads per layer, decode misses in
#          the first 16 and 64 reply tokens, lent entries surviving a sweep
set -u
while systemctl --user is-active --quiet d17-q-b4; do sleep 60; done
cd /data/ds4-d17b || exit 1
L=/data/ds4-d17b/build-b5.log
{
  nix-shell /data/ds4-env/shell.nix --run "python3 scripts/d17/patch_reclaim_union.py" < /dev/null &&
  git add ds4.c ds4_gpu.h rocm/ds4_rocm_runtime.cuh rocm/ds4_rocm_current_api_compat.cuh scripts/d17 &&
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -q -F scripts/d17/msg-reclaim.txt
} > $L 2>&1 || { echo "PATCH_EXIT=1" >> $L; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix \
  --run "scripts/d0/build-incremental.sh ds4-server ds4_test && ./ds4_test --server" >> $L 2>&1 < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> $L
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
R=/data/ds4-d17/scripts/d17/run.sh
D14_BUDGET_MIN=40 DS4_NOTE="D17 B gate: whole-layer sweeps (union off), one session to ~31k (twin)" \
  $R d17-b5-grow-whole s1 163840 "" "grow:target=34000:label=grow"
sleep 120
D17_TWIN_TAG=d17-b5-grow-whole DS4_ROCM_V41_UNION_SWEEP=512 DS4_ROCM_STREAM_STAGING_RECLAIM_UNION=1 D14_BUDGET_MIN=40 \
DS4_NOTE="D17 B gate: union sweep <= 512 rows with partial staging reclaim, one session to ~31k" \
  $R d17-b5-grow-union-partial s1 163840 "" "grow:target=34000:label=grow"
SRC=$(ls -d /data/ds4-bench/runs/*-d17-a2-on512-grow | tail -1)
KV="--kv-disk-dir /data/ds4-bench/d17-kv/on512 --kv-disk-space-mb 32768"
STEP="probe:from=$SRC:src_label=grow:sizes=256,256,256,256,256,256:max=64:label=probe256"
DS4_ROCM_V41_UNION_SWEEP=512 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B item 4: six 256-token appends at ~31k, union <= 512, full D13 reclaim" \
  $R d17-b5-probe256-full s2 163840 "$KV" "$STEP"
D17_TWIN_TAG=d17-b5-probe256-full DS4_ROCM_V41_UNION_SWEEP=512 DS4_ROCM_STREAM_STAGING_RECLAIM_UNION=1 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B item 4: six 256-token appends at ~31k, union <= 512, partial staging reclaim" \
  $R d17-b5-probe256-partial s2 163840 "$KV" "$STEP"
