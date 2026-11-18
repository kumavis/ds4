#!/usr/bin/env bash
# D17 part B step 2 (development): after q_b0, merge part A's default change,
# apply the union-loading sweep (DS4_ROCM_V41_UNION_SWEEP, off by default),
# commit, incremental build + server unit tests under the lock, then one
# development session with union loading on (max 2048 rows) and the layer
# profile, compared token by token with d17-b0-prof-grow (same tree without
# the union code). Not a gated measurement: int4 gates are still running.
set -u
while systemctl --user is-active --quiet d17-q-b0; do sleep 60; done
cd /data/ds4-d17b || exit 1
L=/data/ds4-d17b/build-b1.log
{
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com merge --no-edit -q d17-agent-turns \
    -m "Merge d17-agent-turns (part A default) into d17b-prefill-union" &&
  nix-shell /data/ds4-env/shell.nix --run "python3 scripts/d17/patch_union.py" < /dev/null &&
  git add ds4.c ds4_gpu.h rocm/ds4_rocm_runtime.cuh rocm/ds4_rocm_current_api_compat.cuh scripts/d17 &&
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -q -F scripts/d17/msg-union.txt
} > $L 2>&1 || { echo "PATCH_EXIT=1" >> $L; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix \
  --run "scripts/d0/build-incremental.sh ds4-server ds4_test && ./ds4_test --server" >> $L 2>&1 < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> $L
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
D17_TWIN_TAG=d17-b0-prof-grow DS4_ROCM_V41_UNION_SWEEP=2048 DS4_METAL_GRAPH_PREFILL_PROFILE=1 D14_BUDGET_MIN=40 \
DS4_NOTE="D17 B development: union-loading sweep on (max 2048 rows) with the layer profile; one session to ~31k" \
  /data/ds4-d17/scripts/d17/run.sh d17-b1-dev-union-grow s1 163840 "" "grow:target=34000:label=grow"
