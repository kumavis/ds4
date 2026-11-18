#!/usr/bin/env bash
# D17 part B step (c): the token-major threshold as a setting, and union
# loading on small appends, measured as TTFT by append size at ~31k fill.
# One binary (d17b + DS4_ROCM_V41_TOKEN_MAJOR_MAX); four server sessions, each
# restoring the d17-a2-on512 session (31k, turn checkpoint at its last turn
# end), sending a 24-token warm-up, then appends of 32..512 tokens (reply cap 16):
#   default   stock dispatch (token-major below 256, whole-layer sweep above)
#   sweep     DS4_ROCM_V41_TOKEN_MAJOR_MAX=1: whole-layer sweep for every append
#   tokmajor  DS4_ROCM_V41_TOKEN_MAJOR_MAX=1024: token-major up to 1023 rows
#   union     DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 + DS4_ROCM_V41_UNION_SWEEP=512
set -u
while systemctl --user is-active --quiet d17-q-b3; do sleep 60; done
cd /data/ds4-d17b || exit 1
L=/data/ds4-d17b/build-b4.log
{
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com merge --no-edit -q d17-agent-turns \
    -m "Merge d17-agent-turns (warning fix) into d17b-prefill-union" &&
  nix-shell /data/ds4-env/shell.nix --run "python3 scripts/d17/patch_tokmajor.py" < /dev/null &&
  git add ds4.c scripts/d17 &&
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -q -F scripts/d17/msg-tokmajor.txt
} > $L 2>&1 || { echo "PATCH_EXIT=1" >> $L; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix \
  --run "scripts/d0/build-incremental.sh ds4-server ds4_test && ./ds4_test --server" >> $L 2>&1 < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> $L
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
R=/data/ds4-d17/scripts/d17/run.sh
SRC=$(ls -d /data/ds4-bench/runs/*-d17-a2-on512-grow | tail -1)
KV="--kv-disk-dir /data/ds4-bench/d17-kv/on512 --kv-disk-space-mb 32768"
STEP="probe:from=$SRC:src_label=grow:sizes=32,64,96,128,192,256,384,512:max=16:label=probe"
D14_BUDGET_MIN=30 DS4_NOTE="D17 B (c): appends 32-512 at ~31k, stock dispatch (twin)" \
  $R d17-b4-probe-default s2 163840 "$KV" "$STEP"
D17_TWIN_TAG=d17-b4-probe-default DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B (c): appends 32-512 at ~31k, always a whole-layer sweep" \
  $R d17-b4-probe-sweep s2 163840 "$KV" "$STEP"
D17_TWIN_TAG=d17-b4-probe-default DS4_ROCM_V41_TOKEN_MAJOR_MAX=1024 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B (c): appends 32-512 at ~31k, token-major up to 1023 rows" \
  $R d17-b4-probe-tokmajor s2 163840 "$KV" "$STEP"
D17_TWIN_TAG=d17-b4-probe-default DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 DS4_ROCM_V41_UNION_SWEEP=512 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B (c): appends 32-512 at ~31k, sweep with union loading up to 512 rows" \
  $R d17-b4-probe-union s2 163840 "$KV" "$STEP"
