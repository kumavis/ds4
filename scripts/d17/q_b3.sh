#!/usr/bin/env bash
# D17 part B step 2b (development): union load time split (sync / readback /
# prepare / read), part A's warning fix merged; incremental build + server
# unit tests under the lock; one development session with union loading
# capped at 512 rows (crossover between 260 and 952 rows), compared token by
# token with the whole-layer run d17-b2-stageprof-grow (its first six turns).
set -u
cd /data/ds4-d17b || exit 1
L=/data/ds4-d17b/build-b3.log
{
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com merge --no-edit -q d17-agent-turns \
    -m "Merge d17-agent-turns (warning fix) into d17b-prefill-union" &&
  nix-shell /data/ds4-env/shell.nix --run "python3 scripts/d17/patch_union_timing.py" < /dev/null &&
  git add ds4.c scripts/d17 &&
  git -c user.name=kumavis -c user.email=kumavis@users.noreply.github.com commit -q -F scripts/d17/msg-timing.txt
} > $L 2>&1 || { echo "PATCH_EXIT=1" >> $L; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix \
  --run "scripts/d0/build-incremental.sh ds4-server ds4_test && ./ds4_test --server" >> $L 2>&1 < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> $L
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
D17_TWIN_TAG=d17-b2-stageprof-grow DS4_ROCM_V41_UNION_SWEEP=512 D14_BUDGET_MIN=40 \
DS4_NOTE="D17 B development: union-loading sweep capped at 512 rows, load time split; one session to ~31k" \
  /data/ds4-d17/scripts/d17/run.sh d17-b3-dev-union512-grow s1 163840 "" "grow:target=34000:label=grow"
