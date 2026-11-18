#!/usr/bin/env bash
# D17 part B step 1 (design data): build d17b (kumanix-int4 + part A) under the
# lock, model-free unit tests, then one ~31k plain session with the V4.1
# prefill layer profile (DS4_METAL_GRAPH_PREFILL_PROFILE=1: per layer and
# chunk, map = wait for the expert loader, encode + drain = compute, seed).
set -u
while systemctl --user is-active --quiet d17-q-a2; do sleep 60; done
cd /data/ds4-d17b
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo > build-b0.log 2>&1 && scripts/d0/build-incremental.sh ds4_test >> build-b0.log 2>&1 && ./ds4_test --server >> build-b0.log 2>&1" < /dev/null
rc=$?; echo "BUILD_TEST_EXIT=$rc" >> build-b0.log
[ $rc -eq 0 ] || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
DS4_METAL_GRAPH_PREFILL_PROFILE=1 D14_BUDGET_MIN=40 \
DS4_NOTE="D17 B design: prefill layer profile (map/encode/drain/seed per layer), int4+A, one session to ~31k" \
  /data/ds4-d17/scripts/d17/run.sh d17-b0-prof-grow s1 163840 "" "grow:target=34000:label=grow"
