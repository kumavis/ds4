#!/usr/bin/env bash
# D17 part B step 1b (design data): split a sweep layer's time into stages.
# DS4_METAL_V41_STAGE_PROFILE=1 ends (synchronizes) the command batch after
# every stage of a layer and prints its time, so the seed's time no longer
# includes the layer's GPU compute. Union binary, union OFF (also the
# same-binary off twin of d17-b1-dev-union-grow for its first six turns).
set -u
while systemctl --user is-active --quiet d17-q-b1; do sleep 60; done
grep -q "BUILD_TEST_EXIT=0" /data/ds4-d17b/build-b1.log || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
DS4_METAL_V41_STAGE_PROFILE=1 DS4_METAL_GRAPH_PREFILL_PROFILE=1 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 B design: per-stage prefill profile (synchronized stages), union binary with union off; first six turns" \
  /data/ds4-d17/scripts/d17/run.sh d17-b2-stageprof-grow s1 163840 "" "grow:target=34000:max_turns=6:label=grow"
