#!/usr/bin/env bash
# D17 part B: how far apart the two prefill paths are on a single prompt.
# DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 makes short prompts (under 256 tokens) prefill
# with a layer sweep instead of token-major decode steps. Code prompt (94
# tokens), MXFP4, 256 tokens, the usual bench.sh divergence fields:
#   d17-b8-ref-code         stock dispatch, logits logged (reference)
#   d17-b8-tm1-code         threshold 1, free-running against the reference
#   d17-b8-tm1-code-tf      threshold 1, teacher-forced along the reference
set -u
while systemctl --user is-active --quiet d17-q-b7; do sleep 60; done
grep -q "BUILD_TEST_EXIT=0" /data/ds4-d17b/build-b7.log || exit 1
export DS4_DIR=/data/ds4-d17b
export DS4_MODEL=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
B=/data/ds4-env/bench.sh
DS4_D0_LOGITS=auto DS4_NOTE="D17 B: reference for the prefill-dispatch divergence (stock, code prompt)" \
  $B d17-b8-ref-code 256 /data/ds4-env/p-code.txt --nothink
DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 DS4_OUTPUT_CLASS=approx-prefill-dispatch DS4_REF_TAG=d17-b8-ref-code \
DS4_NOTE="D17 B: DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 (prompt prefilled by a sweep), free-running vs reference" \
  $B d17-b8-tm1-code 256 /data/ds4-env/p-code.txt --nothink
DS4_ROCM_V41_TOKEN_MAJOR_MAX=1 DS4_OUTPUT_CLASS=approx-prefill-dispatch DS4_REF_TAG=d17-b8-ref-code DS4_REF_FORCE=1 \
DS4_NOTE="D17 B: DS4_ROCM_V41_TOKEN_MAJOR_MAX=1, teacher-forced along the reference" \
  $B d17-b8-tm1-code-tf 256 /data/ds4-env/p-code.txt --nothink
