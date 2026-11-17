#!/usr/bin/env bash
# D0 queue 1: rebuild test, kernel test, smoke shadows
cd /data/ds4-d0
DSP="--nothink --dspark --mtp-model /data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf --ssd-streaming-cache-experts 82GB"
export DS4_ROCM_DSPARK_SSD_EXPERIMENT=1 DS4_D0_LOGITS=auto
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "scripts/d0/build-incremental.sh tests/test_d0_verify_rows_rocm" > build-inc.log 2>&1 || { echo BUILD FAIL; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run ./tests/test_d0_verify_rows_rocm > d0-ktest.log 2>&1; echo "KTEST EXIT=$?"
DS4_D0_SHADOW=1 DS4_D0_SHADOW_TRACE=2 DS4_DSPARK_VERIFY=fast DS4_NOTE="d0 shadow smoke mx fast" scripts/d0/run.sh d0-shadow-mx-fast-8 8 p-code.txt mx $DSP
DS4_D0_SHADOW=1 DS4_D0_SHADOW_TRACE=2 DS4_DSPARK_VERIFY=attn DS4_NOTE="d0 shadow smoke mx attn" scripts/d0/run.sh d0-shadow-mx-attn-8 8 p-code.txt mx $DSP
DS4_D0_SHADOW=1 DS4_D0_SHADOW_TRACE=2 DS4_DSPARK_VERIFY=attn-moe DS4_NOTE="d0 shadow smoke q2 attn-moe" scripts/d0/run.sh d0-shadow-q2-attnmoe-8 8 p-code.txt q2 $DSP
DS4_D0_SHADOW=1 DS4_D0_SHADOW_TRACE=2 DS4_DSPARK_VERIFY=fast DS4_NOTE="d0 shadow smoke q2 fast trace" scripts/d0/run.sh d0-shadow-q2-fast-8 8 p-code.txt q2 $DSP
echo QUEUE1 DONE
