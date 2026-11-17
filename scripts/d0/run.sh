#!/usr/bin/env bash
# usage: run.sh TAG NTOK PROMPT MODEL(q2|mx) [extra flags]; env passthrough
tag=$1 n=$2 p=$3 mdl=$4; shift 4
case $mdl in
 q2) export DS4_MODEL=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf;;
 mx) export DS4_MODEL=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
     export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf;;
esac
export DS4_DIR=${DS4_DIR:-/data/ds4-d0} DS4_TOKEN_TIMING=1
exec /data/ds4-env/bench.sh "$tag" "$n" "/data/ds4-env/$p" "$@"
