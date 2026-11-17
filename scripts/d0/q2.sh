#!/usr/bin/env bash
# D0 queue 2: references, gate, setting matrix, shadow statistics.
cd /data/ds4-d0
G=scripts/d0/gate.sh
N=256
DSP="--nothink --dspark --mtp-model /data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf --ssd-streaming-cache-experts 82GB"
for m in q2 mx; do
  # plain references + fast free-running gate on three prompts
  $G d0g $N fast free $m "p-code.txt p-prose.txt p-hash.txt"
  # setting matrix on two prompts
  $G d0g $N exact free $m "p-code.txt p-prose.txt"
  $G d0g $N attn-moe free $m "p-code.txt p-prose.txt"
  for s in fast attn attn-moe; do $G d0g $N $s forced $m "p-code.txt p-prose.txt"; done
  # shadow: batched rows vs exact rows on identical state, every round
  for p in code prose hash; do
    DS4_REF_TAG=d0g-$m-$p-plain DS4_D0_SHADOW=1 DS4_DSPARK_VERIFY=fast DS4_ROCM_DSPARK_SSD_EXPERIMENT=1 \
      DS4_NOTE="d0 shadow fast (exact rows committed)" scripts/d0/run.sh d0g-$m-$p-shadow-fast $N p-$p.txt $m $DSP
  done
done
echo QUEUE2 DONE
