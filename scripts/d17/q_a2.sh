#!/usr/bin/env bash
# D17 part A, step 2 (binary with DS4_KV_TURN_CHECKPOINT_MIN_TOKENS): a
# checkpoint at every turn end (threshold 512), its restore at the last turn
# (time and token-for-token equality with the uninterrupted session), and the
# off twin's restore on this same binary (stock rule, from d17-a-off-grow's
# checkpoints and transcript).
set -u
R=/data/ds4-d17/scripts/d17/run.sh
export DS4_DIR=/data/ds4-d17
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
export DS4_OUTPUT_CLASS=exact
CTX=163840
KV=/data/ds4-bench/d17-kv
lastdir() { ls -d /data/ds4-bench/runs/*-"$1" 2>/dev/null | tail -1; }
mkdir -p $KV/on512

D17_TWIN_TAG=d17-a-off-grow DS4_KV_TURN_CHECKPOINTS=1 DS4_KV_TURN_CHECKPOINT_MIN_TOKENS=512 D14_BUDGET_MIN=40 \
DS4_NOTE="D17 A on: turn checkpoints at every turn end (threshold 512); one session to ~34k" \
  $R d17-a2-on512-grow s1 $CTX "--kv-disk-dir $KV/on512 --kv-disk-space-mb 32768" "grow:target=34000:label=grow"
DS4_KV_TURN_CHECKPOINTS=1 DS4_KV_TURN_CHECKPOINT_MIN_TOKENS=512 D14_BUDGET_MIN=30 \
DS4_NOTE="D17 A on: fresh server, restore at the last turn of d17-a2-on512-grow" \
  $R d17-a2-on512-restore s2 $CTX "--kv-disk-dir $KV/on512 --kv-disk-space-mb 32768" \
  "resume:from=$(lastdir d17-a2-on512-grow):src_label=grow:at=last:turns=1:label=restore"
D14_BUDGET_MIN=30 DS4_NOTE="D17 A off twin on this binary: fresh server, restore at the last turn of d17-a-off-grow" \
  $R d17-a2-off-restore s2 $CTX "--kv-disk-dir $KV/off --kv-disk-space-mb 32768" \
  "resume:from=$(lastdir d17-a-off-grow):src_label=grow:at=last:turns=1:label=restore"
