#!/usr/bin/env bash
# D17 part A queue: turn-end KV checkpoints (DS4_KV_TURN_CHECKPOINTS), same
# binary (/data/ds4-d17) for the off and on twins. Each run.sh call is one
# bounded server session, stopped cleanly; the GPU lock is released between.
#   32k pairs: grow to ~34k with disk KV, then a fresh server restores at the
#   session's last turn (its history ends at the previous turn's reply) and
#   the reply is compared token for token with the uninterrupted session's.
#   100k: one confirmation with the recommended interval.
set -u
R=/data/ds4-d17/scripts/d17/run.sh
export DS4_DIR=/data/ds4-d17
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
export DS4_OUTPUT_CLASS=exact   # part A changes only when checkpoints are written
CTX=163840
KV=/data/ds4-bench/d17-kv
lastdir() { ls -d /data/ds4-bench/runs/*-"$1" 2>/dev/null | tail -1; }
mkdir -p $KV/off $KV/on $KV/on2k $KV/on2k-100k

# off twin: stock rule (exact multiples of the 10,240 interval)
D14_BUDGET_MIN=40 DS4_NOTE="D17 A off: stock continued-checkpoint rule; one session to ~34k with disk KV" \
  $R d17-a-off-grow s1 $CTX "--kv-disk-dir $KV/off --kv-disk-space-mb 32768" "grow:target=34000:label=grow"
D14_BUDGET_MIN=30 DS4_NOTE="D17 A off: fresh server, restore at the last turn of d17-a-off-grow" \
  $R d17-a-off-restore s2 $CTX "--kv-disk-dir $KV/off --kv-disk-space-mb 32768" \
  "resume:from=$(lastdir d17-a-off-grow):src_label=grow:at=last:turns=1:label=restore"

# on: turn-end and mid-prefill checkpoints, default interval (10,240)
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=40 DS4_NOTE="D17 A on: turn checkpoints, default interval; one session to ~34k" \
  $R d17-a-on-grow s1 $CTX "--kv-disk-dir $KV/on --kv-disk-space-mb 32768" "grow:target=34000:label=grow"
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=30 DS4_NOTE="D17 A on: fresh server, restore at the last turn of d17-a-on-grow" \
  $R d17-a-on-restore s2 $CTX "--kv-disk-dir $KV/on --kv-disk-space-mb 32768" \
  "resume:from=$(lastdir d17-a-on-grow):src_label=grow:at=last:turns=1:label=restore"

# on with a 2,048-token interval: a checkpoint at (almost) every turn end
sleep 180   # let the drives cool between long sessions
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=40 DS4_NOTE="D17 A on: turn checkpoints, interval 2048; one session to ~34k" \
  $R d17-a-on2k-grow s1 $CTX "--kv-disk-dir $KV/on2k --kv-disk-space-mb 32768 --kv-cache-continued-interval-tokens 2048" \
  "grow:target=34000:label=grow"
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=30 DS4_NOTE="D17 A on: fresh server, restore at the last turn of d17-a-on2k-grow" \
  $R d17-a-on2k-restore s2 $CTX "--kv-disk-dir $KV/on2k --kv-disk-space-mb 32768 --kv-cache-continued-interval-tokens 2048" \
  "resume:from=$(lastdir d17-a-on2k-grow):src_label=grow:at=last:turns=1:label=restore"

# 100k confirmation (on, interval 2048): replay the D14 132k session's history
# to ~100k (cold prefill, mid-prefill checkpoints) plus one live turn; then a
# fresh server restores at that last turn: restore time and bit-exactness.
sleep 180
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=60 DS4_NOTE="D17 A on: history to ~100k (cold) and one live turn, interval 2048" \
  $R d17-a-on2k-100k-a s2 $CTX "--kv-disk-dir $KV/on2k-100k --kv-disk-space-mb 32768 --kv-cache-continued-interval-tokens 2048" \
  "resume:from=$(lastdir d14-s1-mx-plain-long-b):src_label=grow:at=100000:turns=2:label=grow"
DS4_KV_TURN_CHECKPOINTS=1 D14_BUDGET_MIN=30 DS4_NOTE="D17 A on: fresh server, restore at ~100k (last turn of d17-a-on2k-100k-a)" \
  $R d17-a-on2k-100k-b s2 $CTX "--kv-disk-dir $KV/on2k-100k --kv-disk-space-mb 32768 --kv-cache-continued-interval-tokens 2048" \
  "resume:from=$(lastdir d17-a-on2k-100k-a):src_label=grow:at=last:turns=1:label=restore"
