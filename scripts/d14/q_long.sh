#!/usr/bin/env bash
# D14 long-session queue (MXFP4, two drives), payload v2. Every run.sh call is
# one bounded server session, stopped cleanly at its end; the GPU lock is
# released between calls so other queues can run.
#   D14_CTXL   --ctx for the long sessions (set from the S4 admission probes)
#   D14_PARTS  which parts to run, default "A B S2 S3 Q2"
# Run as: systemd-run --user --collect -p SendSIGKILL=no -p TimeoutStopSec=infinity
#   --unit=d14-q-long /data/ds4-d14/scripts/d14/q_long.sh
set -u
R=/data/ds4-d14/scripts/d14/run.sh
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
CTXL=${D14_CTXL:-163840}
KV=/data/ds4-bench/d14-kv
DRAFT="--mtp-model /data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf --dspark"
lastdir() { ls -d /data/ds4-bench/runs/*-"$1" 2>/dev/null | tail -1; }
has() { case " ${D14_PARTS:-A PC B S2 S3 Q2} " in *" $1 "*) return 0;; esac; return 1; }
mkdir -p $KV/plain $KV/dspark

if has A; then
  D14_BUDGET_MIN=85 DS4_NOTE="D14 S1+S2: plain, one agent session 0->64k, live reuse, disk KV saves (32 GB)" \
    $R d14-s1-mx-plain-long-a s1-long "$CTXL" "--kv-disk-dir $KV/plain --kv-disk-space-mb 32768" \
    "grow:target=66000:label=grow"
  A1=$(lastdir d14-s1-mx-plain-long-a)
  D14_BUDGET_MIN=88 DS4_NOTE="D14 S1+S2: plain, same session restored from disk at ~64k after a restart, then ->128k+" \
    $R d14-s1-mx-plain-long-b s1-long "$CTXL" "--kv-disk-dir $KV/plain --kv-disk-space-mb 32768" \
    "grow:target=134000:from=$A1:label=grow"
fi
if has B; then
  DS4_DSPARK_SCHED=old D14_BUDGET_MIN=85 DS4_NOTE="D14 S1: DSpark, one agent session 0->64k, live reuse, disk KV saves" \
    $R d14-s1-mx-dspark-long-a s1-long "$CTXL" "$DRAFT --kv-disk-dir $KV/dspark --kv-disk-space-mb 32768" \
    "grow:target=66000:label=grow"
  B1=$(lastdir d14-s1-mx-dspark-long-a)
  DS4_DSPARK_SCHED=old D14_BUDGET_MIN=88 DS4_NOTE="D14 S1: DSpark, same session restored from disk at ~64k, then ->128k+" \
    $R d14-s1-mx-dspark-long-b s1-long "$CTXL" "$DRAFT --kv-disk-dir $KV/dspark --kv-disk-space-mb 32768" \
    "grow:target=134000:from=$B1:label=grow"
fi
if has S2; then
  A1=$(lastdir d14-s1-mx-plain-long-a); A2=$(lastdir d14-s1-mx-plain-long-b)
  for at in 16384 32768; do
    D14_BUDGET_MIN=60 DS4_NOTE="D14 S2: fresh server, resume the plain session at ~$at from disk (2 turns), then the same history cold (salted system prompt)" \
      $R "d14-s2-mx-restore-$((at / 1024))k" s2 "$CTXL" "--kv-disk-dir $KV/plain --kv-disk-space-mb 32768" \
      "resume:from=$A1:src_label=grow:at=$at:turns=2:label=restore" \
      "resume:from=$A1:src_label=grow:at=$at:turns=1:salt=1:label=cold"
  done
  D14_BUDGET_MIN=88 DS4_NOTE="D14 S2: fresh server, resume the plain session at ~100k from disk (2 turns), then the same history cold" \
    $R d14-s2-mx-restore-100k s2 "$CTXL" "--kv-disk-dir $KV/plain --kv-disk-space-mb 32768" \
    "resume:from=$A2:src_label=grow:at=100000:turns=2:label=restore" \
    "resume:from=$A2:src_label=grow:at=100000:turns=1:salt=1:label=cold"
fi
if has S3; then
  for n in 1 2 4; do
    b=""; [ "$n" -gt 1 ] && b="--batched-session $n"
    D14_BUDGET_MIN=40 DS4_NOTE="D14 S3: $n concurrent agent sessions sharing the system prompt, each to ~20k" \
      $R "d14-s3-mx-b$n" s3 32768 "$b" "multi:n=$n:target=20480:label=multi"
  done
fi
if has PC; then
  # the fork's own option: a prefill chunk reads the whole routed-expert set
  # once (~220 GiB at 2048), so a larger chunk should cut prefill reads per token
  for pc in ${D14_PCS:-4096 8192}; do
    D14_BUDGET_MIN=40 DS4_NOTE="D14 S1 variant: --prefill-chunk $pc, plain, one agent session to 32k (compare d14-s1-mx-plain-long-a)" \
      $R "d14-s1-mx-plain-32k-pc$pc" s1-pc "$CTXL" "--prefill-chunk $pc" "grow:target=32768:label=grow"
  done
fi
if has Q2; then
  # Q2 has no replica on the second drive: drop the variable entirely
  env -u DS4_ROCM_STREAM_REPLICA DS4_MODEL=/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  D14_BUDGET_MIN=40 DS4_NOTE="D14 S1 Q2 contrast: one agent session to 32k, plain" \
    $R d14-s1-q2-plain-32k s1 32768 "" "grow:target=32768:label=grow"
fi
