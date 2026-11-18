#!/usr/bin/env bash
# Pause guard (added by the PM 2026-11-19): do not start while /data/ds4-bench/PAUSED exists.
[ -e /data/ds4-bench/PAUSED ] && { echo "$(date -Is) QUEUE 2 not started: /data/ds4-bench/PAUSED exists (project paused)" >> /data/ds4-bench/d22-queue.log; exit 0; }
# D22 queue 2 (agent sessions, MXFP4 two drives, payload v2, harness from
# D14/D17 with D22's procmem sampler and hold/release steps):
#  1. 32k plain session, no shed vs forced sheds (per-turn reply_tok_sha twin)
#  2. pressure test: plain 32k session, hold memory between turns, guard sheds
#  3. proof: DSpark 0 -> 128k+ in one server session at --ctx 163840, then plain
# Each run.sh call is one bounded server session holding gpu.lock, stopped by
# the driver's clean stop (one SIGTERM).  Run as a user unit with
# SendSIGKILL=no TimeoutStopSec=infinity; never stop it.
set -u
R=/data/ds4-d22/scripts/d22/harness/run.sh
LOG=/data/ds4-bench/d22-queue.log
export DS4_DIR=/data/ds4-d22
export DS4_ROCM_STREAM_REPLICA=/home/xyz/ds4-replica/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
KV=/data/ds4-bench/d22-kv
DRAFT="--mtp-model /data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf --dspark"
mkdir -p $KV/plain $KV/dspark
lastdir() { ls -d /data/ds4-bench/runs/*-"$1" 2>/dev/null | tail -1; }
note() { d=$(lastdir $1); echo "$(date -Is) end $1 $(grep -o 'EXIT=[0-9]* wall_s=[0-9]*' $d/run.log | tail -1) turns=$(grep -c . $d/turns.jsonl 2>/dev/null) sheds=$(grep -c 'memory guard shed ' $d/server.log) summary=\"$(grep -o 'memory guard summary: .*' $d/server.log | tail -1)\" dir=$d" >> $LOG; }
echo "$(date -Is) QUEUE 2 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG

for v in noshed shed; do
  t=d22-s1-plain-32k-$v; echo "$(date -Is) start $t" >> $LOG
  if [ $v = shed ]; then export DS4_ROCM_STREAM_MEM_GUARD_FAKE=3,300,900,1500,2100; else unset DS4_ROCM_STREAM_MEM_GUARD_FAKE; fi
  D14_BUDGET_MIN=45 DS4_NOTE="D22: plain S1 to 32k, forced sheds $v (twin by reply_tok_sha)" \
    $R $t s1 32768 "" grow:target=32768:label=grow
  note $t
done
unset DS4_ROCM_STREAM_MEM_GUARD_FAKE

t=d22-s1-pressure; echo "$(date -Is) start $t" >> $LOG
D14_BUDGET_MIN=45 DS4_NOTE="D22: pressure test, plain 32k: grow to 16k, take memory in 256 MiB steps (max 1.5 GiB, stop under the guard line, hard release under floor+400), 2+ turns held, release, 1+ turn" \
  $R $t s1 32768 "" grow:target=16000:label=g1 hold:max_mb=1536 \
  grow:target=24000:from=SELF:src_label=g1:norm=0:label=g2 release \
  grow:target=28000:from=SELF:src_label=g2:norm=0:label=g3
note $t

t=d22-s1-dsp-long; echo "$(date -Is) start $t" >> $LOG
DS4_DSPARK_SCHED=old D14_BUDGET_MIN=200 DS4_NOTE="D22 proof: DSpark, one agent session 0 -> 128k+ in one server session, disk KV saves, guard + DSpark charge on" \
  $R $t s1-long 163840 "$DRAFT --kv-disk-dir $KV/dspark --kv-disk-space-mb 32768" grow:target=134000:label=grow
note $t
t=d22-s1-plain-long; echo "$(date -Is) start $t" >> $LOG
D14_BUDGET_MIN=200 DS4_NOTE="D22 proof: plain, one agent session 0 -> 128k+ in one server session, disk KV saves, guard on" \
  $R $t s1-long 163840 "--kv-disk-dir $KV/plain --kv-disk-space-mb 32768" grow:target=134000:label=grow
note $t
echo "$(date -Is) QUEUE 2 DONE" >> $LOG
