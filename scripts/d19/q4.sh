#!/usr/bin/env bash
# D19 queue 4: S3-FIFO measured (binary from q2, d79081f). Single prompts off
# (lru) / on (s3fifo); 32k sessions s3fifo and s3fifo + seed cold (the lru twins
# are q3's d19-s1-*-32k-always/-cold, same code path); traced session to ~100k
# with s3fifo + seed cold.
set -u
export DS4_DIR=/data/ds4-d19
LOG=/data/ds4-bench/d19-queue.log
. /data/ds4-d12/scripts/d12/common.sh
LOG=/data/ds4-bench/d19-queue.log
T=/data/ds4-bench/d19-traces; mkdir -p $T /data/ds4-bench/d19-kv/s3
note() { echo "$(date -Is) $*" >> $LOG; }
S3="DS4_ROCM_STREAM_EVICT_POLICY=s3fifo"
P=$E/p-code.txt
note "D19 q4 start $(git -C $DS4_DIR rev-parse --short HEAD) binary $(stat -c %y $DS4_DIR/ds4-server)"
for p in code prose long; do
  case $p in
    code) A="512 $P --nothink"; want=8290541d324b/511;;
    prose) A="512 $E/p-prose.txt"; want=88ed0e11adfd/511;;
    long) A="256 $E/p-long.txt"; want=c27a3b087fb5/255;;
  esac
  erun DS4_ROCM_STREAM_EVICT_POLICY=lru DS4_NOTE="d19 measure evict lru" d19-m3-$p-lru $A; gate d19-m3-$p-lru $want
  erun $S3 DS4_NOTE="d19 measure evict s3fifo" d19-m3-$p-s3 $A; gate d19-m3-$p-s3 $want
done
for p in code json; do
  if [ $p = code ]; then A="$P --nothink"; else A="$E/p-json.txt"; fi
  erun $TR DS4_ROCM_STREAM_EVICT_POLICY=lru DS4_NOTE="d19 measure DSpark old evict lru" d19-m3-dsp-$p-lru 512 $A --dspark --mtp-model $DSP
  erun $TR $S3 DS4_NOTE="d19 measure DSpark old evict s3fifo" d19-m3-dsp-$p-s3 512 $A --dspark --mtp-model $DSP
  note "DSPARK $p lru $(field d19-m3-dsp-$p-lru dspark_cycles)/$(field d19-m3-dsp-$p-lru dspark_accepted) s3 $(field d19-m3-dsp-$p-s3 dspark_cycles)/$(field d19-m3-dsp-$p-s3 dspark_accepted)"
done
R=$DS4_DIR/scripts/d19/run.sh
export D14_PAYLOAD=/data/ds4-bench/d14-payload-v2.json
DRAFT="--mtp-model $DSP --dspark"
sess() { note "start session $1"; "$R" "$@"; note "end session $1 rc=$?"; }
D14_BUDGET_MIN=50 DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_NOTE="D19 plain session to 32k, s3fifo" \
  sess d19-s1-plain-32k-s3 s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 plain session to 32k, s3fifo + seed cold" \
  sess d19-s1-plain-32k-s3cold s1 163840 "" "grow:target=32768:label=grow"
D14_BUDGET_MIN=50 DS4_DSPARK_SCHED=old DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_NOTE="D19 DSpark session to 32k, s3fifo + seed cold" \
  sess d19-s1-dsp-32k-s3cold s1 163840 "$DRAFT" "grow:target=32768:label=grow"
KV="--kv-disk-dir /data/ds4-bench/d19-kv/s3 --kv-disk-space-mb 32768"
DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-s3cold-trace-a.trace D14_BUDGET_MIN=85 \
  DS4_NOTE="D19 traced plain session 0->66k, s3fifo + seed cold, disk KV saves" \
  sess d19-s1-plain-s3cold-trace-a s1-long 163840 "$KV" "grow:target=66000:label=grow"
A1=$(ls -d /data/ds4-bench/runs/*-d19-s1-plain-s3cold-trace-a | tail -1)
DS4_ROCM_STREAM_EVICT_POLICY=s3fifo DS4_ROCM_STREAM_SEED_POLICY=cold DS4_D12_CACHE_TRACE=$T/d19-s1-plain-s3cold-trace-b.trace D14_BUDGET_MIN=88 \
  DS4_NOTE="D19 traced plain session restored at ~66k, s3fifo + seed cold, ->100k" \
  sess d19-s1-plain-s3cold-trace-b s1-long 163840 "$KV" "grow:target=100000:from=$A1:label=grow"
gzip -f $T/d19-s1-plain-s3cold-trace-*.trace
note "QUEUE DONE (q4)"
