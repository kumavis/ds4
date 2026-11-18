#!/usr/bin/env bash
# D19 queue 2: build with DS4_ROCM_STREAM_EVICT_POLICY (lru default | s3fifo),
# tests, gates (lru = int4; s3fifo and s3fifo+cold equal to it), traced
# s3fifo runs for the simulator replay, the D5-prefetch sequence gate.
set -u
export DS4_DIR=/data/ds4-d19
LOG=/data/ds4-bench/d19-queue.log
. /data/ds4-d12/scripts/d12/common.sh
LOG=/data/ds4-bench/d19-queue.log
T=/data/ds4-bench/d19-traces; mkdir -p $T
note() { echo "$(date -Is) $*" >> $LOG; }
cd $DS4_DIR
note "D19 q2 build start $(git rev-parse --short HEAD)"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; note "build done rc=$rc"
[ $rc = 0 ] || { note "QUEUE DONE (q2 build failed)"; exit 1; }
bash scripts/d19/tests.sh; note "tests: $(grep -c "EXIT=0" /data/ds4-bench/d19-tests.log) EXIT=0 lines, $(grep "EXIT=[1-9]" /data/ds4-bench/d19-tests.log | tr "\n" " ")"
P=$E/p-code.txt
S3="DS4_ROCM_STREAM_EVICT_POLICY=s3fifo"
COLD="DS4_ROCM_STREAM_SEED_POLICY=cold"
for v in lru s3; do
  case $v in lru) X="DS4_ROCM_STREAM_EVICT_POLICY=lru";; s3) X=$S3;; esac
  erun $X DS4_NOTE="d19 gate evict $v" d19-g2-$v-code-8 8 $P --nothink;     gate d19-g2-$v-code-8 b1cd778da121/7
  erun $X DS4_NOTE="d19 gate evict $v" d19-g2-$v-code-64 64 $P --nothink;   gate d19-g2-$v-code-64 3098d05a1cef/63
  erun $X DS4_NOTE="d19 gate evict $v" d19-g2-$v-code-512 512 $P --nothink; gate d19-g2-$v-code-512 8290541d324b/511
  erun $X DS4_NOTE="d19 gate evict $v" d19-g2-$v-long 256 $E/p-long.txt;    gate d19-g2-$v-long c27a3b087fb5/255
done
erun $S3 $COLD DS4_NOTE="d19 gate s3fifo + seed cold" d19-g2-s3cold-code-512 512 $P --nothink; gate d19-g2-s3cold-code-512 8290541d324b/511
erun $S3 $COLD DS4_NOTE="d19 gate s3fifo + seed cold" d19-g2-s3cold-long 256 $E/p-long.txt;    gate d19-g2-s3cold-long c27a3b087fb5/255
for p in json translate; do
  erun $TR DS4_ROCM_STREAM_EVICT_POLICY=lru DS4_NOTE="d19 gate DSpark old, evict lru" d19-g2-lru-dsp-$p 512 $E/p-$p.txt --dspark --mtp-model $DSP
  erun $TR $S3 DS4_NOTE="d19 gate DSpark old, evict s3fifo" d19-g2-s3-dsp-$p 512 $E/p-$p.txt --dspark --mtp-model $DSP
  a="$(field d19-g2-lru-dsp-$p dspark_cycles)/$(field d19-g2-lru-dsp-$p dspark_accepted)"
  b="$(field d19-g2-s3-dsp-$p dspark_cycles)/$(field d19-g2-s3-dsp-$p dspark_accepted)"
  note "DSPARK $p lru $a s3fifo $b"
  [ "$a" = "$b" ] || { note "STOP: DSpark rounds differ on $p; QUEUE DONE"; exit 1; }
done
# traced s3fifo runs for the replay check
erun $S3 DS4_D12_CACHE_TRACE=$T/d19-s3tr-code.trace DS4_NOTE="d19 s3fifo traced" d19-s3tr-code 512 $P --nothink
gate d19-s3tr-code 8290541d324b/511
erun $TR $S3 DS4_D12_CACHE_TRACE=$T/d19-s3tr-dsp-json.trace DS4_NOTE="d19 s3fifo traced DSpark json" d19-s3tr-dsp-json 512 $E/p-json.txt --dspark --mtp-model $DSP
note "DSPARK json traced s3fifo $(field d19-s3tr-dsp-json dspark_cycles)/$(field d19-s3tr-dsp-json dspark_accepted)"
# sequence gate (4 turns, two sweeps, D13 reclaims): lru twin, s3fifo traced, s3fifo with D5 prefetch
seqrun() { local tag=$1; shift; note "start $tag"; env "$@" scripts/d19/seq.sh $tag on 96
  o=/data/ds4-bench/d19-seq/$tag
  note "end $tag $(tr "\n" " " < $o/meta.txt) tokmd5=$(grep -o "D8_STEP pos=[0-9]* tok=[0-9]*" $o/stderr.txt | md5sum | cut -c1-12) reclaims=$(grep -c "staging reclaimed" $o/stderr.txt)"; }
seqrun d19-seq-lru DS4_ROCM_STREAM_EVICT_POLICY=lru
seqrun d19-seq-s3tr $S3 DS4_D12_CACHE_TRACE=$T/d19-seq-s3tr.trace
seqrun d19-seq-pf-lru DS4_ROCM_D5_PREFETCH=1 DS4_ROCM_STREAM_EVICT_POLICY=lru
seqrun d19-seq-pf-s3 DS4_ROCM_D5_PREFETCH=1 $S3
gzip -f $T/*.trace
note "QUEUE DONE (q2)"
