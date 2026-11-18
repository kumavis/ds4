#!/usr/bin/env bash
# D19 queue 1: build, model-free tests, gates for DS4_ROCM_STREAM_SEED_POLICY
# (default always vs cold): plain digests and DSpark rounds equal.
set -u
export DS4_DIR=/data/ds4-d19
LOG=/data/ds4-bench/d19-queue.log
. /data/ds4-d12/scripts/d12/common.sh
LOG=/data/ds4-bench/d19-queue.log
note() { echo "$(date -Is) $*" >> $LOG; }
cd $DS4_DIR
note "D19 build start $(git rev-parse --short HEAD)"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; note "build done rc=$rc"
[ $rc = 0 ] || { note "QUEUE DONE (build failed)"; exit 1; }
bash scripts/d19/tests.sh; note "tests: $(grep -c "EXIT=0" /data/ds4-bench/d19-tests.log) EXIT=0 lines, $(grep "EXIT=[1-9]" /data/ds4-bench/d19-tests.log | tr "\n" " ")"
P=$E/p-code.txt
COLD="DS4_ROCM_STREAM_SEED_POLICY=cold"
erun DS4_NOTE="d19 gate default" d19-gate-code-8 8 $P --nothink;      gate d19-gate-code-8 b1cd778da121/7
erun DS4_NOTE="d19 gate default" d19-gate-code-64 64 $P --nothink;    gate d19-gate-code-64 3098d05a1cef/63
erun DS4_NOTE="d19 gate default" d19-gate-code-512 512 $P --nothink;  gate d19-gate-code-512 8290541d324b/511
erun $COLD DS4_NOTE="d19 gate seed cold" d19-gate-cold-code-8 8 $P --nothink;     gate d19-gate-cold-code-8 b1cd778da121/7
erun $COLD DS4_NOTE="d19 gate seed cold" d19-gate-cold-code-64 64 $P --nothink;   gate d19-gate-cold-code-64 3098d05a1cef/63
erun $COLD DS4_NOTE="d19 gate seed cold" d19-gate-cold-code-512 512 $P --nothink; gate d19-gate-cold-code-512 8290541d324b/511
erun DS4_NOTE="d19 gate default" d19-gate-long 256 $E/p-long.txt;            gate d19-gate-long c27a3b087fb5/255
erun $COLD DS4_NOTE="d19 gate seed cold" d19-gate-cold-long 256 $E/p-long.txt; gate d19-gate-cold-long c27a3b087fb5/255
for p in json translate; do
  erun $TR DS4_NOTE="d19 gate DSpark old default" d19-gate-dsp-$p 512 $E/p-$p.txt --dspark --mtp-model $DSP
  erun $TR $COLD DS4_NOTE="d19 gate DSpark old seed cold" d19-gate-cold-dsp-$p 512 $E/p-$p.txt --dspark --mtp-model $DSP
  a="$(field d19-gate-dsp-$p dspark_cycles)/$(field d19-gate-dsp-$p dspark_accepted)"
  b="$(field d19-gate-cold-dsp-$p dspark_cycles)/$(field d19-gate-cold-dsp-$p dspark_accepted)"
  note "DSPARK $p default $a cold $b"
  [ "$a" = "$b" ] || { note "STOP: DSpark rounds differ on $p; QUEUE DONE"; exit 1; }
done
note "QUEUE DONE (q1)"
