#!/usr/bin/env bash
# D20 step 1 (profile, no behaviour change): build with the routed-MoE launch
# profiler (DS4_ROCM_D20_PROF, off by default), the device-ceiling
# microbenchmark, smoke (-n 8, -n 64), the long-prompt gate with the profiler
# on (c27a3b087fb5/255), and profiled prefills of about 2,048 + 842, 1,150
# and 300 rows (-n 8). Prompt files: /data/ds4-bench/d20-p*.txt (p-long text,
# doubled or cut).
set -u
cd /data/ds4-d20 || exit 1
. scripts/d20/common.sh
echo "$(date -Is) BUILD start $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $LOG
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "make strix-halo" > build.log 2>&1 < /dev/null
rc=$?; echo "BUILD_EXIT=$rc" >> build.log; echo "$(date -Is) BUILD_EXIT=$rc" >> $LOG
[ $rc = 0 ] || stop "build failed"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "cd tools/d20 && hipcc -O3 --offload-arch=gfx1151 peak.cpp -lhipblaslt -o peak && ./peak" > /data/ds4-bench/d20-peak.log 2>&1 < /dev/null
echo "$(date -Is) peak rc=$? $(tr "\n" " " < /data/ds4-bench/d20-peak.log | cut -c1-600)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="d20 smoke" run d20-smoke-mx-8 8 $P --nothink
check d20-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="d20 smoke" run d20-smoke-mx-64 64 $P --nothink
check d20-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
DS4_ROCM_D20_PROF=1 DS4_NOTE="d20 profile: long prompt, gate" run d20-prof-long 256 $E/p-long.txt
check d20-prof-long c27a3b087fb5 255 || stop "profiler changed tokens"
for f in p2890 p1150 p300; do
  DS4_NOTE="d20 baseline prefill $f (profiler off)" run d20-base-$f 8 /data/ds4-bench/d20-$f.txt
  DS4_ROCM_D20_PROF=1 DS4_NOTE="d20 profile prefill $f" run d20-prof-$f 8 /data/ds4-bench/d20-$f.txt
done
echo "$(date -Is) QUEUE DONE (q1)" >> $LOG
