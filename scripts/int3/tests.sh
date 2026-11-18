#!/usr/bin/env bash
# Model-free GPU tests of all parents, under the GPU lock, after the
# strix-halo build (incremental with the same flags: only test objects build):
#   kumanix-int: test_mxfp4_stream_rows_rocm (D1), test_d0_verify_rows_rocm
#     (D0; known EXIT=1 "D0 kernel rows: FAIL"), test_mxfp4_overlap_rocm (D2),
#     test_mxfp4_rocm (routed MXFP4 MoE)
#   D5: test_d5_prefetch_rocm, run as its make target does
#     (DS4_ROCM_D5_PREFETCH=1 ./tests/test_d5_prefetch_rocm; it includes the
#     2-row batch union loads), three times: thermal limit default (84),
#     DS4_ROCM_D5_TEMP_LIMIT_C=0 (no limit), =1 (every speculative issue held)
#   C1: tools/c1/c1_exact.cpp, built with the flags in its header
# Output: /data/ds4-bench/int3-tests.log, one "TEST name EXIT=rc" line each.
set -u
cd /data/ds4-int3 || exit 1
L=/data/ds4-bench/int3-tests.log
: > $L
T="tests/test_mxfp4_stream_rows_rocm tests/test_d0_verify_rows_rocm tests/test_mxfp4_overlap_rocm tests/test_mxfp4_rocm tests/test_d5_prefetch_rocm"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "
  scripts/d0/build-incremental.sh $T >> $L 2>&1; echo \"TESTBUILD EXIT=\$?\" >> $L
  for t in $T; do echo \"=== \$t\" >> $L
    if [ \$t = tests/test_d5_prefetch_rocm ]; then
      for lim in default 0 1; do echo \"--- DS4_ROCM_D5_TEMP_LIMIT_C=\$lim\" >> $L
        if [ \$lim = default ]; then DS4_ROCM_D5_PREFETCH=1 ./\$t >> $L 2>&1; else DS4_ROCM_D5_TEMP_LIMIT_C=\$lim DS4_ROCM_D5_PREFETCH=1 ./\$t >> $L 2>&1; fi
        echo \"TEST \$t limit=\$lim EXIT=\$?\" >> $L; done
    else ./\$t >> $L 2>&1; echo \"TEST \$t EXIT=\$?\" >> $L; fi; done
  hipcc -std=c++20 -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 -o tools/c1/c1_exact tools/c1/c1_exact.cpp >> $L 2>&1
  echo \"C1BUILD EXIT=\$?\" >> $L
  echo '=== tools/c1/c1_exact' >> $L; ./tools/c1/c1_exact >> $L 2>&1; echo \"TEST tools/c1/c1_exact EXIT=\$?\" >> $L" < /dev/null
echo "TESTS DONE" >> $L
