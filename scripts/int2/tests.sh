#!/usr/bin/env bash
# Model-free GPU tests of all parents, under the GPU lock, after the
# strix-halo build (incremental with the same flags: only test objects build):
#   kumanix-int: test_mxfp4_stream_rows_rocm (D1), test_d0_verify_rows_rocm
#     (D0; known EXIT=1 "D0 kernel rows: FAIL"), test_mxfp4_overlap_rocm (D2),
#     test_mxfp4_rocm (routed MXFP4 MoE)
#   D5: test_d5_prefetch_rocm, run as its make target does
#     (DS4_ROCM_D5_PREFETCH=1 ./tests/test_d5_prefetch_rocm)
#   C1: tools/c1/c1_exact.cpp, built with the flags in its header
# Output: /data/ds4-bench/int2-tests.log, one "TEST name EXIT=rc" line each.
set -u
cd /data/ds4-int2 || exit 1
L=/data/ds4-bench/int2-tests.log
: > $L
T="tests/test_mxfp4_stream_rows_rocm tests/test_d0_verify_rows_rocm tests/test_mxfp4_overlap_rocm tests/test_mxfp4_rocm tests/test_d5_prefetch_rocm"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "
  scripts/d0/build-incremental.sh $T >> $L 2>&1; echo \"TESTBUILD EXIT=\$?\" >> $L
  for t in $T; do echo \"=== \$t\" >> $L
    if [ \$t = tests/test_d5_prefetch_rocm ]; then DS4_ROCM_D5_PREFETCH=1 ./\$t >> $L 2>&1; else ./\$t >> $L 2>&1; fi
    echo \"TEST \$t EXIT=\$?\" >> $L; done
  hipcc -std=c++20 -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 -o tools/c1/c1_exact tools/c1/c1_exact.cpp >> $L 2>&1
  echo \"C1BUILD EXIT=\$?\" >> $L
  echo '=== tools/c1/c1_exact' >> $L; ./tools/c1/c1_exact >> $L 2>&1; echo \"TEST tools/c1/c1_exact EXIT=\$?\" >> $L" < /dev/null
echo "TESTS DONE" >> $L
