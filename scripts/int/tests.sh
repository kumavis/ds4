#!/usr/bin/env bash
# Model-free GPU tests of the merged branches, under the GPU lock, after the
# strix-halo build (incremental, same flags: only the test objects compile):
#   test_mxfp4_stream_rows_rocm (D1), test_d0_verify_rows_rocm (D0),
#   test_mxfp4_overlap_rocm (D2), test_mxfp4_rocm (routed MXFP4 MoE).
# Output: /data/ds4-bench/int-tests.log, one TEST name EXIT=rc line per test.
set -u
cd /data/ds4-int
L=/data/ds4-bench/int-tests.log
: > $L
T="tests/test_mxfp4_stream_rows_rocm tests/test_d0_verify_rows_rocm tests/test_mxfp4_overlap_rocm tests/test_mxfp4_rocm"
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "
  scripts/d0/build-incremental.sh $T >> $L 2>&1; echo \"TESTBUILD EXIT=\$?\" >> $L
  for t in $T; do echo \"=== \$t\" >> $L; ./\$t >> $L 2>&1; echo \"TEST \$t EXIT=\$?\" >> $L; done" < /dev/null
echo "TESTS DONE" >> $L
