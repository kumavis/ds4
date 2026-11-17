#!/usr/bin/env bash
# D2 model-free tests (GPU, under the lock): overlap bit-equality, D1 rows,
# MXFP4 routed MoE.  Output to /data/ds4-bench/d2-tests.log.
set -u
cd /data/ds4-d2
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run \
  'make test-mxfp4-overlap-rocm && make test-mxfp4-stream-rows-rocm && make test-mxfp4-rocm' \
  > /data/ds4-bench/d2-tests.log 2>&1
echo "TESTS_EXIT=$?" >> /data/ds4-bench/d2-tests.log
