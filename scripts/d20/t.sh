#!/usr/bin/env bash
# D20: rebuild and run the model-free WMMA test only (incremental, under the lock).
cd /data/ds4-d20w || exit 1
L=/data/ds4-bench/d20-test-wmma.log
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "scripts/d0/build-incremental.sh tests/test_d20_wmma_rocm > /data/ds4-bench/d20-test-wmma-build.log 2>&1 && ./tests/test_d20_wmma_rocm > $L 2>&1; echo WMMA_TEST_RC=\$? >> $L" < /dev/null
echo "$(date -Is) TEST wmma $(git rev-parse --short HEAD): $(tail -2 $L | tr "\n" " ")" >> /data/ds4-bench/d20-queue.log
