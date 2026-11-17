#!/usr/bin/env bash
cd /data/ds4-d0
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run "scripts/d0/build-incremental.sh tests/test_d0_verify_rows_rocm" > build-inc.log 2>&1 || { echo BUILD FAIL; exit 1; }
flock /data/ds4-bench/gpu.lock nix-shell /data/ds4-env/shell.nix --run ./tests/test_d0_verify_rows_rocm > d0-ktest.log 2>&1; echo "KTEST EXIT=$?"
exec scripts/d0/q2.sh
