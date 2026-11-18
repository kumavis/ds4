#!/usr/bin/env bash
# D16 step 2: gates at lambda 0, then (if they pass) the sweep.
set -u
cd /data/ds4-d16 || exit 1
. scripts/d16/common.sh
scripts/d16/gate.sh || { tail -1 $LOG | grep -q "QUEUE DONE (stopped)" || stop "gate failed"; exit 1; }
exec scripts/d16/sweep.sh
