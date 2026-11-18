#!/usr/bin/env bash
# kumanix-int3 step 2: the gate queue, then (only if every gate passed) the
# measurement queue. Both log to /data/ds4-bench/int3-queue.log.
set -u
cd /data/ds4-int3 || exit 1
. scripts/int3/common.sh
scripts/int3/gate.sh || { grep -q "QUEUE DONE (stopped)" $LOG || stop "gate failed"; exit 1; }
exec scripts/int3/measure.sh
