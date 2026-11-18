#!/usr/bin/env bash
# kumanix-int2 step 2: the gate queue, then (only if every gate passed) the
# measurement queue. Both log to /data/ds4-bench/int2-queue.log.
set -u
cd /data/ds4-int2 || exit 1
. scripts/int2/common.sh
scripts/int2/gate.sh || { grep -q "QUEUE DONE (stopped)" $LOG || stop "gate failed"; exit 1; }
exec scripts/int2/measure.sh
