#!/usr/bin/env bash
# kumanix-int3 step 2b: gate part e at a fixed cache size (gate2.sh), then,
# if it passes, the measurement queue.
set -u
cd /data/ds4-int3 || exit 1
. scripts/int3/common.sh
scripts/int3/gate2.sh || { grep -q "QUEUE DONE (stopped)" <(tail -1 $LOG) || stop "gate2 failed"; exit 1; }
exec scripts/int3/measure.sh
