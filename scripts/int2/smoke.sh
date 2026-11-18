#!/usr/bin/env bash
# First runs of the new binary: -n 8, then -n 64 (plain defaults), then -n 64
# of the other kinds the gate uses (prefetch on, DSpark, Q2), so each kind's
# MemAvailable floor is known before the full queue. References from the
# parents: -n 8 b1cd778da121/7, -n 64 3098d05a1cef/63 (MXFP4 code --nothink).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) SMOKE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="int2 smoke plain" run int2-smoke-mx-8 8 $P --nothink
check int2-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="int2 smoke plain" run int2-smoke-mx-64 64 $P --nothink
check int2-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="int2 smoke plain, D5 prefetch" run int2-smoke-mx-pf-64 64 $P --nothink
check int2-smoke-mx-pf-64 3098d05a1cef 63 || stop "smoke digest"
DS4_NOTE="int2 smoke DSpark" run int2-smoke-mx-dsp-64 64 $P --nothink --dspark --mtp-model $DSP
q2
DS4_NOTE="int2 smoke plain Q2" run int2-smoke-q2-64 64 $P --nothink
echo "$(date -Is) QUEUE DONE (smoke)" >> $LOG
