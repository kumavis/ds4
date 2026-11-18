#!/usr/bin/env bash
# First runs of the new binary: -n 8, then -n 64 (plain defaults), then -n 64
# with prefetch on (plain, DSpark) and Q2, so each kind's MemAvailable floor
# is known before the full queue. References: -n 8 b1cd778da121/7,
# -n 64 3098d05a1cef/63 (MXFP4 code --nothink).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) SMOKE QUEUE start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx; P=$E/p-code.txt
DS4_NOTE="int3 smoke plain" run int3-smoke-mx-8 8 $P --nothink
check int3-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
DS4_NOTE="int3 smoke plain" run int3-smoke-mx-64 64 $P --nothink
check int3-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"
DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="int3 smoke plain, D5 prefetch" run int3-smoke-mx-pf-64 64 $P --nothink
check int3-smoke-mx-pf-64 3098d05a1cef 63 || stop "smoke digest"
DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="int3 smoke DSpark, D5 prefetch" run int3-smoke-mx-dsp-pf-64 64 $P --nothink --dspark --mtp-model $DSP
q2
DS4_ROCM_D5_PREFETCH=1 DS4_NOTE="int3 smoke plain Q2, D5 prefetch" run int3-smoke-q2-pf-64 64 $P --nothink
echo "$(date -Is) QUEUE DONE (smoke)" >> $LOG
