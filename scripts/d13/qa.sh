#!/usr/bin/env bash
# D13 queue A: smoke runs of the new binary (switch on), the fact-finding runs
# (switch off, per-layer cache stats), and the in-process sequence gate.
set -u
. /data/ds4-d13/scripts/d13/common.sh
echo "$(date -Is) QUEUE A start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
mx
# First runs of the new binary: -n 8, then -n 64 (code; refs b1cd778da121/7, 3098d05a1cef/63)
erun $ON d13-smoke-on-8 8 $(args code)
erun $ON d13-smoke-on-64 64 $(args code)
# Facts, switch off, per-layer stats: plain short prompt, long prompt, DSpark sweep prompt
LS=DS4_ROCM_STREAM_CACHE_LAYER_STATS=1
erun $OFF $LS d13-fact-code-plain-off 512 $(args code)
erun $OFF $LS d13-fact-long-plain-off 256 $(args long)
erun $OFF $LS $DSPE d13-fact-json-dsp-off 512 $(args json) $DSPF
# Same with the switch on (lend/reclaim log lines, per-layer stats)
erun $ON $LS d13-fact-long-plain-on 256 $(args long)
erun $ON $LS $DSPE d13-fact-json-dsp-on 512 $(args json) $DSPF
# Sequence gate: prefill -> decode -> prefill -> decode -> prefill(sweep) -> decode in one process
for m in off on; do
  echo "$(date -Is) start d13-seq-$m" >> $LOG
  /data/ds4-d13/scripts/d13/seq.sh d13-seq-$m $m 96
  echo "$(date -Is) end d13-seq-$m $(tr '\n' ' ' < /data/ds4-bench/d13-seq/d13-seq-$m/meta.txt)" >> $LOG
done
echo "$(date -Is) QUEUE A DONE" >> $LOG
