#!/usr/bin/env bash
# D3 measurements, same binary: MXFP4, two drives, 512 tokens (p-long 256),
# arms plain / DSpark old scheduler (DS4_DSPARK_SCHED=old: resident table +
# 64-cycle backoff) / DSpark online scheduler, 2 reps interleaved; then the
# in-session plain-step diagnostic (DS4_DSPARK_SCHED_FORCE=plain) and a Q2
# pair (one drive).
set -u; . "$(dirname "$0")/common.sh"
export DS4_MODEL=$MX DS4_ROCM_STREAM_REPLICA=$REPLICA
export DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_DSPARK_SCHED_TRACE=1
E=/data/ds4-env
prompt() { case $1 in
  code) echo "512 $E/p-code.txt --nothink";; prose) echo "512 $E/p-prose.txt";;
  hash) echo "512 $E/p-hash.txt";; json) echo "512 $E/p-json.txt --nothink";;
  translate) echo "512 $E/p-translate.txt --nothink";; list) echo "512 $E/p-list.txt --nothink";;
  long) echo "256 $E/p-long.txt";; esac; }
arms() { pr=$1 rep=$2; read -r -a A <<< "$(prompt $pr)"
  DS4_NOTE="D3 plain $pr" run d3-mx-$pr-plain-r$rep "${A[@]}"
  DS4_DSPARK_SCHED=old DS4_NOTE="D3 DSpark old scheduler $pr" run d3-mx-$pr-old-r$rep "${A[@]}" --dspark --mtp-model $DSP
  DS4_NOTE="D3 DSpark online scheduler $pr" run d3-mx-$pr-new-r$rep "${A[@]}" --dspark --mtp-model $DSP; }
for rep in 1 2; do
  for pr in code prose hash json translate list long; do arms $pr $rep; done
done
for pr in prose code; do read -r -a A <<< "$(prompt $pr)"
  DS4_DSPARK_SCHED_FORCE=plain DS4_NOTE="D3 DSpark session, never draft (in-session plain step) $pr" \
    run d3-mx-$pr-fp-r1 "${A[@]}" --dspark --mtp-model $DSP
done
unset DS4_ROCM_STREAM_REPLICA
export DS4_MODEL=$Q2
read -r -a A <<< "$(prompt code)"
DS4_NOTE="D3 Q2 plain" run d3-q2-code-plain "${A[@]}"
DS4_NOTE="D3 Q2 DSpark online scheduler" run d3-q2-code-new "${A[@]}" --dspark --mtp-model $DSP
DS4_DSPARK_SCHED=old DS4_NOTE="D3 Q2 DSpark old scheduler" run d3-q2-code-old "${A[@]}" --dspark --mtp-model $DSP
echo "$(date -Is) QUEUE DONE" >> $LOG
