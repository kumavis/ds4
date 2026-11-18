#!/usr/bin/env bash
# kumanix-int4 queue 1: smoke, gates, headline lines (defaults: staging cache
# on, prefetch off, scheduler old). Gate runs double as headline repeat 1
# where they carry no logits log (prose, hash, long, Q2 code, DSpark r1).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) QUEUE 1 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
P=$E/p-code.txt
DSPF="--dspark --mtp-model $DSP"
DSPE="DS4_DSPARK_SCHED=old DS4_DSPARK_SCHED_TRACE=1 DS4_ROCM_DSPARK_PHASE_TRACE=1 DS4_D9_ROUND_TRACE=1"
erun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  ( export "${e[@]}"; run "$@" ) || exit 1; }
rounds() { echo "$(field $1 dspark_cycles)/$(field $1 dspark_accepted)"; }
# checkr TAG EXPECTED_ROUNDS/ACC
checkr() { got=$(rounds $1); r=FAIL; [ "$got" = "$2" ] && r=PASS
  echo "$(date -Is) ROUNDS $1 $got ref $2 $r" >> $LOG; [ $r = PASS ]; }
dref() { case $1 in code) echo 172/340;; prose) echo 435/77;; json) echo 207/305;;
  translate) echo 176/336;; list) echo 403/109;; q2code) echo 170/342;; esac; }

# --- first runs of the new binary
mx
run int4-smoke-mx-8 8 $P --nothink; check int4-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"
run int4-smoke-mx-64 64 $P --nothink; check int4-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"

# --- admission refusal: 5000 dynamic slots must still be refused at startup
echo "$(date -Is) start int4-gate-admit-5000 (expect refusal)" >> $LOG
"$B" int4-gate-admit-5000 8 $P --nothink --ssd-streaming-cache-experts 5000 > /dev/null 2>&1
d=$(rundir int4-gate-admit-5000); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" | tail -1)
msg=$(grep -m1 -i "ds4: .*\(needs\|refus\|exceeds\|admi\)" "$d/run.log" | cut -c1-200)
dec=$(grep -c "perf token" "$d/run.log")
if [ -n "$ex" ] && [ "$ex" != "EXIT=0" ] && [ "$dec" = 0 ]; then r=PASS; else r=FAIL; fi
echo "$(date -Is) ADMIT int4-gate-admit-5000 $ex \"$msg\" decode_lines=$dec $r dir=$d" >> $LOG
[ $r = PASS ] || stop "admission refusal gate"

# --- plain token gates (defaults)
DS4_D0_LOGITS=auto run int4-gate-mx-code 512 $P --nothink; check int4-gate-mx-code $(ref code) || stop digest
for p in prose hash long; do run int4-gate-mx-$p $(ntok $p) $(args $p); check int4-gate-mx-$p $(ref $p) || stop digest; done
q2
run int4-gate-q2-code 512 $P --nothink; check int4-gate-q2-code $(ref q2code) || stop digest
run int4-gate-q2-hash 512 $E/p-hash.txt; check int4-gate-q2-hash 63e4df5d2ff1 398 || stop digest
run int4-gate-q2-long 256 $E/p-long.txt; check int4-gate-q2-long 46aca9ac6072 255 || stop digest

# --- exact verify, bit-exact against plain (code: int4-gate-mx-code; json: its own plain ref)
mx
erun $DSPE DS4_REF_TAG=int4-gate-mx-code int4-gate-mx-code-exact 512 $P --nothink $DSPF --dspark-verify exact
DS4_D0_LOGITS=auto run int4-gate-ref-mx-json 512 $E/p-json.txt
erun $DSPE DS4_REF_TAG=int4-gate-ref-mx-json int4-gate-mx-json-exact 512 $E/p-json.txt $DSPF --dspark-verify exact
for t in int4-gate-mx-code-exact int4-gate-mx-json-exact; do
  echo "$(date -Is) EXACT $t first_div=$(rec $t | grep -o '"div_first_div":[^,]*') $(rec $t | grep -o '"div_bitexact_frac":[^,]*') $(rec $t | grep -o '"div_compared":[^,]*')" >> $LOG; done

# --- switch off = int3 path: code plain and code DSpark
erun DS4_ROCM_STREAM_STAGING_CACHE=0 int4-gate-off-mx-code 512 $P --nothink; check int4-gate-off-mx-code $(ref code) || stop digest
erun DS4_ROCM_STREAM_STAGING_CACHE=0 $DSPE int4-gate-off-mx-code-dsp 512 $P --nothink $DSPF; checkr int4-gate-off-mx-code-dsp $(dref code) || stop rounds

# --- DSpark fast verify, old scheduler, repeat 1 (gate: rounds/accepted)
for p in code prose json translate list; do
  erun $DSPE int4-mx-$p-dsp-r1 512 $(args $p) $DSPF; checkr int4-mx-$p-dsp-r1 $(dref $p) || stop rounds; done
q2
erun $DSPE int4-q2-code-dsp-r1 512 $P --nothink $DSPF; checkr int4-q2-code-dsp-r1 $(dref q2code) || stop rounds

# --- prefetch on: reclaim with D5 shelf, staging off vs on twin (four turns, two sweeps)
for m in off on; do
  echo "$(date -Is) start int4-seq-pf-$m" >> $LOG
  DS4_ROCM_D5_PREFETCH=1 "$(dirname "$0")/seq.sh" int4-seq-pf-$m $m 96
  o=/data/ds4-bench/int4-seq/int4-seq-pf-$m
  echo "$(date -Is) end int4-seq-pf-$m $(tr '\n' ' ' < $o/meta.txt) steps=$(grep -c D8_STEP $o/stderr.txt) tokmd5=$(grep -o 'D8_STEP pos=[0-9]* tok=[0-9]*' $o/stderr.txt | md5sum | cut -c1-12) shelf_swap_lines=$(grep -c 'D5 shelf buffer' $o/stderr.txt) reclaims=$(grep -c 'staging reclaimed' $o/stderr.txt)" >> $LOG
done

# --- headline repeats
mx
run int4-mx-code-plain-r1 512 $P --nothink; check int4-mx-code-plain-r1 $(ref code) || stop digest
run int4-mx-code-plain-r2 512 $P --nothink; check int4-mx-code-plain-r2 $(ref code) || stop digest
for p in prose hash long; do run int4-mx-$p-plain-r2 $(ntok $p) $(args $p); check int4-mx-$p-plain-r2 $(ref $p) || stop digest; done
for p in code prose json translate list; do
  erun $DSPE int4-mx-$p-dsp-r2 512 $(args $p) $DSPF; checkr int4-mx-$p-dsp-r2 $(dref $p) || stop rounds; done
q2
run int4-q2-code-plain-r2 512 $P --nothink; check int4-q2-code-plain-r2 $(ref q2code) || stop digest
erun $DSPE int4-q2-code-dsp-r2 512 $P --nothink $DSPF; checkr int4-q2-code-dsp-r2 $(dref q2code) || stop rounds
echo "$(date -Is) QUEUE 1 DONE" >> $LOG
