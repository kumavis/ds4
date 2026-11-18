#!/usr/bin/env bash
# D22 queue 1 (bench gates; after the build): smoke, model-free admission
# test, admission refusal at 6000, switch-off = int4, default plan and gates,
# forced sheds (DS4_ROCM_STREAM_MEM_GUARD_FAKE) on plain digests and DSpark
# rounds, REPL sequence gates with forced sheds (staging on, prefetch on/off).
set -u; . "$(dirname "$0")/common.sh"
echo "$(date -Is) QUEUE 1 start $(git -C $DS4_DIR rev-parse --short HEAD)" >> $LOG
P=$E/p-code.txt
DSPF="--dspark --mtp-model $DSP"
DSPE="DS4_DSPARK_SCHED=old DS4_DSPARK_SCHED_TRACE=1 DS4_ROCM_DSPARK_PHASE_TRACE=1"
erun() { local e=(); while [[ $# -gt 0 && $1 == *=* ]]; do e+=("$1"); shift; done
  ( export "${e[@]}"; run "$@" ) || exit 1; }
rounds() { echo "$(field $1 dspark_cycles)/$(field $1 dspark_accepted)"; }
checkr() { got=$(rounds $1); r=FAIL; [ "$got" = "$2" ] && r=PASS
  echo "$(date -Is) ROUNDS $1 $got ref $2 $r" >> $LOG; [ $r = PASS ]; }
dref() { case $1 in code) echo 172/340;; prose) echo 435/77;; json) echo 207/305;;
  translate) echo 176/336;; list) echo 403/109;; q2code) echo 170/342;; esac; }
guardlog() { d=$(rundir $1); echo "$(date -Is) GUARD $1 plan=\"$(grep -o 'DSpark runtime charge [^;]*' $d/run.log | head -1)\" dyn=$(slots $1) sheds=$(grep -c 'memory guard shed ' $d/run.log) noshed=$(grep -c 'memory guard could not shed' $d/run.log) guard_line=$(grep -c 'V4.1 memory guard: shed below' $d/run.log) summary=\"$(grep -o 'memory guard summary: .*' $d/run.log | tail -1)\"" >> $LOG; }

# --- model-free admission boundary test
echo "$(date -Is) start test_d22_admit" >> $LOG
( cd $DS4_DIR && nix-shell /data/ds4-env/shell.nix --run "cc -O2 -I. -o tests/test_d22_admit tests/test_d22_admit.c && ./tests/test_d22_admit" ) > /data/ds4-bench/d22-admit-test.log 2>&1
echo "$(date -Is) TEST test_d22_admit EXIT=$? $(tail -1 /data/ds4-bench/d22-admit-test.log)" >> $LOG

# --- first runs of the new binary
mx
run d22-smoke-mx-8 8 $P --nothink; check d22-smoke-mx-8 b1cd778da121 7 || stop "smoke digest"; guardlog d22-smoke-mx-8
run d22-smoke-mx-64 64 $P --nothink; check d22-smoke-mx-64 3098d05a1cef 63 || stop "smoke digest"; guardlog d22-smoke-mx-64

# --- admission refusal at 6000 dynamic slots (well past the limit, host-independent)
echo "$(date -Is) start d22-gate-admit-6000 (expect refusal)" >> $LOG
"$B" d22-gate-admit-6000 8 $P --nothink --ssd-streaming-cache-experts 6000 > /dev/null 2>&1
d=$(rundir d22-gate-admit-6000); ex=$(grep -o "EXIT=[0-9]*" "$d/run.log" | tail -1)
msg=$(grep -m1 -i "ds4: .*\(needs\|no room\|refus\|exceeds\)" "$d/run.log" | cut -c1-200); dec=$(grep -c "perf token" "$d/run.log")
if [ -n "$ex" ] && [ "$ex" != "EXIT=0" ] && [ "$dec" = 0 ]; then r=PASS; else r=FAIL; fi
echo "$(date -Is) ADMIT d22-gate-admit-6000 $ex \"$msg\" decode_lines=$dec $r" >> $LOG
[ $r = PASS ] || stop "admission refusal gate"

# --- switch off = int4 (no DSpark runtime charge, no guard): slot counts and outputs
erun DS4_ROCM_STREAM_MEM_GUARD=0 d22-gate-off-mx-code 512 $P --nothink; check d22-gate-off-mx-code $(ref code) || stop digest; guardlog d22-gate-off-mx-code
erun DS4_ROCM_STREAM_MEM_GUARD=0 $DSPE d22-gate-off-mx-code-dsp 512 $P --nothink $DSPF; checkr d22-gate-off-mx-code-dsp $(dref code) || stop rounds; guardlog d22-gate-off-mx-code-dsp

# --- defaults: plain digests, DSpark rounds (plan with the DSpark charge)
for p in code prose hash long; do run d22-gate-mx-$p $(ntok $p) $(args $p); check d22-gate-mx-$p $(ref $p) || stop digest; guardlog d22-gate-mx-$p; done
for p in code prose json translate list; do
  erun $DSPE d22-mx-$p-dsp 512 $(args $p) $DSPF; checkr d22-mx-$p-dsp $(dref $p) || stop rounds; guardlog d22-mx-$p-dsp; done

# --- forced sheds: residency changes only (poll 1 = the sync before prefill)
F=DS4_ROCM_STREAM_MEM_GUARD_FAKE
erun $F=1,120,300,450 d22-shed-mx-code 512 $P --nothink; check d22-shed-mx-code $(ref code) || stop digest; guardlog d22-shed-mx-code
erun $F=1,60,150,230 d22-shed-mx-long 256 $E/p-long.txt; check d22-shed-mx-long $(ref long) || stop digest; guardlog d22-shed-mx-long
erun $F=1,30,90,150 $DSPE d22-shed-mx-code-dsp 512 $P --nothink $DSPF; checkr d22-shed-mx-code-dsp $(dref code) || stop rounds; guardlog d22-shed-mx-code-dsp
erun $F=1,40,120,180 $DSPE d22-shed-mx-json-dsp 512 $E/p-json.txt $DSPF; checkr d22-shed-mx-json-dsp $(dref json) || stop rounds; guardlog d22-shed-mx-json-dsp

# --- REPL sequence gates (four turns, two sweeps; staging on): no-shed twin and forced sheds, prefetch on and off
for t in "seq-pf-noshed 1 -" "seq-pf-shed 1 50,150,250,350" "seq-noshed 0 -" "seq-shed 0 50,150,250,350"; do
  set -- $t; tag=d22-$1; pf=$2; fk=$3
  echo "$(date -Is) start $tag" >> $LOG
  if [ "$fk" = - ]; then DS4_ROCM_D5_PREFETCH=$pf "$(dirname "$0")/seq.sh" $tag on 96
  else DS4_ROCM_D5_PREFETCH=$pf DS4_ROCM_STREAM_MEM_GUARD_FAKE=$fk "$(dirname "$0")/seq.sh" $tag on 96; fi
  o=/data/ds4-bench/d22-seq/$tag
  echo "$(date -Is) end $tag $(tr '\n' ' ' < $o/meta.txt) steps=$(grep -c D8_STEP $o/stderr.txt) tokmd5=$(grep -o 'D8_STEP pos=[0-9]* tok=[0-9]*' $o/stderr.txt | md5sum | cut -c1-12) sheds=$(grep -c 'memory guard shed ' $o/stderr.txt) reclaims=$(grep -c 'staging reclaimed' $o/stderr.txt) shelf_swap=$(grep -c 'D5 shelf buffer' $o/stderr.txt)" >> $LOG
done
echo "$(date -Is) QUEUE 1 DONE" >> $LOG
