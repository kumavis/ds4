#!/usr/bin/env python3
"""Per position: run vs reference logit max-abs diff, with the run record's
kind/round/row/rows and that round's accepted count. Usage: drift_trace.py RUN.bin REF.bin [N]"""
import sys
sys.path.insert(0, "/data/ds4-env")
import diverge as dv
run_p, ref_p = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
rr, fr = dv.read_log(run_p), dv.read_log(ref_p)
acc = {r["rnd"]: r["row"] for r in rr if r["kind"] == 3}
run, ref = dv.committed(rr), dv.committed(fr)
p0 = min(ref)
for p in range(p0, p0 + n):
    if p not in run or p not in ref:
        break
    a, b = run[p], ref[p]
    mx, same, ia, ib, mb = dv.cmp_rows(run_p, a, ref_p, b)
    print(f"pos={p} tok={a['tok']}/{b['tok']} kind={a['kind']} rnd={a['rnd']} row={a['row']}/{a['rows']} "
          f"acc={acc.get(a['rnd'], '-')} maxabs={mx:.3g} argmax={ia}/{ib} margin={mb:.3f}")
