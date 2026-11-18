#!/usr/bin/env python3
"""int4: refit D9's DSpark round cost model on int4 runs (staging cache on).

Usage: python3 refit.py TAG [TAG ...]   (DSpark runs with DS4_DSPARK_SCHED_TRACE=1)
Uses D9_ROUND lines (DS4_D9_ROUND_TRACE=1; the old scheduler prints no
DS41_SCHED lines, and D9_ROUND's miss counter covers single-row loads only, so
verify misses are not available): verify rounds (proposed, k > 0) give
verify+commit ms against rows = k + 1, i.e. directly the model scheduler's
folded form V = VA + VB x rows; plain rounds give P.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d9"))
from d9lib import parse, run_dir, lstsq, mean  # noqa: E402

V, Pl = [], []
for t in sys.argv[1:]:
    d = run_dir(t)
    if not d:
        print("missing run", t)
        continue
    for s in parse(d)["d9"]:
        if s.get("proposed") and s.get("k", 0) > 0:
            V.append((s["k"] + 1, s["verify_ms"] + s["commit_ms"], s["wait_ms"]))
        elif not s.get("proposed") and s.get("plain_ms", 0) > 1:
            Pl.append((s["plain_ms"], s["wait_ms"]))
print(f"verify rounds {len(V)}, plain steps {len(Pl)}, runs {len(sys.argv) - 1}")
print("| rows | n | verify+commit ms | read wait ms | non-read ms |")
print("|---:|---:|---:|---:|---:|")
for r in range(2, 9):
    x = [v for v in V if v[0] == r]
    if x:
        print(f"| {r} | {len(x)} | {mean(v[1] for v in x):.1f} | {mean(v[2] for v in x):.1f} | "
              f"{mean(v[1] - v[2] for v in x):.1f} |")
f = lambda c: " + ".join(f"{x:.2f}" for x in c[0]) + f" x rows (R2 {c[1]:.3f}, rmse {c[2]:.1f} ms)"
fit = lstsq([[1, v[0]] for v in V], [v[1] for v in V])
print("verify+commit ms ~ 1 + rows:", f(fit))
print("read wait ms ~ 1 + rows:    ", f(lstsq([[1, v[0]] for v in V], [v[2] for v in V])))
print("non-read ms ~ 1 + rows:     ", f(lstsq([[1, v[0]] for v in V], [v[1] - v[2] for v in V])))
P = mean(p[0] for p in Pl) if Pl else float("nan")
print(f"plain step ms (DSpark runs' plain rounds): mean {P:.1f}, n {len(Pl)}")
print(f"scheduler constants: DS4_DSPARK_MODEL_P={P:.0f} DS4_DSPARK_MODEL_VA={fit[0][0]:.0f} "
      f"DS4_DSPARK_MODEL_VB={fit[0][1]:.0f}  (D9 defaults MXFP4: P=118 VA=77 VB=49)")
