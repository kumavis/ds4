import sys, math
sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import parse, run_dir, lstsq, mean

tags = sys.argv[1:]
V, Pl = [], []
for t in tags:
    r = parse(run_dir(t))
    for s in r["sched"]:
        if s["proposed"] and s["k"] > 0:
            V.append((s["k"] + 1, s["post_ms"], s["misses"], s["wait_ms"], s["draft_ms"]))
        elif not s["proposed"]:
            Pl.append((s["post_ms"], s["misses"], s["wait_ms"]))
print("verify rounds", len(V), "plain", len(Pl))
for r in range(2, 7):
    x = [v for v in V if v[0] == r]
    if x:
        print(f"rows={r} n={len(x):4d} post={mean(v[1] for v in x):6.1f} miss={mean(v[2] for v in x):6.1f} "
              f"wait={mean(v[3] for v in x):6.1f} nonread={mean(v[1]-v[3] for v in x):6.1f} draft={mean(v[4] for v in x):5.1f}")
if Pl:
    print(f"plain n={len(Pl)} ms={mean(p[0] for p in Pl):.1f} miss={mean(p[1] for p in Pl):.1f} "
          f"wait={mean(p[2] for p in Pl):.1f} nonread={mean(p[0]-p[2] for p in Pl):.1f}")
f = lambda c: "coef=" + ",".join(f"{x:.2f}" for x in c[0]) + f" r2={c[1]:.3f} rmse={c[2]:.1f}"
print("post ~ 1+rows        ", f(lstsq([[1, v[0]] for v in V], [v[1] for v in V])))
print("post ~ 1+rows+miss   ", f(lstsq([[1, v[0], v[2]] for v in V], [v[1] for v in V])))
print("nonread ~ 1+rows     ", f(lstsq([[1, v[0]] for v in V], [v[1]-v[3] for v in V])))
print("wait ~ 1+miss        ", f(lstsq([[1, v[2]] for v in V], [v[3] for v in V])))
print("miss ~ 1+rows        ", f(lstsq([[1, v[0]] for v in V], [v[2] for v in V])))
print("miss ~ 1+log rows    ", f(lstsq([[1, math.log(v[0])] for v in V], [v[2] for v in V])))
if Pl:
    print("plain ms ~ 1+miss    ", f(lstsq([[1, p[1]] for p in Pl], [p[0] for p in Pl])))
