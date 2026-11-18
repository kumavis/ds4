"""D9: tune DS4_DSPARK_SCHED=model constants by offline replay on probe data.

usage: python3 tune.py headroom.json [--quant mx]
The policy's own constants (P, D, VA, VB) are one global set (pooled over
prompts); replay costs are per prompt (headroom.json "costs").  The objective
is the mean over prompts of modelled tok/s(model) / tok/s(old replay).
Also reports leave-one-prompt-out choices to show over-fitting.
"""
import itertools
import json
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
import sim  # noqa: E402
from d9lib import mean  # noqa: E402

H = json.load(open(sys.argv[1]))
Q = sys.argv[2] if len(sys.argv) > 2 else "mx"
costs = H["costs"]
PROMPTS = list(costs)
PR = {p: sim.load_probe(f"d9-probe-{Q}-{p}") for p in PROMPTS}


def mk(p, s=1.0):
    k = costs[p]
    return sim.Cost(k["P"], k["D"], k["Va"], k["Vb"], k["Pw"], k["Vwa"], k["Vwb"], s)


G = {k: mean(costs[p][k] for p in PROMPTS) for k in ("P", "D", "Va", "Vb")}
print("global constants:", {k: round(v, 1) for k, v in G.items()})
cal = H["cal"]
old = {p: sim.replay(PR[p], sim.OldPolicy(), mk(p))["tps"] for p in PROMPTS}
orc = {p: sim.oracle(PR[p], mk(p))["tps"] for p in PROMPTS}


def evaluate(kw, prompts=PROMPTS):
    res = {}
    for p in prompts:
        pol = sim.ModelPolicy(P=G["P"], D=G["D"], VA=G["Va"], VB=G["Vb"], **kw)
        res[p] = sim.replay(PR[p], pol, mk(p))
    return res


grid = dict(
    kappa=[1.0, 1.1, 1.2, 1.3, 1.45],
    cal=["raw", "fit"],
    halflife=[1.0, 2.0, 4.0, 8.0],
    gate=[-0.2, 0.0, 0.1, 0.2],
    skip=[4, 8, 16, 32],
    skip_max=[64, 256],
    predicted=[False, True],
    eta=[0.0, 0.05],
)
rows = []
for vals in itertools.product(*grid.values()):
    g = dict(zip(grid, vals))
    kw = dict(kappa=g["kappa"], halflife=g["halflife"], gate=g["gate"], skip=g["skip"],
              skip_max=g["skip_max"], predicted=g["predicted"], eta=g["eta"])
    if g["cal"] == "fit":
        kw.update(cal_scale=cal["scale"], bias=cal["bias"])
    r = evaluate(kw)
    score = {p: r[p]["tps"] / old[p] for p in PROMPTS}
    rows.append((mean(score.values()), g, kw, {p: r[p]["tps"] for p in PROMPTS}))
rows.sort(key=lambda x: -x[0])
print("top settings (mean model/old):")
for s, g, kw, t in rows[:12]:
    print(f"  {s:.4f} {g} " + " ".join(f"{p}={t[p]:.2f}" for p in PROMPTS))
best = rows[0]
print("\nold replay:   " + " ".join(f"{p}={old[p]:.2f}" for p in PROMPTS))
print("oracle:       " + " ".join(f"{p}={orc[p]:.2f}" for p in PROMPTS))
print("best model:   " + " ".join(f"{p}={best[3][p]:.2f}" for p in PROMPTS))
# robustness: leave one prompt out
print("\nleave-one-out (choose on 4 prompts, score the 5th):")
for p in PROMPTS:
    others = [q for q in PROMPTS if q != p]
    bestlo = max(rows, key=lambda x: mean(x[3][q] / old[q] for q in others))
    print(f"  held-out {p:10s}: setting {bestlo[1]} -> {bestlo[3][p]:.2f} vs old {old[p]:.2f} "
          f"(best-on-all gives {best[3][p]:.2f})")
json.dump(dict(best=best[2], best_grid=best[1], score=best[0], global_const=G,
               top=[(s, g) for s, g, _, _ in rows[:30]]), open("tune.json", "w"), indent=1)
