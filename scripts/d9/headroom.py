"""D9: cost-model fit, calibration fit and headroom table.

usage: python3 headroom.py [--quant mx|q2] [--json out.json]
Reads probe runs d9-probe-<quant>-<prompt>, traced old runs
d9-trace-old-<quant>-<prompt> and the D3 online runs int-mx-<prompt>-online-r*.
"""
import argparse
import json
import math
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import parse, run_dir, record, lstsq, mean, sigmoid  # noqa: E402
import sim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quant", default="mx")
ap.add_argument("--prompts", default="code,prose,json,translate,list")
ap.add_argument("--json")
ap.add_argument("--online", default="1")
args = ap.parse_args()
Q = args.quant
PROMPTS = args.prompts.split(",")


def fmt(c):
    return "coef=[" + ", ".join(f"{x:.3f}" for x in c[0]) + f"] R2={c[1]:.3f} rmse={c[2]:.1f}"


# ------------------------------------------------------------ verify samples
V = []      # (prompt, rows, post_ms, misses, wait_ms, source)
PL = []     # (prompt, ms, misses, wait)
for p in PROMPTS:
    d = run_dir(f"d9-trace-old-{Q}-{p}")
    r = parse(d) if d else {"d9": []}
    for x in r["d9"]:
        if x["proposed"] and x["k"] > 0:
            V.append((p, x["k"] + 1, x["verify_ms"] + x["commit_ms"], x["jobs"] / 3.0, x["wait_ms"], "old"))
        elif not x["proposed"]:
            PL.append((p, x["plain_ms"], x["jobs"] / 3.0, x["wait_ms"], "old"))
    if Q == "mx" and args.online == "1":
        for rep in (1, 2):
            d = run_dir(f"int-mx-{p}-online-r{rep}")
            if not d:
                continue
            for s in parse(d)["sched"]:
                if s["proposed"] and s["k"] > 0:
                    V.append((p, s["k"] + 1, s["post_ms"], s["misses"], s["wait_ms"], "online"))

out = {"quant": Q}
print(f"== cost model ({Q}); verify samples {len(V)} (old {sum(v[5]=='old' for v in V)}, "
      f"online {sum(v[5]=='online' for v in V)}); plain samples in old runs {len(PL)}")
f_full = lstsq([[1, v[1], v[3]] for v in V], [v[2] for v in V])
f_rows = lstsq([[1, v[1]] for v in V], [v[2] for v in V])
f_nr = lstsq([[1, v[1]] for v in V], [v[2] - v[4] for v in V])
f_wait = lstsq([[1, v[3]] for v in V], [v[4] for v in V])
f_miss = lstsq([[1, v[1]] for v in V], [v[3] for v in V])
f_miss0 = lstsq([[v[1]] for v in V], [v[3] for v in V])
print("verify ms ~ 1 + rows + misses :", fmt(f_full))
print("verify ms ~ 1 + rows          :", fmt(f_rows))
print("non-read ms ~ 1 + rows        :", fmt(f_nr))
print("read wait ms ~ 1 + misses     :", fmt(f_wait))
print("misses ~ 1 + rows             :", fmt(f_miss))
print("misses ~ rows (no intercept)  :", fmt(f_miss0))
by_rows = {}
for v in V:
    by_rows.setdefault(v[1], []).append(v)
print("rows   n   verify_ms  misses  misses/row  wait_ms  nonread_ms")
for r in sorted(by_rows):
    x = by_rows[r]
    print(f"{r:4d} {len(x):4d} {mean(v[2] for v in x):9.1f} {mean(v[3] for v in x):7.1f} "
          f"{mean(v[3] for v in x)/r:9.1f} {mean(v[4] for v in x):8.1f} {mean(v[2]-v[4] for v in x):9.1f}")
out["fit"] = dict(full=f_full, rows=f_rows, nonread=f_nr, wait=f_wait, miss=f_miss, miss0=f_miss0,
                  by_rows={r: dict(n=len(x), verify_ms=mean(v[2] for v in x), misses=mean(v[3] for v in x),
                                   wait_ms=mean(v[4] for v in x)) for r, x in by_rows.items()})

# ------------------------------------------------------------ probes
PR = {p: sim.load_probe(f"d9-probe-{Q}-{p}") for p in PROMPTS}
allplain = [(pr["plain_ms"][i], pr["plain_miss"][i], pr["plain_wait"][i])
            for pr in PR.values() for i in range(pr["n"])]
f_plain = lstsq([[1, x[1]] for x in allplain], [x[0] for x in allplain])
print("plain step ms ~ 1 + misses (probe runs):", fmt(f_plain))
out["fit"]["plain"] = f_plain

a, b, c = f_full[0]
w0, w1 = f_wait[0]
costs = {}
print("\n== per-prompt cost constants (ms)")
print("prompt     P(plain) Pwait  D(draft) misses/row  V(r)=Va+Vb*r   Vwait=a+b*r")
for p in PROMPTS:
    pr = PR[p]
    vv = [v for v in V if v[0] == p]
    mu = sum(v[3] for v in vv) / sum(v[1] for v in vv)
    P = mean(pr["plain_ms"])
    Pw = mean(pr["plain_wait"])
    D = mean(pr["prop_ms"])
    Va, Vb = a, b + c * mu
    Vwa, Vwb = w0, w1 * mu
    costs[p] = dict(P=P, Pw=Pw, D=D, Va=Va, Vb=Vb, Vwa=Vwa, Vwb=Vwb, mu=mu)
    print(f"{p:10s} {P:7.1f} {Pw:6.1f} {D:8.1f} {mu:10.1f}   {Va:6.1f}+{Vb:5.1f}r   {Vwa:5.1f}+{Vwb:5.1f}r")
out["costs"] = costs


def mkcost(p, s=1.0):
    k = costs[p]
    return sim.Cost(k["P"], k["D"], k["Va"], k["Vb"], k["Pw"], k["Vwa"], k["Vwb"], s)


# ------------------------------------------------------------ calibration
print("\n== acceptance by position (probe, conditional on earlier drafts accepted)")
cal_pts = []  # (i, conf, label)
for pr in PR.values():
    for j in range(pr["n"] - 6):
        if not pr["draftable"][j]:
            continue
        for i in range(sim.BLOCK):
            if pr["A"][j] < i:
                break
            cal_pts.append((i, pr["conf"][j][i], 1 if pr["A"][j] > i else 0))
for i in range(sim.BLOCK):
    pts = [x for x in cal_pts if x[0] == i]
    print(f"draft {i+1}: n={len(pts)} accept={mean(x[2] for x in pts):.3f} "
          f"pred(sigmoid conf)={mean(sigmoid(x[1]) for x in pts):.3f}")


def fit_cal(pts, iters=300, lr=0.5):
    s, bias = 1.0, [0.0] * sim.BLOCK
    n = len(pts)
    for _ in range(iters):
        gs, gb = 0.0, [0.0] * sim.BLOCK
        for i, cf, y in pts:
            e = sigmoid(s * cf + bias[i]) - y
            gs += e * cf
            gb[i] += e
        s -= lr * gs / n
        for i in range(sim.BLOCK):
            bias[i] -= lr * 5 * gb[i] / n
    ll = mean(math.log(max(1e-9, sigmoid(s * cf + bias[i]) if y else 1 - sigmoid(s * cf + bias[i])))
              for i, cf, y in pts)
    return s, bias, ll


ll0 = mean(math.log(max(1e-9, sigmoid(cf) if y else 1 - sigmoid(cf))) for i, cf, y in cal_pts)
cal_s, cal_b, ll1 = fit_cal(cal_pts)
print(f"calibration: raw logloss {-ll0:.4f}; fitted scale={cal_s:.3f} bias="
      + ",".join(f"{x:.3f}" for x in cal_b) + f" logloss {-ll1:.4f}")
out["cal"] = dict(scale=cal_s, bias=cal_b, logloss_raw=-ll0, logloss_fit=-ll1)
# acceptance-length distribution
print("A[p] distribution (share of draftable positions with A = 0..5):")
for p in PROMPTS:
    pr = PR[p]
    As = [pr["A"][j] for j in range(pr["n"]) if pr["draftable"][j]]
    print(f"  {p:10s} " + " ".join(f"{sum(1 for a in As if a == k)/len(As):.2f}" for k in range(6))
          + f"  mean {mean(As):.2f}")

# ------------------------------------------------------------ headroom
def measured(tags):
    xs = [record(t) for t in tags]
    xs = [x["decode_tps"] for x in xs if x and x.get("decode_tps")]
    return mean(xs) if xs else float("nan")


print("\n== headroom (tok/s; modelled unless 'measured')")
cols = ["meas_plain", "meas_old", "meas_online", "old", "oracle", "fixed2", "fixed3", "fixed4",
        "fixed5", "noplain_old", "noplain_oldmin1", "oracle_noplain", "oracle_wait0",
        "oracle_wait05", "old_wait0", "perfect"]
print("prompt     " + " ".join(f"{c:>9s}" for c in cols))
table = {}
for p in PROMPTS:
    pr = PR[p]
    pre = "int-mx" if Q == "mx" else "int-q2"
    row = dict(
        meas_plain=measured([f"{pre}-{p}-plain-r1", f"{pre}-{p}-plain-r2"] if Q == "mx" else [f"int-q2-{p}-plain"]),
        meas_old=measured([f"{pre}-{p}-old-r1", f"{pre}-{p}-old-r2"] if Q == "mx" else [f"int-q2-{p}-old"]),
        meas_online=measured([f"{pre}-{p}-online-r1", f"{pre}-{p}-online-r2"] if Q == "mx" else [f"int-q2-{p}-online"]),
        old=sim.replay(pr, sim.OldPolicy(), mkcost(p))["tps"],
        oracle=sim.oracle(pr, mkcost(p))["tps"],
        noplain_old=sim.replay(pr, sim.OldPolicy(backoff=False), mkcost(p))["tps"],
        noplain_oldmin1=sim.replay(pr, sim.OldPolicy(backoff=False, min_width=1), mkcost(p))["tps"],
        oracle_noplain=sim.oracle(pr, mkcost(p), allow_plain=False)["tps"],
        oracle_wait0=sim.oracle(pr, mkcost(p, 0.0))["tps"],
        oracle_wait05=sim.oracle(pr, mkcost(p, 0.5))["tps"],
        old_wait0=sim.replay(pr, sim.OldPolicy(), mkcost(p, 0.0))["tps"],
    )
    for w in (2, 3, 4, 5):
        row[f"fixed{w}"] = sim.replay(pr, sim.FixedPolicy(w), mkcost(p))["tps"]
    k = mkcost(p)
    row["perfect"] = 1000.0 * 6 / (k.D + k.V(6))
    table[p] = row
    print(f"{p:10s} " + " ".join(f"{row[c]:9.2f}" for c in cols))
avg = {c: mean(table[p][c] for p in PROMPTS) for c in cols}
print(f"{'mean':10s} " + " ".join(f"{avg[c]:9.2f}" for c in cols))
ratio = mean(table[p]["oracle"] / table[p]["old"] for p in PROMPTS)
ratio_m = mean(table[p]["oracle"] / table[p]["meas_old"] for p in PROMPTS)
print(f"oracle / old (modelled) mean ratio {ratio:.3f}; oracle / measured old {ratio_m:.3f}")
out["table"] = table
for p in PROMPTS:
    o = sim.oracle(PR[p], mkcost(p))
    r = sim.replay(PR[p], sim.OldPolicy(), mkcost(p))
    print(f"  {p:10s} oracle rounds={o['rounds']} acc={o['accepted']} widths={dict(sorted(o['widths'].items()))}"
          f" | old rounds={r['rounds']} verify={r['verify']} acc={r['accepted']} plain={r['plain']}"
          f" widths={dict(sorted(r['widths'].items()))}")
if args.json:
    json.dump(out, open(args.json, "w"), indent=1, default=str)
