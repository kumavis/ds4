#!/usr/bin/env python3
"""D4 part 1 analysis: decode misses/token and ms/token against dynamic
expert-cache slots (d4-c-* runs). Plain: misses from the D8_STEP decode
steps. DSpark: (selected + batch misses - prompt misses) / generated tokens,
the prompt's misses taken from the plain code fit at the same slot count
(plain prefill misses = decode_misses - D8_STEP decode misses).
Fits: least squares, linear in N; log (m = a - b ln N) and power (m = A N^-k)
for the extrapolation, which is labelled as such."""
import math, re, sys
sys.path.insert(0, "/data/ds4-d4/scripts/d4")
from runinfo import rows

SLOT_GIB = None
def lsq(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx; a = my - b*mx
    res = [y-(a+b*x) for x, y in zip(xs, ys)]
    syy = sum((y-my)**2 for y in ys)
    r2 = 1 - sum(e*e for e in res)/syy if syy else 1.0
    return a, b, r2

pts = {"pc": [], "pp": [], "dc": []}
prompt_m = []   # (N, prompt misses) from plain code runs
slot_mib = 17.93
for r, i in rows(r"^d4-c-(pc|pp|dc)-"):
    kind = r["tag"].split("-")[2]
    N = i["res"] or i["dyn"]
    if i["dyn"] and i["res"] and i["res"] != i["dyn"]:
        print(f"note: {r['tag']} resident peak {i['res']} != configured {i['dyn']}")
    tps = r["decode_tps"]; ms = 1000.0/tps
    if kind != "dc":
        mpt = i["step_misses"]/i["steps"]
        if kind == "pc": prompt_m.append((N, r["decode_misses"] - i["step_misses"]))
    else:
        mpt = None
        tot = (r["decode_misses"] or 0) + (r["prefill_misses"] or 0)
    pts[kind].append({"tag": r["tag"], "N": N, "ms": ms, "tps": tps, "m": mpt,
                      "avail": r["avail_min_mb"], "gtt": r["gtt_max_mb"],
                      "tot": None if kind != "dc" else tot, "ntok": r["ntok_max"],
                      "steps": i["steps"], "msl": i["ms_late"]/i["steps_late"] if i["steps_late"] else None,
                      "ml": i["misses_late"]/i["steps_late"] if i["steps_late"] else None})
pa, pb, _ = lsq([p[0] for p in prompt_m], [p[1] for p in prompt_m]) if len(prompt_m) > 1 else (0, 0, 0)
for p in pts["dc"]:
    p["m"] = (p["tot"] - (pa + pb*p["N"])) / p["ntok"]

per_gib = 1024.0/slot_mib
names = {"pc": "MXFP4 plain code", "pp": "MXFP4 plain prose", "dc": "MXFP4 DSpark code (sched old)"}
print(f"prompt (94-token code prefill) misses vs N: {pa:.0f} {pb:+.3f}*N")
for k in ("pc", "pp", "dc"):
    P = sorted(pts[k], key=lambda p: p["N"])
    if len(P) < 2: continue
    print(f"\n## {names[k]}")
    print("| tag | slots | GiB dyn | decode tok/s | ms/tok | misses/tok | avail_min_mb | gtt_max_mb |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in P:
        print(f'| {p["tag"]} | {p["N"]} | {p["N"]*slot_mib/1024:.2f} | {p["tps"]:.2f} | {p["ms"]:.1f} | {p["m"]:.2f} | {p["avail"]} | {p["gtt"]} |')
    xs = [p["N"] for p in P]
    a, b, r2 = lsq(xs, [p["m"] for p in P])
    c, d, r2t = lsq(xs, [p["ms"] for p in P])
    e, f, r2m = lsq([p["m"] for p in P], [p["ms"] for p in P])
    av, bv, r2v = lsq(xs, [p["avail"] for p in P])
    print(f"misses/tok = {a:.2f} {b*1000:+.3f}e-3*N (R2 {r2:.3f}); per GiB {b*per_gib:+.2f} misses/tok")
    print(f"ms/tok     = {c:.1f} {d*1000:+.3f}e-3*N (R2 {r2t:.3f}); per GiB {d*per_gib:+.2f} ms/tok")
    print(f"ms/tok     = {e:.1f} {f:+.3f}*misses/tok (R2 {r2m:.3f})")
    print(f"avail_min  = {av:.0f} {bv:+.2f}*N MiB (R2 {r2v:.3f}; slot {slot_mib} MiB)")
    # log and power fits for extrapolation
    la, lb, lr2 = lsq([math.log(x) for x in xs], [p["m"] for p in P])
    ka, kb, kr2 = lsq([math.log(x) for x in xs], [math.log(p["m"]) for p in P])
    auto = {"pc": 4179, "pp": 4179, "dc": 3719}[k]
    ms_auto = c + d*auto
    print(f"log fit m = {la:.1f} {lb:+.2f} ln N (R2 {lr2:.3f}); power fit m = {math.exp(ka):.3g} N^{kb:.3f} (R2 {kr2:.3f})")
    print("| EXTRAPOLATION: dyn GiB above auto | slots | misses/tok linear / log / power | ms/tok via ms-vs-misses (linear / log / power) | tok/s (linear / log / power) |")
    print("|---|---:|---|---|---|")
    for g in (0, 4, 8, 16, 32, 64):
        n = auto + g*per_gib
        ml = max(0.0, a + b*n); mlg = max(0.0, la + lb*math.log(n)); mp = math.exp(ka + kb*math.log(n))
        if n >= 15360: ml = mlg = mp = 0.0
        ms3 = [e + f*m for m in (ml, mlg, mp)]
        print(f"| +{g} | {n:.0f} | {ml:.1f} / {mlg:.1f} / {mp:.1f} | {ms3[0]:.1f} / {ms3[1]:.1f} / {ms3[2]:.1f} | {1000/ms3[0]:.2f} / {1000/ms3[1]:.2f} / {1000/ms3[2]:.2f} |")
