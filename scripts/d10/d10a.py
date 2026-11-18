"""D10 Part A: modelled DSpark / plain tok/s against aggregate read bandwidth.

All outputs are MODELLED.  Method (D10.md, Part A):
  per-miss time c (fitted: verify 1.369, DSpark-session plain step 1.490,
  plain decode 1.442 ms/miss on MXFP4) = c_fix + c_bw, where c_bw = bytes per
  expert / reader bandwidth today (18.80 MB / 13.0 GB/s = 1.446 ms MXFP4,
  9.95 MB / 13.0 GB/s = 0.765 ms Q2).  c_bw scales with 13.0 / BW_eff of the
  configuration; c_fix does not.  Low case adds a per-burst latency penalty
  per extra drive.  Everything else (non-read time, drafter, acceptance) is
  unchanged.
usage: python3 d10a.py  (reads scripts/d10/out/h_mx.json, h_q2.json)
"""
import json
import re
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
import sim  # noqa: E402
from d9lib import parse, run_dir, record, mean  # noqa: E402

OUT = "/data/ds4-d9/scripts/d10/out"
P5 = ["code", "prose", "json", "translate", "list"]
BW0 = 13.0                      # ds4 reader today, GB/s (94 % of 13.77)
MX_BYTES, Q2_BYTES = 18.80, 9.95  # MB per routed expert (17.93 / 9.49 MiB)

# name, ceiling GB/s, (high BW_eff, low BW_eff), extra drives
CONFIGS = [
    ("2 internal (today)", 13.77, (13.0, 13.0), 0),
    ("3 internal", 20.66, (0.94 * 20.66, 0.88 * 20.66), 1),
    ("3 internal + 1 USB4", 24.26, (0.94 * 24.26, 0.88 * 20.66 + 0.6 * 3.6), 2),
    ("3 internal + 2 USB4", 27.86, (0.94 * 27.86, 0.88 * 20.66 + 0.6 * 3.6), 3),
]
LAT_PER_DRIVE = 0.06            # ms per layer burst per extra drive (0.42 -> 0.54 for 1 -> 2)
LAYERS_VERIFY, LAYERS_PLAIN = 43, 25
FIX_LOW_MX = 0.15               # ms/miss assumed fixed in the low case (MXFP4)


def perf_misses(tag):
    d = run_dir(tag)
    m = re.findall(r"perf token \d+ pos \d+ ms ([0-9.]+) slots \d+ misses (\d+)", open(d + "/run.log").read())
    return mean(float(x[1]) for x in m), mean(float(x[0]) for x in m), len(m)


def batch_misses(tag):
    t = open(run_dir(tag) + "/run.log").read()
    m = re.search(r"batch calls=\d+ unique=\d+ hits=\d+ misses=(\d+)", t)
    return int(m.group(1))


def step_misses(tag):
    s = parse(run_dir(tag))["step"]
    return mean(x["misses"] for x in s) if s else float("nan")


def verify_rows(tag):
    return sum(x["k"] + 1 for x in parse(run_dir(tag))["d9"] if x["k"] > 0)


class Scaled:
    """Per-prompt cost set at a configuration."""

    def __init__(self, P, Pmiss, mu, D, s, c_fix, cv, cp, extra, low):
        cvp = c_fix + (cv - c_fix) * s
        cpp = c_fix + (cp - c_fix) * s
        lat_v = LAT_PER_DRIVE * extra * LAYERS_VERIFY if low else 0.0
        lat_p = LAT_PER_DRIVE * extra * LAYERS_PLAIN if low else 0.0
        self.P = P - cp * Pmiss + cpp * Pmiss + lat_p
        self.Va = 77.2 + lat_v
        self.Vb = 14.2 + cvp * mu
        self.D = D

    def cost(self):
        return sim.Cost(self.P, self.D, self.Va, self.Vb)


def main():
    H = json.load(open(f"{OUT}/h_mx.json"))
    HQ = json.load(open(f"{OUT}/h_q2.json"))
    PR = {p: sim.load_probe(f"d9-probe-mx-{p}") for p in P5}
    PRQ = sim.load_probe("d9-probe-q2-code")
    rows = []
    base = {}
    # ---- cache states
    cache = {}
    for p in P5:
        rv = verify_rows(f"d9-trace-old-mx-{p}")
        mu_int = batch_misses(f"int-mx-{p}-old-r1") / rv
        mu_int2 = batch_misses(f"int2-mx-{p}-dsp-r1") / rv
        st_int, st_int2 = step_misses(f"int-mx-{p}-old-r1"), step_misses(f"int2-mx-{p}-dsp-r1")
        pm = mean(PR[p]["plain_miss"])
        cache[(p, "int")] = dict(mu=mu_int, Pmiss=pm, P=H["costs"][p]["P"])
        pm2 = pm * st_int2 / st_int
        cache[(p, "int2")] = dict(mu=mu_int2, Pmiss=pm2, P=H["costs"][p]["P"] - 1.490 * (pm - pm2))
        print(f"{p:10s} rows={rv} misses/row int {mu_int:.1f} int2 {mu_int2:.1f} | plain-step misses int {st_int:.1f} "
              f"int2 {st_int2:.1f} -> probe {pm:.1f} / {pm2:.1f}")
    D = mean(H["costs"][p]["D"] for p in P5)
    cv, cp, cplain = 1.369, 1.490, 1.442
    cbw = MX_BYTES / BW0
    print(f"MXFP4: transfer per miss at 13.0 GB/s = {cbw:.3f} ms; fitted slopes verify {cv}, session plain {cp}, plain decode {cplain}")
    out = {}
    for cname in ("int", "int2"):
        for (name, ceil, bws, extra) in CONFIGS:
            for case, bw in (("high", bws[0]), ("low", bws[1])):
                s = BW0 / bw
                cfix = 0.0 if case == "high" else FIX_LOW_MX
                for p in P5:
                    c = cache[(p, cname)]
                    sc = Scaled(c["P"], c["Pmiss"], c["mu"], D, s, cfix, cv, cp, extra, case == "low")
                    cost = sc.cost()
                    allc = [Scaled(cache[(q, cname)]["P"], cache[(q, cname)]["Pmiss"], cache[(q, cname)]["mu"], D, s,
                                   cfix, cv, cp, extra, case == "low") for q in P5]
                    mp = dict(P=mean(x.P for x in allc), D=D, VA=mean(x.Va for x in allc), VB=mean(x.Vb for x in allc))
                    r_old = sim.replay(PR[p], sim.OldPolicy(), cost)["tps"]
                    r_mod = sim.replay(PR[p], sim.ModelPolicy(**mp), cost)["tps"]
                    r_orc = sim.oracle(PR[p], cost)["tps"]
                    out[(cname, name, case, p)] = dict(old=r_old, model=r_mod, oracle=r_orc, s=s, model_const=mp)
    # plain decode
    plain = {}
    for p in P5:
        m, ms, n = perf_misses(f"int-mx-{p}-plain-r1")
        plain[(p, "int")] = (m, ms)
    for p in ("code", "prose"):
        m, ms, n = perf_misses(f"int2-mx-{p}-plain-r1")
        plain[(p, "int2")] = (m, ms)
    pout = {}
    for (p, cname), (m, ms) in plain.items():
        for (name, ceil, bws, extra) in CONFIGS:
            for case, bw in (("high", bws[0]), ("low", bws[1])):
                s = BW0 / bw
                cfix = 0.0 if case == "high" else FIX_LOW_MX
                c2 = cfix + (cplain - cfix) * s
                lat = LAT_PER_DRIVE * extra * LAYERS_PLAIN if case == "low" else 0.0
                pout[(cname, name, case, p)] = 1000.0 / (ms - cplain * m + c2 * m + lat)
        print(f"plain {cname} {p}: misses/token {m:.2f}, ms/token {ms:.1f} (fit 77+1.442m = {77 + 1.442 * m:.1f})")
    # Q2 code.  Q2 has no replica: today it reads from ONE drive (reader
    # ~0.94 x 6.89 = 6.48 GB/s).  More drives need Q2 replicas (366 GB each).
    qc = HQ["costs"]["code"]
    cvq, cpq = 1.644, 1.904
    BWQ0 = 0.94 * 6.89
    cbwq = Q2_BYTES / BWQ0
    QCONF = [("1 internal (today, Q2)", (BWQ0, BWQ0), 0), ("2 internal", (13.0, 13.0), 1)] + \
            [(n, b, e + 1) for (n, c, b, e) in CONFIGS[1:]]
    qout = {}
    mq, msq, _ = perf_misses("int-q2-code-plain")
    pm = mean(PRQ["plain_miss"])
    for (name, bws, extra) in QCONF:
        for case, bw in (("high", bws[0]), ("low", bws[1])):
            s = BWQ0 / bw
            lat_on = case == "low"
            fv, fp = cvq - cbwq, cpq - cbwq     # non-bandwidth part of each slope
            cvp, cpp = fv + cbwq * s, fp + cbwq * s
            P = qc["P"] - cpq * pm + cpp * pm + (LAT_PER_DRIVE * extra * LAYERS_PLAIN if lat_on else 0)
            Va = 65.7 + (LAT_PER_DRIVE * extra * LAYERS_VERIFY if lat_on else 0)
            Vb = 26.6 + cvp * qc["mu"]
            cost = sim.Cost(P, qc["D"], Va, Vb)
            mp = dict(P=P, D=qc["D"], VA=Va, VB=Vb)
            plain_ms = msq - cpq * mq + cpp * mq + (LAT_PER_DRIVE * extra * LAYERS_PLAIN if lat_on else 0)
            qout[(name, case)] = dict(old=sim.replay(PRQ, sim.OldPolicy(), cost)["tps"],
                                      model=sim.replay(PRQ, sim.ModelPolicy(**mp), cost)["tps"],
                                      oracle=sim.oracle(PRQ, cost)["tps"], plain=1000.0 / plain_ms)
    print(f"Q2: one drive today; transfer per miss {cbwq:.3f} ms (9.95 MB / {BWQ0:.2f} GB/s) of fitted verify {cvq} / plain {cpq}; "
          f"plain misses/token {mq:.2f} ms {msq:.1f}")
    # ---- D1 validation: same binary and token trajectory on 1 vs 2 drives.
    print("\n### validation on measured 1-drive vs 2-drive runs (d1-*, binary 1341ab3/605df40, same tokens)")
    for p, two, one, kind in (("code", ["d1-mx-code-2dc-plain-r1", "d1-mx-code-2dc-plain-r2"], ["d1-mx-code-1d-plain-r1", "d1-mx-code-1d-plain-r2"], "plain"),
                              ("prose", ["d1-mx-prose-2dc-plain-r1"], ["d1-mx-prose-1d-plain-r1", "d1-mx-prose-1d-plain-r2"], "plain"),
                              ("code", ["d1-mx-code-2dc-dsp-r1", "d1-mx-code-2dc-dsp-r2"], ["d1-mx-code-1d-dsp-r1", "d1-mx-code-1d-dsp-r2"], "dspark")):
        t2 = mean(record(t)["decode_tps"] for t in two)
        t1 = mean(record(t)["decode_tps"] for t in one)
        d = run_dir(two[0])
        txt = open(d + "/run.log").read()
        if kind == "plain":
            m = mean(float(x) for x in re.findall(r"perf token \d+ pos \d+ ms [0-9.]+ slots \d+ misses (\d+)", txt))
            c = 1.442
        else:
            bm = int(re.search(r"batch calls=\d+ unique=\d+ hits=\d+ misses=(\d+)", txt).group(1))
            sm = sum(x["misses"] for x in parse(d)["step"])
            m = (bm + sm) / 511.0
            c = 1.369
        for case, bw1, cb in (("high", 0.94 * 6.89, MX_BYTES / BW0), ("1-drive at 89 %", 0.89 * 6.89, MX_BYTES / BW0)):
            s1 = BW0 / bw1
            pred_ms = 1000.0 / t2 + cb * (s1 - 1.0) * m
            print(f"  {kind} {p}: misses/token {m:.2f}; measured 2 drives {t2:.2f} -> 1 drive {t1:.2f} tok/s; "
                  f"modelled 1 drive ({case}, c_bw {cb:.3f} ms) {1000.0 / pred_ms:.2f} ({100 * ((1000.0 / pred_ms) / t1 - 1):+.1f} %); "
                  f"implied c_bw from the measured change {(1000.0 / t1 - 1000.0 / t2) / ((BW0 / (0.94 * 6.89) - 1) * m):.3f} ms")
    # ---- measured anchors
    meas = {}
    for p in P5:
        meas[("int", "old", p)] = mean([record(f"int-mx-{p}-old-r{i}")["decode_tps"] for i in (1, 2)])
        meas[("int", "model", p)] = mean([record(f"d9-mx-{p}-model-r{i}")["decode_tps"] for i in (1, 2)])
        meas[("int2", "old", p)] = mean([record(f"int2-mx-{p}-dsp-r{i}")["decode_tps"] for i in (1, 2)])
        meas[("int", "plain", p)] = mean([record(f"int-mx-{p}-plain-r{i}")["decode_tps"] for i in (1, 2)])
    for p in ("code", "prose"):
        meas[("int2", "plain", p)] = mean([record(f"int2-mx-{p}-plain-r{i}")["decode_tps"] for i in (1, 2)])
    json.dump(dict(dsp={"|".join(k): v for k, v in out.items()}, plain={"|".join(k): v for k, v in pout.items()},
                   q2={"|".join(k): v for k, v in qout.items()}, meas={"|".join(k): v for k, v in meas.items()},
                   cache={"|".join(k): v for k, v in cache.items()}), open(f"{OUT}/d10a.json", "w"), indent=1)

    # ---- tables
    def g(cname, name, case, p, pol):
        return out[(cname, name, case, p)][pol]
    print("\n## modelled tok/s and gain over 2 drives (gain range low..high); anchored = measured today x modelled gain")
    for cname in ("int", "int2"):
        print(f"\n### cache {cname}")
        print("| config | scheduler | " + " | ".join(P5) + " | mean gain |")
        print("|---|---|" + "---|" * (len(P5) + 1))
        for (name, ceil, bws, extra) in CONFIGS:
            for pol in ("old", "model", "oracle"):
                cells, gains = [], []
                for p in P5:
                    b = g(cname, CONFIGS[0][0], "high", p, pol)
                    hi, lo = g(cname, name, "high", p, pol), g(cname, name, "low", p, pol)
                    ghi, glo = hi / b - 1, lo / b - 1
                    gains.append((glo, ghi))
                    anchor = meas.get((cname, "old" if pol == "oracle" else pol, p))
                    if pol == "model" and cname == "int2":
                        anchor = None
                    a = f" [{anchor * (1 + glo):.2f}-{anchor * (1 + ghi):.2f}]" if anchor and name != CONFIGS[0][0] else (f" [{anchor:.2f}]" if anchor else "")
                    cells.append(f"{lo:.2f}-{hi:.2f} ({100 * glo:+.0f}..{100 * ghi:+.0f}%){a}")
                mg = (mean(x[0] for x in gains), mean(x[1] for x in gains))
                print(f"| {name} | {pol} | " + " | ".join(cells) + f" | {100 * mg[0]:+.1f}..{100 * mg[1]:+.1f}% |")
        print("| plain decode | | " + " | ".join(
            (f"{pout[(cname, CONFIGS[0][0], 'high', p)]:.2f}" if (cname, CONFIGS[0][0], "high", p) in pout else "-") for p in P5) + " | |")
        for (name, ceil, bws, extra) in CONFIGS[1:]:
            cells = []
            for p in P5:
                if (cname, name, "high", p) not in pout:
                    cells.append("-")
                    continue
                b = pout[(cname, CONFIGS[0][0], "high", p)]
                lo, hi = pout[(cname, name, "low", p)], pout[(cname, name, "high", p)]
                cells.append(f"{lo:.2f}-{hi:.2f} ({100 * (lo / b - 1):+.0f}..{100 * (hi / b - 1):+.0f}%)")
            print(f"| {name} | plain | " + " | ".join(cells) + " | |")
    print("\n### Q2 code (cache int; today = ONE drive)")
    print("| config | old | model | oracle | plain |")
    print("|---|---|---|---|---|")
    for (name, bws, extra) in QCONF:
        b0 = qout[(QCONF[0][0], "high")]
        cells = []
        for pol in ("old", "model", "oracle", "plain"):
            lo, hi = qout[(name, "low")][pol], qout[(name, "high")][pol]
            cells.append(f"{lo:.2f}-{hi:.2f} ({100 * (lo / b0[pol] - 1):+.0f}..{100 * (hi / b0[pol] - 1):+.0f}%)")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n### model check today (modelled vs measured)")
    for cname in ("int", "int2"):
        print(cname, " ".join(f"{p}: old {g(cname, CONFIGS[0][0], 'high', p, 'old'):.2f}/{meas[(cname, 'old', p)]:.2f}" for p in P5))
    print("plain int", " ".join(f"{p}: {pout[('int', CONFIGS[0][0], 'high', p)]:.2f}/{meas[('int', 'plain', p)]:.2f}" for p in P5))
    print("plain int2", " ".join(f"{p}: {pout[('int2', CONFIGS[0][0], 'high', p)]:.2f}/{meas[('int2', 'plain', p)]:.2f}" for p in ("code", "prose")))
    print("model constants per config (int, high):")
    for (name, ceil, bws, extra) in CONFIGS:
        mc = out[("int", name, "high", "code")]["model_const"]
        mcl = out[("int", name, "low", "code")]["model_const"]
        print(f"  {name}: P={mc['P']:.1f} VA={mc['VA']:.1f} VB={mc['VB']:.1f} (low: P={mcl['P']:.1f} VB={mcl['VB']:.1f})")


main()
