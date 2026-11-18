"""D11 Part A: predicted one-drive tok/s on the int2 binary (D10 method),
written BEFORE the one-drive runs.  Also compares with measured d11-1d-* runs
when they exist (python3 predict.py --measured)."""
import re
import sys
sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import parse, run_dir, record, mean  # noqa: E402

BW2, MB = 13.0, 18.80
CASES = (("high: reader 94 % of 6.89", 0.94 * 6.89), ("low: reader 89 %", 0.89 * 6.89))


def plain_m(tag):
    t = open(run_dir(tag) + "/run.log").read()
    return mean(float(x) for x in re.findall(r"perf token \d+ pos \d+ ms [0-9.]+ slots \d+ misses (\d+)", t))


def dsp_m(tag):
    d = run_dir(tag)
    t = open(d + "/run.log").read()
    bm = int(re.search(r"batch calls=\d+ unique=\d+ hits=\d+ misses=(\d+)", t).group(1))
    st = parse(d)["step"]
    return bm / 511.0, sum(x["misses"] for x in st) / 511.0


tp2 = mean([record(f"int2-mx-code-plain-r{i}")["decode_tps"] for i in (1, 2)])
td2 = mean([record(f"int2-mx-code-dsp-r{i}")["decode_tps"] for i in (1, 2)])
mp = mean([plain_m(f"int2-mx-code-plain-r{i}") for i in (1, 2)])
mv, ms = dsp_m("int2-mx-code-dsp-r1")
print(f"two-drive anchors: plain {tp2:.3f} tok/s, misses/token {mp:.2f}; DSpark {td2:.3f} tok/s, "
      f"verify misses/token {mv:.2f}, plain-step misses/token {ms:.2f}")
pred = {}
for name, bw1 in CASES:
    s = BW2 / bw1
    p_ms = 1000 / tp2 + 1.442 * (s - 1) * mp
    d_ms = 1000 / td2 + 1.369 * (s - 1) * mv + 1.490 * (s - 1) * ms
    pred[name] = (1000 / p_ms, 1000 / d_ms)
    print(f"PREDICTED one drive ({name}, scale {s:.3f}): plain {1000 / p_ms:.2f} tok/s, DSpark old {1000 / d_ms:.2f} tok/s")
if "--measured" in sys.argv:
    mpl = [record(f"d11-1d-plain-r{i}") for i in (1, 2)]
    mds = [record(f"d11-1d-dsp-r{i}") for i in (1, 2)]
    t1p = mean(r["decode_tps"] for r in mpl)
    t1d = mean(r["decode_tps"] for r in mds)
    print(f"MEASURED one drive: plain {t1p:.3f} ({', '.join(str(r['decode_tps']) for r in mpl)}), "
          f"DSpark {t1d:.3f} ({', '.join(str(r['decode_tps']) for r in mds)})")
    for name in pred:
        print(f"  {name}: plain pred/meas {pred[name][0] / t1p - 1:+.1%}, DSpark {pred[name][1] / t1d - 1:+.1%}")
    s94 = BW2 / (0.94 * 6.89)
    cp = (1000 / t1p - 1000 / tp2) / ((s94 - 1) * mp)
    cd = (1000 / t1d - 1000 / td2) / ((s94 - 1) * (mv + ms))
    print(f"implied per-miss cost at two drives (bandwidth part, reader 94 %): plain {cp:.3f} ms, DSpark {cd:.3f} ms "
          f"(transfer {MB / BW2:.3f}; fitted plain 1.442, verify 1.369)")
    print(f"implied extra ms per miss going 2 -> 1 drive: plain {(1000 / t1p - 1000 / tp2) / mp:.3f}, DSpark "
          f"{(1000 / t1d - 1000 / td2) / (mv + ms):.3f} (pure transfer at 94 %: {MB / (0.94 * 6.89) - MB / BW2:.3f})")
