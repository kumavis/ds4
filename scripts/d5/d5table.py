"""Markdown tables from d5model.py output.  Usage: python3 d5table.py OUT.json"""
import json
import sys

res = json.load(open(sys.argv[1]))
runs = res["runs"]
print("| run | decode tok | misses/tok | fitted R ms/expert | fitted A ms | fit R² | measured ms/tok | model ms/tok | probe us P0/P1/P2 (median, incl. drains) | shelf 16: extra misses/tok |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n, r in runs.items():
    pu = r["probe_us_median"]
    print(f"| {n} | {r['decode_tokens']} | {r['misses_per_tok']:.1f} | {r['R']:.2f} | {r['A']:.2f} | {r['r2']:.3f} | "
          f"{r['measured_ms']:.1f} | {r['model_ms']:.1f} | {pu['P0']:.0f}/{pu['P1']:.0f}/{pu['P2']:.0f} | "
          f"{r.get('shelf16_extra_misses_per_tok', 0):.2f} |")
print()


def fmt(r):
    return (f"{r['recall']*100:.0f}% / {r['in_time']*100:.0f}% / {r['wasted_reads_per_tok']:.1f} / "
            f"{r['gain_pct']:+.1f}")


main = [r for r in res["table"] if r["tag"] == "main"]
names = list(runs)
print("**Main grid** (fitted R and A per run, probe 0.06 ms per probe per layer, preemption delay 0.35 ms, shelf 16).")
print("Cell: recall on misses / finished before the router / wasted speculative reads per token / modelled tok/s gain.")
print()
print("| predictor | depth | k | " + " | ".join(names) + " | mean gain % |")
print("|---|---|---|" + "---|" * len(names) + "---|")
keys = []
for r in main:
    key = (r["pred"], r["depth"], r["k"])
    if key not in keys:
        keys.append(key)
for key in keys:
    cells, gains = [], []
    for n in names:
        rr = [r for r in main if r["run"] == n and (r["pred"], r["depth"], r["k"]) == key]
        if rr:
            cells.append(fmt(rr[0]))
            gains.append(rr[0]["gain_pct"])
        else:
            cells.append("-")
    mg = sum(gains) / len(gains) if gains else 0
    print(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + f" | {mg:+.1f} |")
print()
sens = [r for r in res["table"] if r["tag"] != "main"]
if sens:
    print("**Sensitivity** (k = 2, depth 6; one parameter changed from the main grid)")
    print()
    print("| predictor | variant | " + " | ".join(names) + " | mean gain % |")
    print("|---|---|" + "---|" * len(names) + "---|")
    keys = []
    for r in sens:
        key = (r["pred"], r["tag"])
        if key not in keys:
            keys.append(key)
    for key in keys:
        cells, gains = [], []
        for n in names:
            rr = [r for r in sens if r["run"] == n and (r["pred"], r["tag"]) == key]
            if rr:
                cells.append(fmt(rr[0]))
                gains.append(rr[0]["gain_pct"])
            else:
                cells.append("-")
        mg = sum(gains) / len(gains) if gains else 0
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {mg:+.1f} |")
    print()

if res.get("pollution"):
    print("**Naive prefetch into the cache (no shelf): per decode token**")
    print()
    print("| run | predictor | k | recorded misses | demand misses | speculative reads | of which wasted | total reads |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res["pollution"]:
        print(f"| {r['run']} | {r['pred']} | {r['k']} | {r['recorded_misses']:.1f} | {r['demand_misses']:.1f} | "
              f"{r['spec_reads']:.1f} | {r['wasted']:.1f} | {r['total_reads']:.1f} |")
