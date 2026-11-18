#!/usr/bin/env python3
"""D18 markdown tables from run_d18.py's d18.json.  Usage: tables_d18.py d18.json"""
import json
import sys

R = json.load(open(sys.argv[1]))
F = ["~16k", "~32k", "~64k", "~100k"]
MS = 1.44
step = R.get("step", {}).get("buckets", {})


def g(s, f, lab):
    v = s.get(f, {}).get(lab)
    return v[0] if v and v[0] is not None else None


def f1(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


print("## Runtime: decode misses per token by fill and reply position\n")
print("| fill | decode tokens | first 16 | 17-64 | after 64 | after 16 (D14's 'rest') | all | slots |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for f in F:
    s = R["runtime"]
    n = s.get(f, {}).get("all", [None, 0])[1]
    print(f"| {f} | {n} | {f1(g(s,f,'first16'))} | {f1(g(s,f,'17-64'))} | {f1(g(s,f,'after64'))} | "
          f"{f1(g(s,f,'after16'))} | {f1(g(s,f,'all'))} | {R['causes'].get(f, {}).get('_slots')} |")

print("\n## Causes of decode misses (per token; after 16 / first 16)\n")
causes = ["first-use", "evicted-by-decode", "evicted-by-sweep-seed", "evicted-by-prompt-step", "other"]
print("| fill | " + " | ".join(causes) + " | distinct experts in reply tokens 16-79 |")
print("|---|" + "---|" * (len(causes) + 1))
for f in F:
    c = R["causes"].get(f, {})
    cells = []
    for k in causes:
        v = c.get(k)
        cells.append(f"{v[0]:.2f} / {v[1]:.2f}" if v and v[0] is not None and v[1] is not None else "0")
    print(f"| {f} | " + " | ".join(cells) + f" | {f1(c.get('_distinct_keys_reply_tokens_16_79_mean'), 0)} |")

print("\n## Recovery after the prompt: runtime misses per token by reply position (4-token bins)\n")
rec = R["recovery"]
print(" ".join(f"{k}:{v:.0f}" for k, v in list(rec.items())[:40]))

print("\n## Policies (simulated). Decode misses per token over the whole session; first 16 / after 16 per fill; other reads\n")
base = R["pol"]["lru"]
tok = sum(R["runtime"].get(f, {}).get("all", [0, 0])[1] for f in F)
bt = base["totals"]
print("| policy | decode misses/token (all fills) | change | Δ ms/token (×1.44, modelled) | first16 ~16k/32k/64k/100k | after16 ~16k/32k/64k/100k | prompt-step misses | sweep reads (experts) | seeds inserted |")
print("|---|---:|---:|---:|---|---|---:|---:|---:|")
order = ["lru", "belady", "s3fifo", "look:1", "look:4", "look:16", "look:64",
         "keep:256", "keep:512", "keep:1024", "keep:1536", "keep:2048", "keep:3072",
         "seed:0", "seed:8", "seed:16", "seed:32", "seed:64", "seed:cold",
         "keep:1024+seed:32", "keep:2048+seed:32", "keep:1024+seed:64", "keep:2048+seed:16",
         "d13", "d13+keep:1024", "d13+seed:0", "d13+seed:cold", "cap+768 (no reclaim)",
         "s3fifo+seed:0", "belady+seed:0"]


def allm(s):
    a, n = 0.0, 0
    for f in F:
        v = s.get(f, {}).get("all")
        if v and v[0] is not None:
            a += v[0] * v[1]
            n += v[1]
    return a / n if n else None


b_all = allm(base)
for p in order:
    s = R["pol"].get(p)
    if not s:
        continue
    a = allm(s)
    t = s["totals"]
    ex = s.get("extra", {})
    sr = ex.get("sweep_reads") if ex.get("seed_calls") else None
    print(f"| {p} | {f1(a,2)} | {100*(a-b_all)/b_all:+.1f}% | {MS*(a-b_all):+.1f} | "
          + "/".join(f1(g(s, f, 'first16'), 0) for f in F) + " | "
          + "/".join(f1(g(s, f, 'after16'), 1) for f in F) + f" | {t.get('pstep', 0)} | {sr if sr is not None else '–'} | {t.get('seed', 0)} |")

print("\n## Modelled decode tok/s per fill (non-read ms measured in that fill + 1.44 ms x misses; estimates)\n")
pols = ["lru", "s3fifo", "seed:cold", "keep:2048", "d13", "d13+seed:cold", "s3fifo+seed:0", "belady"]
print("| fill | non-read ms (measured) | " + " | ".join(pols) + " |")
print("|---|---:|" + "---:|" * len(pols))
for f in F:
    nr = step.get(f, {}).get("nonread")
    if nr is None:
        continue
    cells = []
    for p in pols:
        s = R["pol"].get(p)
        m = g(s, f, "all") if s else None
        cells.append(f1(1000 / (nr + MS * m), 2) if m is not None else "–")
    print(f"| {f} | {nr:.1f} | " + " | ".join(cells) + " |")

print("\n## Step time against fill (D8_STEP, all single-token steps in the server log)\n")
print("| fill | steps | ms/step | misses | read wait ms | non-read ms |")
print("|---|---:|---:|---:|---:|---:|")
for f in ["~8k"] + F + ["~128k"]:
    b = step.get(f)
    if b:
        print(f"| {f} | {b['n']} | {b['ms']:.1f} | {b['misses']:.1f} | {b['wait']:.1f} | {b['nonread']:.1f} |")
st = R.get("step", {})
if "nonread_fit_ms" in st:
    print(f"\nFit: non-read = {st['nonread_fit_ms']:.1f} + {st['nonread_per_1k_pos_ms']:.4f} ms per 1k positions; "
          f"read wait = {st['wait_intercept_ms']:.1f} + {st['wait_per_miss_ms']:.2f} ms per miss.")
print("\nValidation:", R["validation"], "slots:", R["slots"])
