#!/usr/bin/env python3
"""D12: markdown tables from analyze.py's JSON.  Usage: tables.py OUT.json"""
import json
import sys

R = json.load(open(sys.argv[1]))
ORDER = ["d12-tr-plain-code", "d12-tr-plain-prose", "d12-tr-plain-json", "d12-tr-plain-translate",
         "d12-tr-plain-list", "d12-tr-plain-long", "d12-tr-plain-code-2k", "d12-tr-plain-prose-2k",
         "d12-tr-dsp-code", "d12-tr-dsp-prose", "d12-tr-dsp-json", "d12-tr-dsp-translate",
         "d12-tr-dsp-list"]
TAGS = [t for t in ORDER if t in R] + [t for t in R if t not in ORDER]


def short(t):
    return t.replace("d12-tr-", "")


def pt(s, b):
    x = s["buckets"].get(b)
    return x["per_token"] if x else None


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def pct(a, b):
    if a is None or b is None or b == 0:
        return "–"
    return f"{100.0 * (a - b) / b:+.1f}%"


print("## Validation\n")
print("| trace | calls | lookups | hits | misses | evictions | bad hit/miss | bad victim | stray x | slots at eviction | budget | record misses (decode+batch) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for t in TAGS:
    r = R[t]
    v = r["validation"]
    print(f"| {short(t)} | {r['calls']} | {v['hits'] + v['misses']} | {v['hits']} | {v['misses']} | {v['evictions']} | "
          f"{v['bad_hitmiss']} | {v['bad_victim']} | {v['stray_evictions']} | {v['cache_size_at_eviction']} | "
          f"{v['budget']} | {r['record_decode_misses']}+{r['record_batch_misses']} (evictions {r['record_evictions']}) |")

pols = list(next(iter(R.values()))["policies"].keys())
for bucket, title in (("1-128", "warm-up, tokens 1-128"), ("129-512", "steady, tokens 129-512"),
                      ("513-2048", "long steady, tokens 513-2048")):
    tags = [t for t in TAGS if bucket in R[t]["policies"]["lru"]["buckets"]]
    if not tags:
        continue
    print(f"\n## Misses per generated token, {title} (change against simulated LRU)\n")
    print("| policy | " + " | ".join(short(t) for t in tags) + " |")
    print("|---|" + "---:|" * len(tags))
    for p in pols + ["lru-noseed", "belady-noseed"]:
        cells = []
        any_ = False
        for t in tags:
            s = R[t]["policies"].get(p)
            if not s:
                cells.append("")
                continue
            any_ = True
            a, b = pt(s, bucket), pt(R[t]["policies"]["lru"], bucket)
            cells.append(f(a) if p == "lru" else f"{f(a)} ({pct(a, b)})")
        if any_:
            print(f"| {p} | " + " | ".join(cells) + " |")
    cells = []
    for t in tags:
        a, b = pt(R[t]["runtime"], bucket), pt(R[t]["policies"]["lru"], bucket)
        cells.append(f"{f(a)} ({pct(a, b)})")
    print("| runtime (trace) | " + " | ".join(cells) + " |")
    cells = []
    for t in tags:
        a, b = pt(R[t]["first_use"], bucket), pt(R[t]["policies"]["lru"], bucket)
        cells.append(f"{f(a)} ({100 * a / b:.0f}% of LRU)" if a is not None and b else "–")
    print("| first use (floor) | " + " | ".join(cells) + " |")

print("\n## Misses by phase (whole run)\n")
print("| trace | phase | runtime | LRU sim | Belady | first use |")
print("|---|---|---:|---:|---:|---:|")
for t in TAGS:
    r = R[t]
    for ph in sorted(r["runtime"]["phase"]):
        print(f"| {short(t)} | {ph} | {r['runtime']['phase'].get(ph, 0)} | {r['policies']['lru']['phase'].get(ph, 0)} | "
              f"{r['policies']['belady']['phase'].get(ph, 0)} | {r['first_use']['phase'].get(ph, 0)} |")

print("\n## Slot curves: steady misses/token (129-512; 513-2048 for 2k traces), LRU / Belady\n")
caps = sorted({int(c) for t in TAGS for c in R[t]["curve"]})
print("| trace | runtime slots | " + " | ".join(str(c) for c in caps) + " |")
print("|---|---:|" + "---|" * len(caps))
for t in TAGS:
    r = R[t]
    b = "513-2048" if "513-2048" in r["policies"]["lru"]["buckets"] else "129-512"
    cells = []
    for c in caps:
        cv = r["curve"].get(str(c)) or r["curve"].get(c)
        cells.append(f"{f(pt(cv['lru'], b), 1)} / {f(pt(cv['belady'], b), 1)}")
    print(f"| {short(t)} ({b}) | {r['capacity']} | " + " | ".join(cells) + " |")

print("\n## Modelled decode tok/s (estimates; whole generated run)\n")
print("| trace | measured | runtime trace, modelled | LRU sim | " + " | ".join(p for p in pols if p != "lru") + " |")
print("|---|---:|---:|---:|" + "---:|" * (len(pols) - 1))
for t in TAGS:
    r = R[t]
    base = r["policies"]["lru"]["model_tps"]
    cells = [f"{f(r['policies'][p]['model_tps'])} ({pct(r['policies'][p]['model_tps'], base)})" for p in pols if p != "lru"]
    print(f"| {short(t)} | {f(r['measured_tps'])} | {f(r['runtime']['model_tps'])} | {f(base)} | " + " | ".join(cells) + " |")
