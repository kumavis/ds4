#!/usr/bin/env python3
"""D12 analysis driver: validation, policy table, first-use share, slot
curves and modelled tok/s for every trace.

Usage: analyze.py RESULTS.jsonl OUT.json TRACE.gz [TRACE.gz ...]
(the tag is the trace file name without .trace.gz; prompt_tokens, ntok and
measured decode tok/s come from the tag's record in RESULTS.jsonl)
"""
import json
import os
import sys
import time
from collections import defaultdict

import sim
import validate

POLICIES = ["lru", "belady", "lfu:8", "lfu:16", "lfu:32", "lfu:64", "lfu:128", "lfu:512",
            "lfu:2048", "lru2", "arc", "2q", "lirs", "lirs:0.1", "s3fifo", "alfu", "lru-layer-eq", "lru-layer-dem"]
CURVE = list(range(3700, 5301, 200))

# cost model (MAINTENANCE.md D4 / D9): estimates
PLAIN_FIXED, PLAIN_MISS = 77.0, 1.44
V_FIXED, V_ROW, V_MISS, DRAFT = 77.2, 14.2, 1.37, 20.2


def records(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out[r["tag"]] = r
    return out


def model_tps(st, miss):
    """Modelled decode tok/s over the generated part of the run."""
    per_mark = defaultdict(int)
    for ci, c in enumerate(st.calls):
        if c[4] in ("step", "verify"):
            per_mark[c[6]] += miss[ci]
    P = st.P
    dec = [m for m in st.marks if (m.kind == "S" and m.pos >= P) or m.kind == "V"]
    ms = tok = 0.0
    for i, m in enumerate(dec):
        if m.kind == "S":
            ms += PLAIN_FIXED + PLAIN_MISS * per_mark[m.idx]
            tok += 1
        else:
            if i + 1 >= len(dec):
                continue   # last round: advance unknown
            ms += V_FIXED + V_ROW * m.rows + V_MISS * per_mark[m.idx] + DRAFT
            tok += dec[i + 1].pos - m.pos
    return 1000.0 * tok / ms if ms else None


def bucket_tps(per_token, dspark, rows_per_token=None):
    if dspark:
        return None
    return 1000.0 / (PLAIN_FIXED + PLAIN_MISS * per_token)


def main():
    rec = records(sys.argv[1])
    out_path = sys.argv[2]
    results = {}
    for path in sys.argv[3:]:
        tag = os.path.basename(path).split(".trace")[0]
        r = rec.get(tag, {})
        P = r.get("prompt_tokens")
        t0 = time.time()
        v = validate.replay(path, verbose=False)
        st = sim.Stream(path, P)
        cap = v["cache_size_at_eviction"][0] if v["cache_size_at_eviction"] else st.capacity_runtime
        res = dict(tag=tag, prompt_tokens=P, ntok=r.get("ntok_max"),
                   measured_tps=r.get("decode_tps"), dspark=bool(r.get("dspark")),
                   record_decode_misses=r.get("decode_misses"),
                   record_batch_misses=r.get("prefill_misses"),
                   record_evictions=r.get("evictions"),
                   validation={k: v[k] for k in ("hits", "misses", "evictions", "bad_hitmiss",
                                                 "bad_victim", "stray_evictions", "budget",
                                                 "slots_max", "cache_size_at_eviction")},
                   capacity=cap, calls=len(st.calls))
        # first use
        fu = sim.first_use_flags(st)
        lru_miss = sim.replay(st, sim.make("lru", cap))
        layer_miss = [0] * sim.NL
        for ci, c in enumerate(st.calls):
            layer_miss[c[1]] += lru_miss[ci]
        pol = {}
        for name in POLICIES:
            p = sim.make(name, cap, st, layer_miss)
            miss = lru_miss if name == "lru" else sim.replay(st, p)
            s = sim.summarize(st, miss)
            s["model_tps"] = model_tps(st, miss)
            pol[name] = s
        if any(c[4] == "seed" for c in st.calls):
            miss = sim.replay(st, sim.make("lru", cap), skip_seeds=True)
            s = sim.summarize(st, miss)
            s["model_tps"] = model_tps(st, miss)
            pol["lru-noseed"] = s
            miss = sim.replay(st, sim.make("belady", cap), skip_seeds=True)
            s = sim.summarize(st, miss)
            s["model_tps"] = model_tps(st, miss)
            pol["belady-noseed"] = s
        res["policies"] = pol
        # trace's own misses by phase (runtime), for comparison with sim lru
        own = [sum(1 for k, h in c.demand() if not h) for c in __import__("d12trace").load(path)[1]]
        res["runtime"] = sim.summarize(st, own)
        res["runtime"]["model_tps"] = model_tps(st, own)
        fsum = sim.summarize(st, fu)
        res["first_use"] = fsum
        # slot curves
        curve = {}
        for c in CURVE:
            lm = sim.replay(st, sim.make("lru", c))
            bm = sim.replay(st, sim.make("belady", c))
            curve[c] = dict(lru=sim.summarize(st, lm), belady=sim.summarize(st, bm))
        res["curve"] = curve
        res["seconds"] = time.time() - t0
        results[tag] = res
        print(tag, "done", round(res["seconds"], 1), "s", file=sys.stderr)
        json.dump(results, open(out_path, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
