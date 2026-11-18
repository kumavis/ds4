"""D11 Part B: drafter expert reuse from DS4_D11_DRAFT_TRACE lines, and the
modelled memory options (ESTIMATES).

usage: python3 reuse.py   (reads runs d11-trace-mx-<prompt>, d11-trace-q2-code)
"""
import collections
import json
import re
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import run_dir, record, mean  # noqa: E402

P5 = ["code", "prose", "json", "translate", "list"]
NEXP, NSTAGE = 128, 3
EXPERT_MB = 18.80                 # gate+up+down of one drafter expert (MXFP4)
GIB_PER_EXPERT = 3 * 6266880 / 2**30   # 0.01751 GiB
MISS_MS = 2.9                     # one-drive read of 18.8 MB (drafter file is on the data drive)
TPS_PER_GIB = 0.12                # D4, DSpark code at the automatic size (estimate)
OUT = "/data/ds4-d11/scripts/d11/out"


def load(tag):
    props = []
    for line in open(run_dir(tag) + "/run.log", errors="replace"):
        i = line.find("D11_DRAFT v=1 ")
        if i < 0:
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", line[i:]))
        stages = []
        for s in range(NSTAGE):
            rows = [[int(x) for x in r.split(",")] for r in kv[f"s{s}"].split("|")]
            stages.append(rows)
        props.append(dict(pos=int(kv["pos"]), stages=stages))
    return props


def stats(props):
    out = {}
    for s in range(NSTAGE):
        sel = [e for p in props for r in p["stages"][s] for e in r]
        cnt = collections.Counter(sel)
        uniq_per_prop = [len({e for r in p["stages"][s] for e in r}) for p in props]
        out[s] = dict(n_sel=len(sel), distinct=len(cnt), counter=cnt, uniq_per_prop=mean(uniq_per_prop))
    return out


def coverage(counter, k, ref=None):
    """share of selections in `counter` covered by the top-k of `ref` (default: itself)."""
    ref = ref or counter
    hot = {e for e, _ in ref.most_common(k)}
    tot = sum(counter.values())
    return sum(c for e, c in counter.items() if e in hot) / tot if tot else float("nan")


def lru_sim(props, s, slots, hot=frozenset()):
    """Per proposal, each distinct needed expert of stage s is looked up once
    (a proposal loads an expert once for all its rows).  Returns (lookups,
    misses).  Hot experts always hit; the LRU holds `slots` others."""
    lru = collections.OrderedDict()
    look = miss = 0
    for p in props:
        need = []
        for r in p["stages"][s]:
            for e in r:
                if e not in need:
                    need.append(e)
        for e in need:
            look += 1
            if e in hot:
                continue
            if e in lru:
                lru.move_to_end(e)
                continue
            miss += 1
            if slots:
                lru[e] = 1
                if len(lru) > slots:
                    lru.popitem(last=False)
    return look, miss


def shared_cache_misses(props, s, window_tokens):
    """Option 1: drafter experts live in the main expert cache.  An entry
    survives if reused within `window_tokens` decode positions (estimate of
    the main LRU's residence time); misses = first uses + reuses after a
    longer gap."""
    last = {}
    look = miss = 0
    for p in props:
        need = {e for r in p["stages"][s] for e in r}
        for e in need:
            look += 1
            if e not in last or p["pos"] - last[e] > window_tokens:
                miss += 1
            last[e] = p["pos"]
    return look, miss


def main():
    tags = {p: f"d11-trace-mx-{p}" for p in P5}
    tags["q2-code"] = "d11-trace-q2-code"
    T = {k: load(v) for k, v in tags.items() if run_dir(v)}
    S = {k: stats(v) for k, v in T.items()}
    res = {"per_prompt": {}, "options": {}}
    print("## Selections and distinct experts (per stage: s0 / s1 / s2)")
    print("| run | proposals | selections per stage | distinct experts | distinct per proposal (of 15 selections) |")
    print("|---|---:|---:|---|---|")
    for k, v in T.items():
        st = S[k]
        print(f"| {k} | {len(v)} | {st[0]['n_sel']} | " + " / ".join(str(st[s]['distinct']) for s in range(NSTAGE)) +
              " | " + " / ".join(f"{st[s]['uniq_per_prop']:.2f}" for s in range(NSTAGE)) + " |")
    allc = [sum((S[p][s]["counter"] for p in P5 if p in S), collections.Counter()) for s in range(NSTAGE)]
    print(f"| all five MXFP4 prompts | {sum(len(T[p]) for p in P5 if p in T)} | {sum(allc[0].values())} | " +
          " / ".join(str(len(allc[s])) for s in range(NSTAGE)) + " | |")
    Ks = [8, 16, 32, 64, 96]
    print("\n## Coverage: share of selections covered by the K most used experts of that stage (own prompt)")
    print("| run | stage | " + " | ".join(f"K={k}" for k in Ks) + " |")
    print("|---|---|" + "---:|" * len(Ks))
    for k in list(T) + ["all"]:
        for s in range(NSTAGE):
            c = allc[s] if k == "all" else S[k][s]["counter"]
            print(f"| {k} | s{s} | " + " | ".join(f"{coverage(c, K):.3f}" for K in Ks) + " |")
    print("\n## Leave-one-out: hot set = top-K of the other four prompts, coverage on the held-out prompt (mean of stages)")
    print("| held out | " + " | ".join(f"K={k}" for k in Ks) + " | own top-K (upper bound) at K=32 |")
    print("|---|" + "---:|" * (len(Ks) + 1))
    loo = {}
    for p in P5:
        if p not in S:
            continue
        row = []
        for K in Ks:
            cov = []
            for s in range(NSTAGE):
                ref = sum((S[q][s]["counter"] for q in P5 if q != p and q in S), collections.Counter())
                cov.append(coverage(S[p][s]["counter"], K, ref))
            row.append(mean(cov))
        loo[p] = row
        own = mean(coverage(S[p][s]["counter"], 32) for s in range(NSTAGE))
        print(f"| {p} | " + " | ".join(f"{x:.3f}" for x in row) + f" | {own:.3f} |")
    if "q2-code" in S:
        row = [mean(coverage(S["q2-code"][s]["counter"], K, allc[s]) for s in range(NSTAGE)) for K in Ks]
        print("| q2-code (hot set from the 5 MXFP4 prompts) | " + " | ".join(f"{x:.3f}" for x in row) + " | |")
    Ss = [8, 16, 32, 64]
    print("\n## LRU of S slots per stage, no hot set: hit rate per expert lookup (one lookup per distinct expert per proposal)")
    print("| run | " + " | ".join(f"S={x}" for x in Ss) + " | lookups per proposal (3 stages) |")
    print("|---|" + "---:|" * (len(Ss) + 1))
    for k, v in T.items():
        hr = []
        for x in Ss:
            lk = ms = 0
            for s in range(NSTAGE):
                a, b = lru_sim(v, s, x)
                lk += a
                ms += b
            hr.append(1 - ms / lk)
        print(f"| {k} | " + " | ".join(f"{h:.3f}" for h in hr) + f" | {lk / len(v):.1f} |")
    print("\n## Clustering within one proposal (rows = the 5 drafted positions)")
    print("| run | stage | distinct experts per proposal (max 15) | share of a row's experts also used by the previous row | Jaccard with the previous proposal |")
    print("|---|---|---:|---:|---:|")
    for k, v in T.items():
        for s in range(NSTAGE):
            same = []
            jac = []
            prev = None
            for p in v:
                rows = p["stages"][s]
                for a, b in zip(rows, rows[1:]):
                    same.append(len(set(a) & set(b)) / len(b))
                cur = {e for r in rows for e in r}
                if prev is not None:
                    jac.append(len(cur & prev) / len(cur | prev))
                prev = cur
            print(f"| {k} | s{s} | {S[k][s]['uniq_per_prop']:.2f} | {mean(same):.3f} | {mean(jac):.3f} |")

    # ------------------------------------------------------------- options
    print("\n## Options against today's all-resident drafter (ESTIMATES)")
    print("baseline = measured tok/s of the trace run (drafter resident, 6.72 GiB of drafter experts).")
    print("tok/s' = N / (N / (tps + 0.12 x freed GiB) + drafter miss ms / 1000); drafter miss = 2.9 ms each.")
    print("| run | baseline | (1) via main cache, window 140 tok | (1) window 35 tok | (1b) K=32 + LRU 16 | (1b) K=48 + LRU 16 | (1b) K=64 + LRU 16 | (1c) K=32 drop rest: selections altered | (1c) K=64: altered |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in T.items():
        tag = tags[k]
        r = record(tag)
        tps = r["decode_tps"]
        N = 511.0
        cells = []
        for win in (140, 35):
            miss = sum(shared_cache_misses(v, s, win)[1] for s in range(NSTAGE))
            # entries the drafter holds in the shared cache: mean distinct experts used per window
            held_est = 0
            for s in range(NSTAGE):
                lastpos = {}
                tot = 0
                for p in v:
                    for rr in p["stages"][s]:
                        for e in rr:
                            lastpos[e] = p["pos"]
                    tot += sum(1 for e, q in lastpos.items() if p["pos"] - q <= win)
                held_est += tot / len(v)
            freed = 6.724 - held_est * GIB_PER_EXPERT
            t = N / (N / (tps + TPS_PER_GIB * freed) + miss * MISS_MS / 1000.0)
            cells.append((t, miss, freed))
        for K in (32, 48, 64):
            miss = 0
            for s in range(NSTAGE):
                hot = frozenset(e for e, _ in S[k][s]["counter"].most_common(K)) if not k.startswith("q2") else \
                    frozenset(e for e, _ in allc[s].most_common(K))
                if k in P5:   # leave-one-out hot set: chosen on the other four prompts
                    ref = sum((S[q][s]["counter"] for q in P5 if q != k and q in S), collections.Counter())
                    hot = frozenset(e for e, _ in ref.most_common(K))
                miss += lru_sim(v, s, 16, hot)[1]
            freed = (NEXP - K - 16) * NSTAGE * GIB_PER_EXPERT
            t = N / (N / (tps + TPS_PER_GIB * freed) + miss * MISS_MS / 1000.0)
            cells.append((t, miss, freed))
        alt = []
        for K in (32, 64):
            a = []
            for s in range(NSTAGE):
                ref = sum((S[q][s]["counter"] for q in P5 if q != k and q in S), collections.Counter()) if k in P5 else allc[s]
                a.append(1 - coverage(S[k][s]["counter"], K, ref))
            alt.append(mean(a))
        res["options"][k] = dict(baseline=tps, cells=cells, altered=alt)
        print(f"| {k} | {tps:.2f} | " + " | ".join(f"{t:.2f} ({100 * (t / tps - 1):+.1f} %; {m} misses, {f:.2f} GiB)" for t, m, f in cells) +
              f" | {100 * alt[0]:.1f} % | {100 * alt[1]:.1f} % |")
    json.dump(res, open(f"{OUT}/reuse.json", "w"), indent=1, default=str)


main()
