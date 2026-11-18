#!/usr/bin/env python3
"""D18 analysis driver.

Usage: run_d18.py OUT_DIR TRACE_A [TRACE_B] [--server-log LOG ...] [--parts causes,policies,keep,seed,d13,bounds,step]

Writes OUT_DIR/d18.json and prints markdown tables.  Trace B (the restored
second server session) starts with an empty cache; its first turn is
reported apart (turn 0 of trace B is excluded from the fill buckets).
"""
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sess  # noqa: E402
import sim  # noqa: E402  (scripts/d12, path set by sess)

MS_PER_MISS = 1.44
args = sys.argv[1:]
out_dir = args.pop(0)
logs, parts = [], "causes,policies,keep,seed,d13,bounds,step"
traces = []
while args:
    a = args.pop(0)
    if a == "--server-log":
        logs.append(args.pop(0))
    elif a == "--parts":
        parts = args.pop(0)
    else:
        traces.append(a)
parts = set(parts.split(","))
os.makedirs(out_dir, exist_ok=True)
R = {}
TR = [sess.Trace(p) for p in traces]
for i, tr in enumerate(TR):
    print(f"trace {i}: {tr.path} calls={len(tr.src)} marks={len(tr.marks)} cap={tr.cap} "
          f"turns={1 + max(m.turn for m in tr.marks)} flushes={tr.flush_at[:4]}", file=sys.stderr)


def skip(i):
    return (0,) if i == 1 else ()


def merge_sum(sums):
    """Combine summarize() outputs of several traces by bucket (weighted)."""
    out = {}
    for name, lo, hi in sess.FILLS:
        acc = defaultdict(lambda: [0.0, 0])
        for s in sums:
            for lab, (v, n) in s.get(name, {}).items():
                if v is not None:
                    acc[lab][0] += v * n
                    acc[lab][1] += n
        out[name] = {lab: (a[0] / a[1] if a[1] else None, a[1]) for lab, a in acc.items()}
    out["totals"] = defaultdict(int)
    for s in sums:
        for k, v in s["totals"].items():
            out["totals"][k] += v
    return out


def run_policy(make, **kw):
    sums, extra = [], defaultdict(int)
    for i, tr in enumerate(TR):
        pol = make(tr)
        miss, dropped, prep = sess.replay(tr, pol, **kw)
        sums.append(sess.summarize(tr, miss, skip(i)))
        extra["seeds_dropped"] += dropped
        extra["sweep_reads"] += sum(384 - min(384, h) for h in prep) if prep else 0
        extra["seed_calls"] += len(prep)
        if hasattr(pol, "reclaimed"):
            extra["d13_reclaimed"] += pol.reclaimed
    s = merge_sum(sums)
    s["extra"] = dict(extra)
    return s


# ------------------------------------------------------------- validation
R["validation"] = [sess.validate(tr) for tr in TR]
R["slots"] = [sorted({(m.budget, m.slots) for m in tr.marks if m.slots}) for tr in TR]
runtime = merge_sum([sess.summarize(tr, [sum(1 for k, h in tr.look[ci] if not h) for ci in range(len(tr.src))], skip(i))
                     for i, tr in enumerate(TR)])
R["runtime"] = runtime

# ------------------------------------------------------------- causes
if "causes" in parts:
    cause_tot = {name: defaultdict(lambda: [0, 0]) for name, _, _ in sess.FILLS}
    recover = defaultdict(lambda: [0, 0])          # k -> misses, tokens
    ws = {name: [] for name, _, _ in sess.FILLS}    # distinct keys in reply tokens 16..79
    slots_by_fill = {name: set() for name, _, _ in sess.FILLS}
    for ti, tr in enumerate(TR):
        seen = set()
        evphase = {}
        cur_turn_keys = defaultdict(set)
        for ci in range(len(tr.src)):
            M = tr.marks[tr.mk[ci]] if tr.mk[ci] >= 0 else None
            ph = tr.phase[ci]
            xs = tr.raw_x[ci]
            for li, (k, hit) in enumerate(tr.look[ci]):
                if ph == "dec" and not hit and M is not None and not (ti == 1 and M.turn == 0):
                    if k not in seen:
                        c = "first-use"
                    else:
                        c = {"seed": "evicted-by-sweep-seed", "pstep": "evicted-by-prompt-step",
                             "dec": "evicted-by-decode"}.get(evphase.get(k), "other")
                    for name, lo, hi in sess.FILLS:
                        if lo <= M.pos <= hi:
                            cause_tot[name][c][0 if M.k >= 16 else 1] += 1
                seen.add(k)
            for _, xk in xs:
                evphase[xk] = ph
            if ph == "dec" and M is not None and 16 <= M.k < 80:
                cur_turn_keys[M.turn].update(tr.keys[ci])
        # tokens per bucket
        pm = defaultdict(int)
        for ci in range(len(tr.src)):
            pm[tr.mk[ci]] += sum(1 for k, h in tr.look[ci] if not h)
        turn_pos = {}
        for i, M in enumerate(tr.marks):
            if M.phase != "dec" or (ti == 1 and M.turn == 0):
                continue
            recover[min(M.k, 255) // 4 * 4][0] += pm[i]
            recover[min(M.k, 255) // 4 * 4][1] += 1
            turn_pos.setdefault(M.turn, M.pos)
            for name, lo, hi in sess.FILLS:
                if lo <= M.pos <= hi:
                    cause_tot[name]["_tokens"][0 if M.k >= 16 else 1] += 1
                    slots_by_fill[name].add(M.slots)
        for t, keys in cur_turn_keys.items():
            p = turn_pos.get(t)
            for name, lo, hi in sess.FILLS:
                if p is not None and lo <= p <= hi:
                    ws[name].append(len(keys))
    R["causes"] = {}
    for name, _, _ in sess.FILLS:
        ct = cause_tot[name]
        tok = ct.get("_tokens", [0, 0])
        R["causes"][name] = {c: (v[0] / tok[0] if tok[0] else None, v[1] / tok[1] if tok[1] else None)
                             for c, v in ct.items() if c != "_tokens"}
        R["causes"][name]["_tokens_after16_first16"] = tok
        R["causes"][name]["_distinct_keys_reply_tokens_16_79_mean"] = (sum(ws[name]) / len(ws[name])) if ws[name] else None
        R["causes"][name]["_slots"] = sorted(slots_by_fill[name])
    R["recovery"] = {k: v[0] / v[1] for k, v in sorted(recover.items()) if v[1]}

# ------------------------------------------------------------- policies
R.setdefault("pol", {})
if "policies" in parts:
    pass
    R["pol"]["lru"] = run_policy(lambda tr: sess.LRUC(tr.cap))
    R["pol"]["belady"] = run_policy(lambda tr: sess.Wrap(sim.Belady(tr.cap)))
    R["pol"]["s3fifo"] = run_policy(lambda tr: sess.Wrap(sim.S3FIFO(tr.cap)))
if "bounds" in parts:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d15"))
    import policies as P15  # noqa: E402

    class StubSt:
        def __init__(self, tr):
            self.calls = [(None, None, None, None, None, tr.pos[i], None, None) for i in range(len(tr.src))]
    for h in (1, 4, 16, 64):
        R["pol"][f"look:{h}"] = run_policy(lambda tr, h=h: sess.Wrap(P15.Lookahead(tr.cap, StubSt(tr), h)))
if "keep" in parts:
    for n in (256, 512, 1024, 1536, 2048, 3072):
        R["pol"][f"keep:{n}"] = run_policy(lambda tr: sess.LRUC(tr.cap), keep_n=n)
if "seed" in parts:
    for n in (0, 8, 16, 32, 64):
        R["pol"][f"seed:{n}"] = run_policy(lambda tr: sess.LRUC(tr.cap), seed_n=n)
    for n, s in ((1024, 32), (2048, 32), (1024, 64), (2048, 16)):
        R["pol"][f"keep:{n}+seed:{s}"] = run_policy(lambda tr: sess.LRUC(tr.cap), keep_n=n, seed_n=s)
if "combo" in parts:
    R["pol"]["seed:cold"] = run_policy(lambda tr: sess.LRUC(tr.cap), seed_n="cold")
    R["pol"]["s3fifo+seed:0"] = run_policy(lambda tr: sess.Wrap(sim.S3FIFO(tr.cap)), seed_n=0)
    R["pol"]["d13+seed:0"] = run_policy(lambda tr: sess.LRUC(tr.cap), d13=768, seed_n=0)
    R["pol"]["d13+seed:cold"] = run_policy(lambda tr: sess.LRUC(tr.cap), d13=768, seed_n="cold")
    R["pol"]["belady+seed:0"] = run_policy(lambda tr: sess.Wrap(sim.Belady(tr.cap)), seed_n=0)
if "d13" in parts:
    R["pol"]["d13"] = run_policy(lambda tr: sess.LRUC(tr.cap), d13=768)
    R["pol"]["d13+keep:1024"] = run_policy(lambda tr: sess.LRUC(tr.cap), d13=768, keep_n=1024)
    R["pol"]["cap+768 (no reclaim)"] = run_policy(lambda tr: sess.LRUC(tr.cap + 768))

# ------------------------------------------------------------- step time
if "step" in parts and logs:
    rx = re.compile(r"D8_STEP pos=(\d+) tok=\d+ dspark=0 ms=([\d.]+) pend_ms=[\d.]+ slots=\d+ misses=(\d+) wait_ms=([\d.]+)")
    steps = []
    for lg in logs:
        for line in open(lg, errors="replace"):
            m = rx.search(line)
            if m:
                steps.append(tuple(float(x) for x in m.groups()))
    byb = {}
    for name, lo, hi in sess.FILLS + (("~8k", 7000, 11999), ("~128k", 115000, 140000)):
        v = [s for s in steps if lo <= s[0] <= hi]
        if v:
            byb[name] = dict(n=len(v), ms=sum(s[1] for s in v) / len(v), misses=sum(s[2] for s in v) / len(v),
                             wait=sum(s[3] for s in v) / len(v), nonread=sum(s[1] - s[3] for s in v) / len(v))
    # least squares nonread = a + b*pos
    n = len(steps)
    if n > 2:
        mx = sum(s[0] for s in steps) / n
        my = sum(s[1] - s[3] for s in steps) / n
        sxx = sum((s[0] - mx) ** 2 for s in steps)
        sxy = sum((s[0] - mx) * (s[1] - s[3] - my) for s in steps)
        b = sxy / sxx
        a = my - b * mx
        # wait per miss
        mm = sum(s[2] for s in steps) / n
        mw = sum(s[3] for s in steps) / n
        b2 = sum((s[2] - mm) * (s[3] - mw) for s in steps) / sum((s[2] - mm) ** 2 for s in steps)
        R["step"] = dict(buckets=byb, n=n, nonread_fit_ms=a, nonread_per_1k_pos_ms=b * 1000,
                         wait_per_miss_ms=b2, wait_intercept_ms=mw - b2 * mm)
    else:
        R["step"] = dict(buckets=byb, n=n)

json.dump(R, open(os.path.join(out_dir, "d18.json"), "w"), indent=1, default=str)
print("written", file=sys.stderr)
