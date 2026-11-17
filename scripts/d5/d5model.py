"""D5 stage 1: evaluate next-layer expert predictors on D5 traces and model
the decode time with speculative prefetch.  Stdlib only.

Usage: python3 d5model.py OUT.json NAME:TRACE:RUNLOG [NAME:TRACE:RUNLOG ...]

Model (see D5.md, "Model"):
 * The cache trajectory is the recorded one.  Prefetched experts land in a
   private shelf of SHELF slots and are adopted on demand by a pointer swap,
   so hits, misses and evictions are exactly as today; prefetch moves read
   time only.  (Shelf memory comes out of the cache; its miss cost is reported
   separately by replaying the LRU with SHELF fewer slots.)
 * Per layer: A ms of compute before the router (attention etc.), then the
   router; demand reads of the layer's misses start at the router and take R
   ms each on the drive; the layer ends at max(router + M, reads_done + TAIL)
   (D2: resident experts' work overlaps the reads).  Per token +H ms.
 * One drive server (both drives striped).  Demand reads preempt speculative
   ones (chunk granularity, no penalty).  Speculative reads run only while the
   drive has no demand work, FIFO by issue time, at most SHELF started and
   unconsumed at once.  At the target layer's router: a needed prefetch that
   finished is free, a partial one finishes as demand, an unstarted one becomes
   demand; unneeded ones are dropped (work already done is waste).
 * R and the per-token constant are fitted to the measured per-token ms of the
   same run (no prefetch); A/M/TAIL/H are set from the D2 sizing.
"""
import collections
import json
import statistics
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from d5trace import LRU, N_LAYER, load_run  # noqa: E402

M = 0.5        # MoE compute after the router (resident experts, quantize, down)
TAIL = 0.15    # compute after the last expert lands (its gate/up + down)
H = 3.0        # per-token overhead outside the layers (head, sampling, host)
W_REC = 8      # recency window (tokens)


# ---------------------------------------------------------------- predictors
class History:
    def __init__(self):
        self.rec = [collections.deque(maxlen=W_REC) for _ in range(N_LAYER)]
        self.freq = [collections.Counter() for _ in range(N_LAYER)]
        self.cond = [collections.defaultdict(collections.Counter) for _ in range(N_LAYER)]
        self.cnt = [collections.Counter() for _ in range(N_LAYER)]

    def add_token(self, tok):
        for s in tok:
            j = s.layer
            self.rec[j].append(list(s.ids))
            self.freq[j].update(s.ids)
            self.cnt[j].update(s.ids)
            if j > 0:
                prev = tok[j - 1].ids
                for e in prev:
                    self.cond[j][e].update(s.ids)


def rank_rec(h, j, **_):
    out, seen = [], set()
    for ids in reversed(h.rec[j]):
        for e in ids:
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out


def rank_freq(h, j, **_):
    return [e for e, _ in h.freq[j].most_common(64)]


def rank_cond(h, j, prev_ids, off=None, **_):
    if j == 0:
        return []
    score = collections.Counter()
    for e in prev_ids:
        for tab, cnt in ((h.cond[j], h.cnt[j - 1]),) + (((off.cond[j], off.cnt[j - 1]),) if off else ()):
            c = cnt[e]
            if c:
                for f, n in tab[e].items():
                    score[f] += n / c
    fr = h.freq[j] + (off.freq[j] if off else collections.Counter())
    return sorted(score, key=lambda f: (-score[f], -fr[f], f))[:64]


def interleave(*lists):
    out, seen = [], set()
    for row in zip(*[l + [None] * (64 - len(l)) for l in lists]):
        for e in row:
            if e is not None and e not in seen:
                seen.add(e)
                out.append(e)
    return out


def probe(tag):
    return lambda h, j, s, **kw: s.probes.get(tag, (0, []))[1]


# Each predictor: list of sources (when, rank function); a source is issued at
# its time for its target layer.  when: 'token' = token start; 'P0' = start of
# layer j-1 (+probe); 'P1' = layer j-1's FFN input, just before its router
# (+probe); 'cond' = layer j-1's router; 'P2' = start of layer j (+probe).
# rank function None = oracle (exactly the layer's recorded misses).
PREDICTORS = {
    "rec": [("token", lambda h, j, s, **kw: rank_rec(h, j))],
    "freq": [("token", lambda h, j, s, **kw: rank_freq(h, j))],
    "cond": [("cond", lambda h, j, s, prev=None, **kw: rank_cond(h, j, prev))],
    "cond_off": [("cond", lambda h, j, s, prev=None, off=None, **kw: rank_cond(h, j, prev, off))],
    "P0": [("P0", probe("P0"))],
    "P1": [("P1", probe("P1"))],
    "P2": [("P2", probe("P2"))],
    "P1|cond_off": [("cond", lambda h, j, s, prev=None, off=None, **kw:
                     interleave(s.probes.get("P1", (0, []))[1], rank_cond(h, j, prev, off)))],
    "P0+P2": [("P0", probe("P0")), ("P2", probe("P2"))],
    "P1+P2": [("P1", probe("P1")), ("P2", probe("P2"))],
    "oracle@token": [("token", None)],
    "oracle@P0": [("P0", None)],
    "oracle@P1": [("P1", None)],
    "oracle@P2": [("P2", None)],
}
N_PROBES = {"P0": 1, "P1": 1, "P2": 1, "P1|cond_off": 1, "P0+P2": 2, "P1+P2": 2}


# ---------------------------------------------------------------- timeline
class Item:
    __slots__ = ("layer", "e", "issue", "rem", "started", "released")

    def __init__(self, layer, e, issue, r):
        self.layer, self.e, self.issue, self.rem = layer, e, issue, r
        self.started = False
        self.released = False


def simulate(run, pred, k, R, A, shelf, probe_ms, off=None, depth=16, pre=0.0, rate=1.0,
             adopt_block=False):
    """Returns per-decode-token modelled ms and counters.  k = speculative
    reads per layer and source (the first k non-resident candidates among the
    first `depth` of the source's ranking); pre = delay of a layer's demand
    reads when speculative reads are in flight at its router (draining the
    read pool's in-flight speculative chunks); rate = speculative read speed
    relative to demand reads (a cap on in-flight speculative chunks lowers
    their queue depth); adopt_block = a needed prefetch still in flight at
    the router is finished first and the host waits for it before launching
    the layer's resident experts (the simple adoption of stage 2)."""
    toks, first_dec = run["toks"], run["first_dec"]
    budget = run["budget"]
    lru = LRU(budget[0] if budget else 10 ** 9, run["snap"])
    h = History()
    sources = PREDICTORS[pred] if pred else []
    whens = {w for w, _ in sources}
    c_probe = probe_ms * N_PROBES.get(pred, 0)   # per layer, GPU compute added
    ms_tok = []
    cnt = collections.Counter()
    for ti, tok in enumerate(toks):
        decode = ti >= first_dec and pred is not None
        if ti < first_dec:
            for s in tok:
                lru.access(s.layer, s.ids)
            h.add_token(tok)
            continue
        t = 0.0
        t_drive = 0.0
        queue = []                      # FIFO by issue time
        queued = set()

        def issue(j, at, when):
            s = tok[j]
            for w, fn in sources:
                if w != when:
                    continue
                if fn is None:
                    cands = sorted(s.miss)
                else:
                    cands = fn(h, j, s, prev=tok[j - 1].ids if j else None, off=off)[:depth]
                n = 0
                for e in cands:
                    if n >= k:
                        break
                    if (j, e) in lru or (j, e) in queued:
                        continue
                    queue.append(Item(j, e, at, R))
                    queued.add((j, e))
                    n += 1
                cnt["issued"] += n

        def advance(T):
            nonlocal t_drive
            cur = t_drive
            occ = sum(1 for it in queue if it.started and not it.released)
            busy = False
            for it in queue:
                if it.released or it.rem <= 0:
                    continue
                if not it.started and occ >= shelf:
                    break
                start = max(cur, it.issue)
                if start >= T:
                    break
                work = min(it.rem, (T - start) * rate)
                if not it.started:
                    it.started = True
                    occ += 1
                it.rem -= work
                cur = start + work / rate
                if it.rem > 1e-12:
                    busy = True
                    break
            t_drive = max(t_drive, min(cur, T))
            return busy

        if decode and "token" in whens:
            for j in range(N_LAYER):
                issue(j, 0.0, "token")
        for j, s in enumerate(tok):
            t_start = t
            if decode and "P2" in whens:
                issue(j, t_start + c_probe, "P2")
            if decode and "P0" in whens and j + 1 < N_LAYER:
                issue(j + 1, t_start + c_probe, "P0")
            if decode and "P1" in whens and j + 1 < N_LAYER:
                issue(j + 1, t_start + A + c_probe, "P1")
            T = t_start + A + (c_probe if decode else 0.0)
            busy = advance(T) if decode else False
            missed = lru.access(s.layer, s.ids)
            demand = 0.0
            urgent = 0.0
            if decode:
                for it in queue:
                    if it.layer != j or it.released:
                        continue
                    it.released = True
                    if it.e in missed:
                        cnt["covered_issued"] += 1
                        if it.rem <= 1e-12:
                            cnt["covered_done"] += 1
                        else:
                            demand += it.rem
                            urgent += it.rem
                        cnt["saved_ms"] += R - it.rem
                        missed = missed - {it.e}
                    elif it.started:
                        cnt["wasted_started"] += 1
                        cnt["wasted_ms"] += R - it.rem
                queue = [it for it in queue if not it.released]
            demand += R * len(missed)
            cnt["misses"] += len(missed)
            if demand > 0:
                if busy:
                    cnt["preempts"] += 1
                t0 = max(T, t_drive) + (pre if busy else 0.0)
                t_drive = t0 + demand
                t_res = t0 + urgent if adopt_block else T
                t = max(t_res + M, t_drive + TAIL)
            else:
                t = T + M
            if decode and "cond" in whens and j + 1 < N_LAYER:
                issue(j + 1, T, "cond")
        h.add_token(tok)
        ms_tok.append(t + H)
    return ms_tok, cnt


def pollution(run, pred, k, off=None, depth=16):
    """Timing-free alternative to the shelf: prefetched experts are inserted
    straight into the LRU cache (MRU, evicting the least recently used entry),
    as a naive prefetch through the existing cache would.  Returns per decode
    token (demand misses, speculative reads, wasted speculative reads)."""
    toks, first_dec = run["toks"], run["first_dec"]
    lru = LRU(run["budget"][0], run["snap"])
    h = History()
    sources = PREDICTORS[pred]
    whens = {w for w, _ in sources}
    dem = spec = waste = 0
    for ti, tok in enumerate(toks):
        dec = ti >= first_dec
        pend = collections.defaultdict(set)

        def issue(j, when):
            s = tok[j]
            for w, fn in sources:
                if w != when:
                    continue
                cands = sorted(s.miss) if fn is None else \
                    fn(h, j, s, prev=tok[j - 1].ids if j else None, off=off)[:depth]
                got = []
                for e in cands:
                    if len(got) >= k:
                        break
                    if (j, e) in lru:
                        continue
                    got.append(e)
                for e in got:
                    lru.access(j, [e])
                pend[j] |= set(got)

        if dec and "token" in whens:
            for j in range(N_LAYER):
                issue(j, "token")
        for j, s in enumerate(tok):
            if dec and "P2" in whens:
                issue(j, "P2")
            if dec and j + 1 < N_LAYER:
                for w in ("P0", "P1"):
                    if w in whens:
                        issue(j + 1, w)
            m = lru.access(s.layer, s.ids)
            if dec:
                dem += len(m)
                p = pend.pop(j, set())
                spec += len(p)
                waste += len(p - set(s.ids))
            if dec and "cond" in whens and j + 1 < N_LAYER:
                issue(j + 1, "cond")
        h.add_token(tok)
    n = len(toks) - first_dec
    return dem / n, spec / n, waste / n


def fit(run):
    """Fit R (drive ms per expert) and A (pre-router ms) to measured ms."""
    perf = run["perf"]
    toks = run["toks"][run["first_dec"]:]
    meas = [p["ms"] for p in perf]
    best = None
    for r100 in range(80, 300):
        R = r100 / 100.0
        exp = []
        for tok in toks:
            e = 0.0
            for s in tok:
                n = len(s.miss)
                if n:
                    e += max(M, n * R + TAIL) - M
            exp.append(e)
        # meas = 40*(A+M) + H + exp  ->  A from the mean residual
        resid = [m - x for m, x in zip(meas, exp)]
        K = statistics.median(resid)
        sse = sum((x - K) ** 2 for x in resid)
        if best is None or sse < best[0]:
            best = (sse, R, K)
    sse, R, K = best
    A = (K - H) / N_LAYER - M
    mm = statistics.mean(meas)
    r2 = 1 - sse / sum((m - mm) ** 2 for m in meas)
    return R, A, K, r2


def main():
    out_path = sys.argv[1]
    runs = {}
    for spec in sys.argv[2:]:
        name, trace, log = spec.split(":")
        run = load_run(trace, log)
        runs[name] = run
        print(f"{name}: tokens={len(run['toks'])} decode={len(run['perf'])} "
              f"budget={run['budget']} snap={len(run['snap'])} "
              f"perf_mismatch={run['perf_mismatch']} replay_mismatch={run['replay_mismatch']}/{run['n_steps']}",
              flush=True)
    result = {"runs": {}, "table": []}
    for name, run in runs.items():
        R, A, K, r2 = fit(run)
        dec = run["toks"][run["first_dec"]:]
        misses = sum(len(s.miss) for t in dec for s in t)
        probe_us = {tag: statistics.median([s.probes[tag][0] for t in dec for s in t if tag in s.probes] or [0])
                    for tag in ("P0", "P1", "P2")}
        meas = statistics.mean(p["ms"] for p in run["perf"])
        base_ms, _ = simulate(run, None, 0, R, A, 16, 0.0)
        result["runs"][name] = dict(R=R, A=A, K=K, r2=r2, decode_tokens=len(dec),
                                    misses_per_tok=misses / len(dec), measured_ms=meas,
                                    model_ms=statistics.mean(base_ms), probe_us_median=probe_us)
        print(f"{name}: fit R={R:.2f} ms/expert A={A:.2f} ms r2={r2:.3f}  misses/tok={misses/len(dec):.1f} "
              f"measured {meas:.1f} ms model {statistics.mean(base_ms):.1f} ms probe_us={probe_us}", flush=True)
    # offline conditional tables: leave-one-prompt-out
    offs = {}
    for name in runs:
        hh = History()
        for other, run in runs.items():
            if other != name:
                for tok in run["toks"]:
                    hh.add_token(tok)
        offs[name] = hh
    preds = [p for p in PREDICTORS]
    PRE = 0.35       # default preemption delay (see D5.md)
    PMS = 0.06       # default probe cost per layer and probe
    base_cache = {}

    def emit(name, run, pred, k, depth, A_use, pms, pre, shelf=16, tag="main", rate=1.0, adopt_block=False):
        R = result["runs"][name]["R"]
        bkey = (name, A_use, shelf)
        if bkey not in base_cache:
            base_cache[bkey] = statistics.mean(simulate(run, None, 0, R, A_use, shelf, 0.0)[0])
        bm = base_cache[bkey]
        ms, c = simulate(run, pred, k, R, A_use, shelf, pms, off=offs[name], depth=depth, pre=pre, rate=rate,
                         adopt_block=adopt_block)
        n_tok = len(ms)
        mean = statistics.mean(ms)
        tot = c["misses"] + c["covered_issued"]
        row = dict(tag=tag, run=name, pred=pred, k=k, depth=depth, shelf=shelf, A=round(A_use, 3),
                   probe_ms=pms, pre=pre, rate=rate,
                   recall=c["covered_issued"] / max(1, tot),
                   in_time=c["covered_done"] / max(1, tot),
                   issued_per_tok=c["issued"] / n_tok,
                   wasted_reads_per_tok=c["wasted_started"] / n_tok,
                   wasted_drive_ms_per_tok=c["wasted_ms"] / n_tok,
                   saved_drive_ms_per_tok=c["saved_ms"] / n_tok,
                   preempts_per_tok=c["preempts"] / n_tok,
                   base_ms=bm, ms=mean, gain_pct=(bm / mean - 1) * 100)
        result["table"].append(row)
        print(json.dumps({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
        return row

    only = __import__("os").environ.get("D5_ONLY")
    for name, run in runs.items():
        if only and name != only:
            continue
        A = result["runs"][name]["A"]
        for pred in preds:
            if pred.startswith("oracle"):
                emit(name, run, pred, 6, 16, A, PMS, PRE)
                continue
            for depth in (6, 16):
                for k in (1, 2, 4, 6):
                    if depth == 6 and k == 6 and False:
                        pass
                    emit(name, run, pred, k, depth, A, PMS, PRE)
        # sensitivity for the probe predictors at k = 2, depth 6
        for pred in ("P1", "P2", "P0+P2", "P1+P2"):
            emit(name, run, pred, 2, 6, A, PMS, 0.2, tag="pre0.2,rate0.5", rate=0.5)
            emit(name, run, pred, 1, 6, A, PMS, 0.2, tag="k1,pre0.2,rate0.5", rate=0.5)
            emit(name, run, pred, 1, 6, A, PMS, 0.7, tag="k1,pre0.7", rate=1.0)
            emit(name, run, pred, 1, 6, A, PMS, 0.2, tag="k1,pre0.2,rate0.7,block", rate=0.7, adopt_block=True)
            emit(name, run, pred, 1, 6, A, PMS, 0.35, tag="k1,block", adopt_block=True)
            for (A_use, pms, pre, tag) in ((A, PMS, 0.0, "pre0"), (A, PMS, 1.5, "pre1.5"),
                                           (1.7, PMS, PRE, "A1.7"), (A, 0.3, PRE, "probe0.3"),
                                           (1.0, PMS, PRE, "A1.0")):
                emit(name, run, pred, 2, 6, A_use, pms, pre, tag=tag)
            emit(name, run, pred, 2, 6, A, PMS, PRE, shelf=4, tag="shelf4")
    # naive prefetch into the cache itself (pollution / eviction cost)
    result["pollution"] = []
    for name, run in runs.items():
        if only and name != only:
            continue
        for pred in preds:
            for k in (1, 2, 4):
                if pred.startswith("oracle") and k != 4:
                    continue
                dem, spec, waste = pollution(run, pred, k, off=offs[name], depth=6)
                row = dict(run=name, pred=pred, k=k, recorded_misses=result["runs"][name]["misses_per_tok"],
                           demand_misses=dem, spec_reads=spec, wasted=waste, total_reads=dem + spec)
                result["pollution"].append(row)
                print("POLL", json.dumps({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in row.items()}),
                      flush=True)
    # shelf cost: replay the cache with fewer slots
    for name, run in runs.items():
        cap = run["budget"][0]
        for shelf in (8, 16, 32):
            lru = LRU(cap - shelf, run["snap"])
            extra = 0
            for ti, tok in enumerate(run["toks"]):
                for s in tok:
                    mm = lru.access(s.layer, s.ids)
                    if ti >= run["first_dec"]:
                        extra += len(mm) - len(s.miss)
            result["runs"][name][f"shelf{shelf}_extra_misses_per_tok"] = extra / len(run["perf"])
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
