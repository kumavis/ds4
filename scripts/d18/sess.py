#!/usr/bin/env python3
"""D18: expert-cache analysis of an agent-session trace (DS4_D12_CACHE_TRACE
through ds4-server, with the D18 Q/q prompt-sync marks).

Marks: Q/q bracket a prompt sync (one turn's new prompt); P = layer-sweep
prefill chunk; S = single-token step (inside Q..q: token-major prompt step;
outside: decode step).  Calls: s (step lookup), e/d (sweep seeds), b (batched).

Phases per call: 'dec' (decode step), 'pstep' (token-major prompt step),
'seed' (sweep seed), 'other'.

Policies are replayed on the demand stream (every lookup inserts; the call's
protected set is never evicted).  Seeds are inserts the runtime makes at the
end of each sweep layer; they are device copies, not SSD reads, so their
misses are not counted as reads, but they evict.

Usage: imported by run_d18.py; sess.py TRACE validates and prints a summary.
"""
import gzip
import heapq
import math
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d12"))
import sim  # noqa: E402

NE = 384
NL = 40


class Mark:
    __slots__ = ("kind", "pos", "rows", "budget", "slots", "resident", "free", "turn", "inprompt",
                 "k", "phase")


class Trace:
    def __init__(self, path):
        op = gzip.open if path.endswith(".gz") else open
        self.path = path
        self.marks = []
        # per call: src, layer, ids(tuple), lookups [(key, hit)], evictions-after list
        self.src, self.layer, self.ids, self.look, self.mk = [], [], [], [], []
        self.flush_at = []
        self.raw_x = []          # per call: list of (after lookup index, evicted key)
        cur = -1
        turn = -1
        inprompt = False
        k = 0
        with op(path, "rt") as fh:
            for line in fh:
                t = line[0]
                if t in "hmtx":
                    if cur < 0:
                        continue
                    _, L, e = line.split()
                    key = int(L) * NE + int(e)
                    if t == "h" or t == "m":
                        self.look[cur].append((key, t == "h"))
                    elif t == "x":
                        self.raw_x[cur].append((len(self.look[cur]), key))
                elif t == "A":
                    f = line.split()
                    self.src.append(f[1])
                    self.layer.append(int(f[3]))
                    self.ids.append(tuple(int(x) for x in f[6:]))
                    self.look.append([])
                    self.raw_x.append([])
                    self.mk.append(len(self.marks) - 1)
                    cur = len(self.src) - 1
                elif t == "M":
                    f = line.split()
                    m = Mark()
                    m.kind = f[1]
                    m.pos, m.rows, m.budget, m.slots, m.resident, m.free = map(int, f[2:8])
                    if m.kind == "Q":
                        turn += 1
                        inprompt = True
                    elif m.kind == "q":
                        inprompt = False
                        k = 0
                    m.turn, m.inprompt = turn, inprompt
                    if m.kind == "S" and not inprompt:
                        m.k = k
                        k += 1
                        m.phase = "dec"
                    elif m.kind == "S":
                        m.k, m.phase = -1, "pstep"
                    else:
                        m.k, m.phase = -1, m.kind
                    self.marks.append(m)
                    cur = -1
                elif t == "F":
                    self.flush_at.append(len(self.src))
                    cur = -1
        n = len(self.src)
        self.phase = []
        for i in range(n):
            if self.src[i] in "de":
                self.phase.append("seed")
            elif self.mk[i] >= 0 and self.marks[self.mk[i]].phase in ("dec", "pstep"):
                self.phase.append(self.marks[self.mk[i]].phase)
            else:
                self.phase.append("other")
        self.pos = [self.marks[m].pos if m >= 0 else -1 for m in self.mk]
        self.keys = []
        self.prot = []
        for i in range(n):
            L = self.layer[i]
            seen, ks = set(), []
            for kk, h in self.look[i]:
                if kk not in seen:
                    seen.add(kk)
                    ks.append(kk)
            self.keys.append(ks)
            self.prot.append(frozenset(L * NE + e for e in self.ids[i]))
        # next use (call index) for Belady, demand lookups only
        nxt = {}
        self.next_use = [None] * n
        for ci in range(n - 1, -1, -1):
            self.next_use[ci] = [nxt.get(kk, math.inf) for kk in self.keys[ci]]
            for kk in self.keys[ci]:
                nxt[kk] = ci
        slots = [m.slots for m in self.marks if m.slots]
        self.cap = max(slots) if slots else None


# ------------------------------------------------------------- validation
def validate(tr):
    """Exact replay of the runtime LRU (clock bumps by lookups only: the 't'
    bumps reorder only the current step's six, see D12) with protected sets;
    evictions where the trace has them; flushes reset.  Returns mismatches."""
    lru = OrderedDict()
    bad_hm = bad_x = 0
    fl = set(tr.flush_at)
    sizes = defaultdict(int)
    for ci in range(len(tr.src)):
        if ci in fl:
            lru.clear()
        prot = tr.prot[ci]
        xs = tr.raw_x[ci]
        xi = 0
        for li, (k, hit) in enumerate(tr.look[ci]):
            if (k in lru) != hit:
                bad_hm += 1
            if hit:
                if k in lru:
                    lru.move_to_end(k)
                continue
            while xi < len(xs) and xs[xi][0] == li + 1:
                xk = xs[xi][1]
                sizes[len(lru)] += 1
                victim = next((v for v in lru if v not in prot), None)
                if victim != xk:
                    bad_x += 1
                lru.pop(xk, None)
                xi += 1
            lru[k] = None
    return dict(bad_hitmiss=bad_hm, bad_victim=bad_x, sizes_at_eviction=dict(sorted(sizes.items())[:5]))


# ------------------------------------------------------------- policies
class LRUC:
    """LRU with variable capacity, retention set and D13 lend/reclaim by
    physical slot (free list LIFO, lent slots pushed on top)."""

    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()
        self.keep = frozenset()
        self.slot = {}
        self.free = []
        self.nslab = 0
        self.lent = False
        self.n_staging = 0
        self.reclaimed = 0

    def _victim(self, prot):
        for v in self.d:
            if v not in prot and v not in self.keep:
                return v
        return None

    def access(self, k, ci, t, prot, nu, seed=False):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        while len(self.d) >= self.cap:
            v = self._victim(prot)
            if v is None:
                if seed:
                    return None          # seed dropped: nothing evictable
                v = next(x for x in self.d if x not in prot)
            del self.d[v]
            self.free.append(self.slot.pop(v))
        if self.free:
            s = self.free.pop()
        else:
            s = ("slab", self.nslab)
            self.nslab += 1
        self.slot[k] = s
        self.d[k] = None
        return False

    # D13
    def lend(self, n):
        if self.lent:
            return
        self.lent = True
        self.cap += n
        self.free.extend(("stg", i) for i in range(n))

    def reclaim(self, n):
        if not self.lent:
            return
        self.lent = False
        for k in [k for k, s in self.slot.items() if s[0] == "stg"]:
            del self.d[k]
            del self.slot[k]
            self.reclaimed += 1
        self.free = [s for s in self.free if s[0] != "stg"]
        self.cap -= n


class Wrap:
    def __init__(self, p):
        self.p = p

    def access(self, k, ci, t, prot, nu, seed=False):
        return self.p.access(k, ci, t, prot, nu)


def pol_cap(pol):
    return getattr(pol, "cap", None) or getattr(getattr(pol, "p", None), "cap", 4577)


def replay(tr, pol, keep_n=0, seed_n=None, d13=0, look_h=None):
    """Returns per-call miss counts, seeds dropped, and sweep prepare hits.

    keep_n: at each Q mark, the keep_n most recently used experts are kept
            (not evictable) until the matching q (prompt-tail retention).
    seed_n: insert only the first seed_n ids (highest priority) of each seed
            call; None = all, as the runtime does.
    d13:    lend this many slots between sweeps, reclaim at each sweep (P).
    """
    n = len(tr.src)
    miss = [0] * n
    dropped = 0
    prep_hits = []           # per seed call: resident experts of its layer before it
    lastmark = -2
    in_sweep = False
    if d13 and hasattr(pol, "lend"):
        pol.lend(d13)
    fl = set(tr.flush_at)
    for ci in range(n):
        mk = tr.mk[ci]
        if mk != lastmark:
            for m in range(lastmark + 1 if lastmark >= -1 else 0, mk + 1):
                if m < 0:
                    continue
                M = tr.marks[m]
                if M.kind == "Q" and keep_n and hasattr(pol, "keep"):
                    pol.keep = frozenset(list(pol.d.keys())[-keep_n:])
                if M.kind == "q" and hasattr(pol, "keep"):
                    pol.keep = frozenset()
                if d13 and hasattr(pol, "lend"):
                    if M.kind == "P":
                        pol.reclaim(d13)
                        in_sweep = True
                    elif in_sweep:
                        pol.lend(d13)
                        in_sweep = False
            lastmark = mk
        ph = tr.phase[ci]
        keys = tr.keys[ci]
        nus = tr.next_use[ci]
        seed = ph == "seed"
        if seed:
            L = tr.layer[ci]
            res = getattr(pol, "d", None)
            if res is not None:
                prep_hits.append(sum(1 for kk in res if kk // NE == L))
            n_eff = seed_n
            if seed_n == "cold":
                # full seeding only while this layer holds fewer than half its
                # share of the cache (cold start / after a restart), else none
                n_eff = None if (prep_hits and prep_hits[-1] < 0.5 * pol_cap(pol) / NL) else 0
            if n_eff is not None:
                L = tr.layer[ci]
                want = set(L * NE + e for e in tr.ids[ci][:n_eff])
                pairs = [(k, u) for k, u in zip(keys, nus) if k in want]
                keys = [p[0] for p in pairs]
                nus = [p[1] for p in pairs]
        m = 0
        for k, u in zip(keys, nus):
            r = pol.access(k, ci, tr.pos[ci], tr.prot[ci], u, seed=seed) if isinstance(pol, (LRUC, Wrap)) \
                else pol.access(k, ci, tr.pos[ci], tr.prot[ci], u)
            if r is None:
                dropped += 1
            elif not r:
                m += 1
        miss[ci] = m
    return miss, dropped, prep_hits


# ------------------------------------------------------------- summaries
FILLS = (("~16k", 12000, 21999), ("~32k", 25000, 44999), ("~64k", 52000, 80999), ("~100k", 81000, 114999))


def per_mark(tr, miss):
    pm = defaultdict(int)
    for ci, m in enumerate(miss):
        pm[tr.mk[ci]] += m
    return pm


def summarize(tr, miss, skip_turns=()):
    """Decode misses per token by fill bucket and by position in the reply
    (k = decode step index after the prompt sync: first 16, 17-64, >64,
    and >16 'rest' as D14), and prompt-step misses."""
    pm = per_mark(tr, miss)
    out = {}
    for name, lo, hi in FILLS:
        b = defaultdict(lambda: [0, 0])
        for i, M in enumerate(tr.marks):
            if M.phase != "dec" or not (lo <= M.pos <= hi) or M.turn in skip_turns:
                continue
            for lab, ok in (("first16", M.k < 16), ("17-64", 16 <= M.k < 64), ("after64", M.k >= 64),
                            ("after16", M.k >= 16), ("all", True)):
                if ok:
                    b[lab][0] += pm[i]
                    b[lab][1] += 1
        out[name] = {lab: (v[0] / v[1] if v[1] else None, v[1]) for lab, v in b.items()}
    tot = defaultdict(int)
    for ci, m in enumerate(miss):
        tot[tr.phase[ci]] += m
    out["totals"] = dict(tot)
    out["dec_tokens"] = sum(1 for M in tr.marks if M.phase == "dec")
    return out


if __name__ == "__main__":
    tr = Trace(sys.argv[1])
    print("calls", len(tr.src), "marks", len(tr.marks), "cap", tr.cap, "flushes", tr.flush_at[:5])
    print("phases", {p: tr.phase.count(p) for p in set(tr.phase)})
    print("marks", {k: sum(1 for m in tr.marks if m.kind == k) for k in set(m.kind for m in tr.marks)})
    print(validate(tr))
