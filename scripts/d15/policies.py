#!/usr/bin/env python3
"""D15 part A.2: history-based policies that try to approximate Belady, and
bounds.

Same demand stream and rules as scripts/d12/sim.py: every lookup inserts, the
call's protected set is never evicted, and seeds insert but are counted
apart.

Realizable policies:
  hawkeye      OPTgen (Belady with bypass, replayed on the past: a lookup is
               an OPT hit iff the slot occupancy over the liveness interval
               [previous lookup, this lookup), counted in cache calls, stays
               below the slot count) trains a 3-bit counter per expert (layer,
               id).  Experts predicted cache-averse go to an averse LRU list
               that is evicted first.  Evicting a friendly expert detrains it.
  hawkeye-sig  the same, with counters shared by a signature (lookups so far,
               capped at 4; log2 bucket of the last reuse distance), like
               Hawkeye's PC-indexed predictor.
  pnu          predicted next use = last lookup + EWMA (alpha 0.5, log space)
               of the expert's own reuse distances (prior: median of all
               reuse distances seen so far).  Evict the furthest predicted
               next use.
  pnu-od       the same, but an overdue expert (now > predicted) is evicted
               first.
  co-same      LRU.  After each call, the 2 most frequent past partners (same
               layer, same token) of each selected expert are bumped to MRU
               if resident.
  co-next      LRU.  After each call in layer L, the 3 most frequent layer L+1
               followers of each selected expert are bumped to MRU if resident.
Bounds (oracles, NOT realizable):
  renewal      each expert's reuse-distance distribution is taken from the
               whole trace (future included).  Evict the largest expected time
               to the next lookup given the current age.  Bounds policies that
               model each expert independently from its own history.
  renewal-g    the same with one distribution pooled over all experts.
  look:h       knows every lookup in the next h token positions (h = 0: the
               rest of the current position), nothing beyond.  Evict the LRU
               expert not needed within the horizon, else the one needed last.
               Bounds router-informed policies that see h tokens ahead.

Usage: policies.py RESULTS.jsonl OUT.json TRACE.gz ...   (D15_POLS=a,b,...)
"""
import bisect
import heapq
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d12"))
import sim  # noqa: E402
import d12trace as T  # noqa: E402

NE = 384


class SegTree:
    """Range add, range max over [0, n)."""

    def __init__(self, n):
        self.n = 1
        while self.n < n:
            self.n *= 2
        self.mx = [0] * (2 * self.n)
        self.lz = [0] * (2 * self.n)

    def _add(self, o, l, r, a, b, v):
        if b <= l or r <= a:
            return
        if a <= l and r <= b:
            self.mx[o] += v
            self.lz[o] += v
            return
        m = (l + r) // 2
        self._add(2 * o, l, m, a, b, v)
        self._add(2 * o + 1, m, r, a, b, v)
        self.mx[o] = max(self.mx[2 * o], self.mx[2 * o + 1]) + self.lz[o]

    def _max(self, o, l, r, a, b):
        if b <= l or r <= a:
            return -10**18
        if a <= l and r <= b:
            return self.mx[o]
        m = (l + r) // 2
        return max(self._max(2 * o, l, m, a, b), self._max(2 * o + 1, m, r, a, b)) + self.lz[o]

    def add(self, a, b, v):
        if a < b:
            self._add(1, 0, self.n, a, b, v)

    def max(self, a, b):
        return self._max(1, 0, self.n, a, b) if a < b else 0


class Base:
    def begin(self, ci, st, raw):
        pass

    def end(self, ci, st, raw):
        pass

    def tick(self, t):
        pass


class Wrap(Base):
    def __init__(self, p):
        self.p = p

    def access(self, *a):
        return self.p.access(*a)


class Hawkeye(Base):
    def __init__(self, cap, ncalls, init=4, sig=False):
        self.cap, self.sig = cap, sig
        self.seg = SegTree(ncalls + 1)
        self.prev = {}
        self.ctr = defaultdict(lambda: init)
        self.av, self.fr = OrderedDict(), OrderedDict()
        self.n, self.lt, self.ld, self.sigof = defaultdict(int), {}, {}, {}

    def access(self, k, ci, t, prot, nu):
        p = self.prev.get(k)
        if p is not None and p < ci:
            c = self.sigof[k] if self.sig else k
            if self.seg.max(p, ci) < self.cap:
                self.seg.add(p, ci, 1)
                self.ctr[c] = min(7, self.ctr[c] + 1)
            else:
                self.ctr[c] = max(0, self.ctr[c] - 1)
        self.prev[k] = ci
        if k in self.lt and t > self.lt[k]:
            self.ld[k] = int(math.log2(t - self.lt[k]))
        self.lt[k] = t
        self.n[k] += 1
        self.sigof[k] = (min(self.n[k], 4), self.ld.get(k, -1))
        ck = self.sigof[k] if self.sig else k
        hit = k in self.av or k in self.fr
        self.av.pop(k, None)
        self.fr.pop(k, None)
        if not hit:
            while len(self.av) + len(self.fr) >= self.cap:
                v = next((x for x in self.av if x not in prot), None)
                if v is not None:
                    del self.av[v]
                    continue
                v = next(x for x in self.fr if x not in prot)
                del self.fr[v]
                cv = self.sigof[v] if self.sig else v
                self.ctr[cv] = max(0, self.ctr[cv] - 1)     # detrain
        (self.fr if self.ctr[ck] >= 4 else self.av)[k] = None
        return hit


class Scored(Base):
    """Evict the resident expert with the largest score (predicted position
    of its next lookup).  Scores are recomputed for all residents whenever
    the token position changes, and on each lookup."""

    def __init__(self, cap):
        self.cap = cap
        self.res = {}
        self.h = []
        self.t = None

    def note(self, k, t):
        pass

    def tick(self, t):
        if t != self.t:
            self.t = t
            self.h = []
            for k in self.res:
                s = self.score(k, t)
                self.res[k] = s
                self.h.append((-s, k))
            heapq.heapify(self.h)

    def access(self, k, ci, t, prot, nu):
        hit = k in self.res
        self.note(k, t)
        if not hit:
            while len(self.res) >= self.cap:
                skipped = []
                while True:
                    ns, v = heapq.heappop(self.h)
                    if v not in self.res or self.res[v] != -ns:
                        continue
                    if v in prot:
                        skipped.append((ns, v))
                        continue
                    break
                for e in skipped:
                    heapq.heappush(self.h, e)
                del self.res[v]
        s = self.score(k, t)
        self.res[k] = s
        heapq.heappush(self.h, (-s, k))
        return hit


class PNU(Scored):
    def __init__(self, cap, overdue=False):
        super().__init__(cap)
        self.last, self.lg, self.allg = {}, {}, []
        self.overdue = overdue

    def note(self, k, t):
        l = self.last.get(k)
        if l is not None and t > l:
            x = math.log(t - l)
            self.lg[k] = x if k not in self.lg else 0.5 * self.lg[k] + 0.5 * x
            bisect.insort(self.allg, x)
        self.last[k] = t

    def score(self, k, t):
        g = self.lg.get(k)
        if g is None:
            g = self.allg[len(self.allg) // 2] if self.allg else math.log(64)
        nxt = self.last[k] + math.exp(g)
        if self.overdue and nxt < t:
            return 1e12 + (t - nxt)
        return nxt


class Renewal(Scored):
    def __init__(self, cap, dists, pooled=False, big=1e6):
        super().__init__(cap)
        self.last, self.big, self.pooled = {}, big, pooled
        if pooled:
            allv = sorted(d for v in dists.values() for d in v)
            self.g = (allv, self._suffix(allv), len(dists))
        else:
            self.d = {k: (sorted(v), self._suffix(sorted(v))) for k, v in dists.items()}

    @staticmethod
    def _suffix(v):
        s = [0.0] * (len(v) + 1)
        for i in range(len(v) - 1, -1, -1):
            s[i] = s[i + 1] + v[i]
        return s

    def note(self, k, t):
        self.last[k] = t

    def score(self, k, t):
        a = t - self.last[k]
        if self.pooled:
            v, suf, ncens = self.g
            i = bisect.bisect_right(v, a)
            m = len(v) - i
            return t + ((suf[i] - a * m) + ncens * self.big) / (m + ncens)
        v, suf = self.d.get(k, ((), [0.0]))
        i = bisect.bisect_right(v, a)
        m = len(v) - i
        return t + ((suf[i] - a * m) + self.big) / (m + 1)   # one censored interval per expert


class Lookahead(Base):
    def __init__(self, cap, st, h):
        self.lru = OrderedDict()
        self.cap, self.h = cap, h
        self.pos = [c[5] for c in st.calls]
        self.nu = {}

    def access(self, k, ci, t, prot, nu):
        self.nu[k] = self.pos[nu] if nu != math.inf else math.inf
        if k in self.lru:
            self.lru.move_to_end(k)
            return True
        while len(self.lru) >= self.cap:
            lim = t + self.h
            victim, far, fk = None, -1, None
            for v in self.lru:
                if v in prot:
                    continue
                n = self.nu[v]
                if n > lim:
                    victim = v
                    break
                if n > far:
                    far, fk = n, v
            del self.lru[victim if victim is not None else fk]
        self.lru[k] = None
        return False


class CoOcc(Base):
    def __init__(self, cap, mode):
        self.lru = sim.LRU(cap)
        self.mode = mode
        self.cnt = defaultdict(lambda: defaultdict(int))
        self.prev_rows = {}

    def access(self, k, ci, t, prot, nu):
        return self.lru.access(k, ci, t, prot, nu)

    def end(self, ci, st, raw):
        c = st.calls[ci]
        src, layer, mk = c[0], c[1], c[6]
        if src not in "sb":
            return
        ids = raw[ci].ids
        rows = [ids[i:i + 6] for i in range(0, len(ids), 6)]
        d = self.lru.d
        if self.mode == "same":
            sel = {e for row in rows for e in row}
            for e in sel:
                cc = self.cnt[layer * NE + e]
                for p in heapq.nlargest(2, cc, key=cc.get):
                    pk = layer * NE + p
                    if pk in d and p not in sel:
                        d.move_to_end(pk)
            for row in rows:
                for e in row:
                    cc = self.cnt[layer * NE + e]
                    for p in row:
                        if p != e:
                            cc[p] += 1
        else:
            per = self.prev_rows.setdefault(mk, {})
            prevr = per.get(layer - 1)
            if prevr is not None and len(prevr) == len(rows):
                for ra, rb in zip(prevr, rows):
                    for e in ra:
                        cc = self.cnt[(layer - 1) * NE + e]
                        for p in rb:
                            cc[p] += 1
            per[layer] = rows
            if len(self.prev_rows) > 4:
                for m in sorted(self.prev_rows)[:-4]:
                    del self.prev_rows[m]
            if layer + 1 < sim.NL:
                sel = {e for row in rows for e in row}
                for e in sel:
                    cc = self.cnt[layer * NE + e]
                    for p in heapq.nlargest(3, cc, key=cc.get):
                        pk = (layer + 1) * NE + p
                        if pk in d:
                            d.move_to_end(pk)


def replay(st, raw, pol):
    miss = [0] * len(st.calls)
    for ci, (src, layer, prot, keys, phase, t, mk, rows) in enumerate(st.calls):
        pol.tick(t)
        pol.begin(ci, st, raw)
        m = 0
        for k, nu in zip(keys, st.next_use[ci]):
            if not pol.access(k, ci, t, prot, nu):
                m += 1
        miss[ci] = m
        pol.end(ci, st, raw)
    return miss


def reuse_dists(st):
    last, d = {}, defaultdict(list)
    for (src, layer, prot, keys, phase, t, mk, rows) in st.calls:
        for k in keys:
            if k in last and t > last[k]:
                d[k].append(t - last[k])
            last[k] = t
    for k in last:
        d.setdefault(k, [])
    return d


POLS = ["lru", "s3fifo", "belady", "hawkeye", "hawkeye-sig", "pnu", "pnu-od", "co-same", "co-next",
        "renewal", "renewal-g", "look:0", "look:1", "look:4", "look:16", "look:64", "look:128"]


def make(name, cap, st, dists):
    if name in ("lru", "s3fifo", "belady"):
        return Wrap(sim.make(name, cap))
    if name == "hawkeye":
        return Hawkeye(cap, len(st.calls))
    if name == "hawkeye-sig":
        return Hawkeye(cap, len(st.calls), sig=True)
    if name == "pnu":
        return PNU(cap)
    if name == "pnu-od":
        return PNU(cap, overdue=True)
    if name == "co-same":
        return CoOcc(cap, "same")
    if name == "co-next":
        return CoOcc(cap, "next")
    if name == "renewal":
        return Renewal(cap, dists)
    if name == "renewal-g":
        return Renewal(cap, dists, pooled=True)
    if name.startswith("look:"):
        return Lookahead(cap, st, int(name[5:]))
    raise ValueError(name)


def main():
    rec = {}
    for l in open(sys.argv[1]):
        r = json.loads(l)
        rec[r["tag"]] = r
    out = {}
    pols = os.environ.get("D15_POLS", ",".join(POLS)).split(",")
    for path in sys.argv[3:]:
        tag = os.path.basename(path).split(".trace")[0]
        st = sim.Stream(path, rec[tag]["prompt_tokens"])
        raw = T.load(path)[1]
        cap = st.capacity_runtime
        dists = reuse_dists(st)
        res = {}
        for p in pols:
            res[p] = sim.summarize(st, replay(st, raw, make(p, cap, st, dists)))
        out[tag] = dict(capacity=cap, policies=res)
        print(tag, {p: round(res[p]["buckets"]["129-512"]["per_token"], 2) for p in pols},
              file=sys.stderr)
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
