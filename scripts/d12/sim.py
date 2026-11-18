#!/usr/bin/env python3
"""D12 trace-driven cache-policy simulator.

Replays the demand stream of a DS4_D12_CACHE_TRACE (every routed-expert
lookup, in order, grouped by cache call) through alternative eviction
policies at a fixed slot count and reports misses by phase.

Rules shared by every policy (they are the runtime's):
  * every looked-up expert must be resident when its call runs, so a miss
    always inserts (no bypass);
  * the current call's protected set (same layer, ids of the call: the six
    selected experts of a step, the union of a verify round's rows, the
    seed list of a prefill seed) is never evicted while serving that call;
  * seed calls (prefill layer sweep, src d/e) insert like the runtime does,
    but their misses are device copies from the layer slot, not SSD reads,
    so they are counted apart.
Time for the decay policies is the token position of the call's phase mark.

Policies: lru (the runtime's, minus the 't' bumps, which only reorder the six
experts of the current step), belady (MIN, call granularity: all ids of one
call are needed at the same instant; next use = next call that looks the key
up), lfu:H (decayed frequency, half-life H tokens, history kept for evicted
keys), lru2 (LRU-K, K=2, history kept), arc, lru-layer-eq / lru-layer-dem
(per-layer LRU partitions: equal shares, or shares proportional to each
layer's LRU misses in the same trace), lru-noseed (LRU with the prefill seed
calls dropped).
"""
import heapq
import math
from collections import OrderedDict, defaultdict
import d12trace as T

NL = 40


class Stream:
    """Demand stream: per call (src, layer, prot, keys, phase, t, mark)."""

    def __init__(self, path, prompt_tokens):
        marks, calls, flushes, loose = T.load(path)
        mid = [f for f in flushes if 0 < f < len(calls)]
        assert not mid, "resident cache flushed during the run"
        self.path = path
        self.marks = marks
        self.P = prompt_tokens
        self.calls = []
        self.trace_hits = self.trace_misses = 0
        for c in calls:
            if c.map != 0:
                raise ValueError("non-main model in the resident cache")
            m = marks[c.mark] if c.mark >= 0 else None
            if c.src in "de":
                phase = "seed"
            elif m is None:
                phase = "pre"
            elif m.kind == "V":
                phase = "verify"
            elif m.kind == "S":
                phase = "prompt" if m.pos < prompt_tokens else "step"
            else:
                phase = "sweep"
            dem = c.demand()
            # dedupe within a call, keep first-occurrence order
            seen, keys = set(), []
            for k, hit in dem:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
                if hit:
                    self.trace_hits += 1
                else:
                    self.trace_misses += 1
            self.calls.append((c.src, c.layer, frozenset(c.protected()), keys, phase,
                               m.pos if m else -1, c.mark, m.rows if m else 0))
        # next-use (call index) for Belady
        nxt = {}
        self.next_use = [None] * len(self.calls)
        for ci in range(len(self.calls) - 1, -1, -1):
            keys = self.calls[ci][3]
            self.next_use[ci] = [nxt.get(k, math.inf) for k in keys]
            for k in keys:
                nxt[k] = ci
        self.capacity_runtime = max(m.slots for m in marks) if marks else None
        self.budget = marks[0].budget if marks else None


# ---------------------------------------------------------------- policies
class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def access(self, k, ci, t, prot, nu):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        while len(self.d) >= self.cap:
            for v in self.d:
                if v not in prot:
                    del self.d[v]
                    break
            else:
                raise RuntimeError("all protected")
        self.d[k] = None
        return False


class Belady:
    def __init__(self, cap):
        self.cap, self.res, self.nu, self.h = cap, set(), {}, []

    def access(self, k, ci, t, prot, nu):
        hit = k in self.res
        if not hit:
            while len(self.res) >= self.cap:
                skipped = []
                while True:
                    negn, v = heapq.heappop(self.h)
                    if v not in self.res or self.nu[v] != -negn:
                        continue
                    if v in prot:
                        skipped.append((negn, v))
                        continue
                    break
                for s in skipped:
                    heapq.heappush(self.h, s)
                self.res.discard(v)
            self.res.add(k)
        self.nu[k] = nu
        heapq.heappush(self.h, (-nu, k))
        return hit


class LFUDecay:
    """Decayed count s, half-life H tokens; order by v = t/H + log2(s), which
    ranks keys by decayed score at any common time."""

    def __init__(self, cap, H):
        self.cap, self.H = cap, float(H)
        self.res, self.v, self.s, self.tl, self.h = set(), {}, {}, {}, []
        self.ctr = 0

    def access(self, k, ci, t, prot, nu):
        hit = k in self.res
        s = self.s.get(k)
        s = 1.0 if s is None else s * 2.0 ** (-(t - self.tl[k]) / self.H) + 1.0
        if not hit:
            while len(self.res) >= self.cap:
                skipped = []
                while True:
                    v, c, x = heapq.heappop(self.h)
                    if x not in self.res or self.v[x] != v:
                        continue
                    if x in prot:
                        skipped.append((v, c, x))
                        continue
                    break
                for e in skipped:
                    heapq.heappush(self.h, e)
                self.res.discard(x)
            self.res.add(k)
        self.s[k], self.tl[k] = s, t
        vv = t / self.H + math.log2(s)
        self.v[k] = vv
        self.ctr += 1
        heapq.heappush(self.h, (vv, self.ctr, k))   # ties: older entry first
        return hit


class LRU2:
    """LRU-K with K=2 on call-index time; history kept for evicted keys.
    Victim: oldest 2nd-last access; keys seen once first (by last access)."""

    def __init__(self, cap):
        self.cap, self.res, self.last, self.prev, self.h = cap, set(), {}, {}, []

    def access(self, k, ci, t, prot, nu):
        hit = k in self.res
        if not hit:
            while len(self.res) >= self.cap:
                skipped = []
                while True:
                    p, l, x = heapq.heappop(self.h)
                    if x not in self.res or self.prev.get(x, -1) != p or self.last[x] != l:
                        continue
                    if x in prot:
                        skipped.append((p, l, x))
                        continue
                    break
                for e in skipped:
                    heapq.heappush(self.h, e)
                self.res.discard(x)
            self.res.add(k)
        if k in self.last and self.last[k] != ci:
            self.prev[k] = self.last[k]
        self.last[k] = ci
        heapq.heappush(self.h, (self.prev.get(k, -1), ci, k))
        return hit


class ARC:
    """Adaptive Replacement Cache (Megiddo & Modha), with protected entries
    skipped in REPLACE."""

    def __init__(self, cap):
        self.c, self.p = cap, 0.0
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def _pop_unprot(self, L, prot):
        for v in L:
            if v not in prot:
                del L[v]
                return v
        return None

    def _replace(self, k, prot):
        if self.T1 and (len(self.T1) > self.p or (k in self.B2 and len(self.T1) == self.p)):
            v = self._pop_unprot(self.T1, prot)
            if v is not None:
                self.B1[v] = None
                return
            v = self._pop_unprot(self.T2, prot)
            self.B2[v] = None
        else:
            v = self._pop_unprot(self.T2, prot)
            if v is not None:
                self.B2[v] = None
                return
            v = self._pop_unprot(self.T1, prot)
            self.B1[v] = None

    def access(self, k, ci, t, prot, nu):
        c = self.c
        if k in self.T1:
            del self.T1[k]
            self.T2[k] = None
            return True
        if k in self.T2:
            self.T2.move_to_end(k)
            return True
        if k in self.B1:
            self.p = min(c, self.p + max(len(self.B2) / len(self.B1), 1))
            self._replace(k, prot)
            del self.B1[k]
            self.T2[k] = None
            return False
        if k in self.B2:
            self.p = max(0, self.p - max(len(self.B1) / len(self.B2), 1))
            self._replace(k, prot)
            del self.B2[k]
            self.T2[k] = None
            return False
        l1 = len(self.T1) + len(self.B1)
        if l1 == c:
            if len(self.T1) < c:
                self.B1.popitem(last=False)
                self._replace(k, prot)
            else:
                v = self._pop_unprot(self.T1, prot)
        elif l1 < c:
            tot = l1 + len(self.T2) + len(self.B2)
            if tot >= c:
                if tot == 2 * c:
                    self.B2.popitem(last=False)
                self._replace(k, prot)
        self.T1[k] = None
        return False


class LayerLRU:
    def __init__(self, cap, shares):
        q = [max(6, int(cap * s)) for s in shares]
        # fix rounding so the quotas sum to cap
        i = 0
        while sum(q) < cap:
            q[i % NL] += 1
            i += 1
        while sum(q) > cap:
            j = max(range(NL), key=lambda x: q[x])
            q[j] -= 1
        self.q = q
        self.d = [OrderedDict() for _ in range(NL)]

    def access(self, k, ci, t, prot, nu):
        L = self.d[k // 384]
        if k in L:
            L.move_to_end(k)
            return True
        while len(L) >= self.q[k // 384]:
            for v in L:
                if v not in prot:
                    del L[v]
                    break
            else:
                # whole partition protected (a wide verify round): borrow a
                # slot from the fullest other layer for the rest of the run
                j = max((x for x in range(NL) if x != k // 384), key=lambda x: len(self.d[x]))
                self.d[j].popitem(last=False)
                self.q[j] -= 1
                self.q[k // 384] += 1
                self.borrowed = getattr(self, "borrowed", 0) + 1
                break
        L[k] = None
        return False


class TwoQ:
    """Full 2Q (Johnson & Shasha): A1in FIFO (Kin = 25 %), ghost A1out
    (Kout = 50 %), Am LRU.  Protected entries are skipped."""

    def __init__(self, cap, kin=0.25, kout=0.5):
        self.cap, self.kin, self.kout = cap, int(cap * kin), int(cap * kout)
        self.A1in, self.A1out, self.Am = OrderedDict(), OrderedDict(), OrderedDict()

    def _pop(self, L, prot):
        for v in L:
            if v not in prot:
                del L[v]
                return v
        return None

    def _reclaim(self, prot):
        if len(self.A1in) > self.kin:
            v = self._pop(self.A1in, prot)
            if v is not None:
                self.A1out[v] = None
                if len(self.A1out) > self.kout:
                    self.A1out.popitem(last=False)
                return
        v = self._pop(self.Am, prot)
        if v is None:
            v = self._pop(self.A1in, prot)
            self.A1out[v] = None

    def access(self, k, ci, t, prot, nu):
        if k in self.Am:
            self.Am.move_to_end(k)
            return True
        if k in self.A1in:
            return True
        if len(self.A1in) + len(self.Am) >= self.cap:
            self._reclaim(prot)
        if k in self.A1out:
            del self.A1out[k]
            self.Am[k] = None
        else:
            self.A1in[k] = None
        return False


class LIRS:
    """LIRS (Jiang & Zhang), resident HIR share `hir` of the slots; the
    non-resident HIR history in the stack is bounded at 3x the slots."""

    def __init__(self, cap, hir=0.01):
        self.cap = cap
        self.lhirs = max(1, int(cap * hir))
        self.llirs = cap - self.lhirs
        self.S, self.Q = OrderedDict(), OrderedDict()
        self.lir, self.res = set(), set()

    def _prune(self):
        while self.S:
            b = next(iter(self.S))
            if b in self.lir:
                break
            del self.S[b]

    def _demote_bottom(self):
        b = next(iter(self.S))
        del self.S[b]
        self.lir.discard(b)
        self.Q[b] = None
        self._prune()

    def _bound(self):
        if len(self.S) > 3 * self.cap:
            for b in list(self.S):
                if len(self.S) <= 2 * self.cap:
                    break
                if b not in self.res:
                    del self.S[b]
            self._prune()

    def access(self, k, ci, t, prot, nu):
        if k in self.lir:
            bottom = next(iter(self.S)) == k
            self.S.move_to_end(k)
            if bottom:
                self._prune()
            return True
        if k in self.res:           # resident HIR
            if k in self.S:
                self.S.move_to_end(k)
                del self.Q[k]
                self.lir.add(k)
                self._demote_bottom()
            else:
                self.S[k] = None
                self.Q.move_to_end(k)
            self._bound()
            return True
        # miss
        if len(self.res) >= self.cap:
            v = None
            for x in self.Q:
                if x not in prot:
                    v = x
                    break
            if v is None:            # every resident HIR is protected: evict bottom LIR
                for x in self.S:
                    if x in self.lir and x not in prot:
                        v = x
                        break
                self.lir.discard(v)
                del self.S[v]
                self._prune()
            else:
                del self.Q[v]
            self.res.discard(v)
        self.res.add(k)
        if len(self.lir) < self.llirs:
            self.lir.add(k)
            self.S[k] = None
            self.S.move_to_end(k)
        elif k in self.S:
            self.S.move_to_end(k)
            self.lir.add(k)
            self._demote_bottom()
        else:
            self.S[k] = None
            self.Q[k] = None
        self._bound()
        return False


class S3FIFO:
    """S3-FIFO (Yang et al. 2023): small FIFO (10 %), main FIFO with a 2-bit
    frequency, ghost FIFO of keys evicted from small (size = main)."""

    def __init__(self, cap, small=0.1):
        self.cap = cap
        self.ks = max(1, int(cap * small))
        self.S, self.M, self.G = OrderedDict(), OrderedDict(), OrderedDict()
        self.f = {}

    def _evict_s(self, prot):
        while self.S:
            v = next(iter(self.S))
            del self.S[v]
            if v in prot:
                self.S[v] = None            # keep, rotate
                if all(x in prot for x in self.S):
                    return False
                continue
            if self.f.get(v, 0) > 0:
                self.M[v] = None
                self.f[v] = 0
                if len(self.M) > self.cap - self.ks:
                    self._evict_m(prot)
            else:
                self.G[v] = None
                if len(self.G) > self.cap - self.ks:
                    self.G.popitem(last=False)
                self.f.pop(v, None)
                return True
        return False

    def _evict_m(self, prot):
        guard = 0
        while self.M:
            v = next(iter(self.M))
            del self.M[v]
            if self.f.get(v, 0) > 0 or v in prot:
                if v not in prot:
                    self.f[v] -= 1
                self.M[v] = None
                guard += 1
                if guard > 4 * len(self.M) + 8:
                    raise RuntimeError("s3fifo: main all protected")
                continue
            self.f.pop(v, None)
            return True
        return False

    def access(self, k, ci, t, prot, nu):
        if k in self.S or k in self.M:
            self.f[k] = min(3, self.f.get(k, 0) + 1)
            return True
        while len(self.S) + len(self.M) >= self.cap:
            if len(self.S) >= self.ks:
                if self._evict_s(prot):
                    continue
            if not self._evict_m(prot):
                self._evict_s(prot)
        if k in self.G:
            del self.G[k]
            self.M[k] = None
        else:
            self.S[k] = None
        self.f[k] = 0
        return False


class AdaptiveLFU:
    """Decayed-frequency eviction whose half-life is chosen online: shadow
    caches (metadata only) run each candidate half-life; every W tokens the
    real cache switches to the candidate with the fewest shadow misses over
    the last window.  H = 0 means plain LRU."""

    def __init__(self, cap, Hs=(0, 16, 64, 256), W=32):
        self.cap, self.Hs, self.W = cap, Hs, W
        self.shadow = [LRU(cap) if H == 0 else LFUDecay(cap, H) for H in Hs]
        self.win = [0] * len(Hs)
        self.cur = 0
        self.t0 = None
        self.res = set()
        self.last = OrderedDict()     # LRU order for H = 0
        self.s, self.tl = {}, {}
        self.choices = []

    def _score(self, k, H, t):
        if H == 0:
            return None
        return self.tl[k] / H + math.log2(self.s[k][H])

    def access(self, k, ci, t, prot, nu):
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 >= self.W:
            self.cur = min(range(len(self.Hs)), key=lambda i: (self.win[i], i))
            self.choices.append(self.Hs[self.cur])
            self.win = [0] * len(self.Hs)
            self.t0 = t
        for i, sh in enumerate(self.shadow):
            if not sh.access(k, ci, t, prot, nu):
                self.win[i] += 1
        hit = k in self.res
        # update scores for every H
        d = self.s.setdefault(k, {})
        for H in self.Hs:
            if H:
                d[H] = 1.0 if H not in d else d[H] * 2.0 ** (-(t - self.tl[k]) / H) + 1.0
        if not hit:
            while len(self.res) >= self.cap:
                H = self.Hs[self.cur]
                if H == 0:
                    v = next(x for x in self.last if x not in prot)
                else:
                    # exact scan of the minimum (simulation only; a real
                    # implementation keeps one heap per candidate)
                    v = min((x for x in self.res if x not in prot),
                            key=lambda x: self.tl[x] / H + math.log2(self.s[x][H]))
                self.res.discard(v)
                self.last.pop(v, None)
            self.res.add(k)
        self.tl[k] = t
        self.last[k] = None
        self.last.move_to_end(k)
        return hit


# ---------------------------------------------------------------- replay
def replay(st, policy, skip_seeds=False):
    """Returns per-call miss counts (list) and a summary dict."""
    miss = [0] * len(st.calls)
    for ci, (src, layer, prot, keys, phase, t, mk, rows) in enumerate(st.calls):
        if skip_seeds and phase == "seed":
            continue
        nus = st.next_use[ci]
        m = 0
        for k, nu in zip(keys, nus):
            if not policy.access(k, ci, t, prot, nu):
                m += 1
        miss[ci] = m
    return miss


def make(name, cap, st=None, lru_layer_miss=None):
    if name == "lru" or name == "lru-noseed":
        return LRU(cap)
    if name == "belady":
        return Belady(cap)
    if name.startswith("lfu:"):
        return LFUDecay(cap, float(name[4:]))
    if name == "lru2":
        return LRU2(cap)
    if name == "arc":
        return ARC(cap)
    if name == "s3fifo":
        return S3FIFO(cap)
    if name.startswith("alfu"):
        return AdaptiveLFU(cap)
    if name == "2q":
        return TwoQ(cap)
    if name.startswith("lirs"):
        return LIRS(cap, float(name[5:]) if ":" in name else 0.01)
    if name == "lru-layer-eq":
        return LayerLRU(cap, [1.0 / NL] * NL)
    if name == "lru-layer-dem":
        tot = sum(lru_layer_miss)
        # demand share: half equal, half proportional to LRU misses (a pure
        # proportional split starves layers whose misses are rare)
        return LayerLRU(cap, [0.5 / NL + 0.5 * x / tot for x in lru_layer_miss])
    raise ValueError(name)


def first_use_flags(st):
    """Per call: number of keys looked up for the first time in the session."""
    seen, out = set(), []
    for (src, layer, prot, keys, phase, t, mk, rows) in st.calls:
        n = 0
        for k in keys:
            if k not in seen:
                seen.add(k)
                n += 1
        out.append(n)
    return out


def summarize(st, miss, buckets=((1, 128), (129, 512), (513, 2048))):
    """Misses by phase; per generated token in buckets (plain steps by step
    index g = pos - P + 1; verify rounds by their first position, divided by
    the tokens the round advanced)."""
    P = st.P
    tot = defaultdict(int)
    per_mark = defaultdict(int)
    for ci, c in enumerate(st.calls):
        tot[c[4]] += miss[ci]
        per_mark[c[6]] += miss[ci]
    # token advance per decode mark
    dec = [m for m in st.marks if (m.kind == "S" and m.pos >= P) or m.kind == "V"]
    out = dict(phase=dict(tot))
    b = {}
    for lo, hi in buckets:
        mm = tok = 0
        rows = 0
        for i, m in enumerate(dec):
            g = m.pos - P + 1
            if not (lo <= g <= hi):
                continue
            adv = (dec[i + 1].pos - m.pos) if i + 1 < len(dec) else None
            if m.kind == "S":
                adv = 1
            if adv is None:
                continue
            mm += per_mark[m.idx]
            tok += adv
            rows += m.rows
        if tok:
            b[f"{lo}-{hi}"] = dict(misses=mm, tokens=tok, rows=rows, per_token=mm / tok)
    out["buckets"] = b
    return out
