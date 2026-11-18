#!/usr/bin/env python3
"""D15 part A.1: characterise the experts that come back after eviction.

For each trace, replay the runtime's LRU (sim.LRU) over the demand stream and
classify each decode miss (plain steps and verify rounds) as first use or a
*return* (the key was looked up before in the session).  For the returns:
  * reuse distance in tokens (positions since the key's previous lookup) and
    in distinct experts of the same layer looked up in between (per-layer
    stack distance);
  * concentration per layer and per key;
  * predictability: consecutive reuse distances of the same key (Spearman
    rank correlation of log distances; share of next distance within 2x of
    the previous one, against a random pairing);
  * clustering: P(return miss in layer L+1 of the same step | return miss in
    layer L) against the unconditional rate; P(an expert co-selected with k
    at k's previous lookup, and selected again, also returns | k returns)
    against the rate at which any selected expert is a return miss.

Usage: returns.py RESULTS.jsonl OUT.json TRACE.gz ...
(needs scripts/d12 next to this directory)
"""
import bisect
import json
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "d12"))
import sim  # noqa: E402


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


class Fenwick:
    def __init__(self, n):
        self.n = n + 2
        self.t = [0] * (self.n + 1)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def sum(self, i):  # prefix [0, i)
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def analyse(path, P):
    st = sim.Stream(path, P)
    cap = st.capacity_runtime
    lru = sim.LRU(cap)
    n_layer_acc = defaultdict(int)
    for c in st.calls:
        n_layer_acc[c[1]] += len(c[3])
    stacks = {L: Fenwick(n_layer_acc[L]) for L in n_layer_acc}
    lseq = defaultdict(int)
    last_seq, last_pos = {}, {}
    hist = defaultdict(list)
    ret_tok, ret_stack = [], []
    first = ret = 0
    per_layer_ret, per_key_ret = defaultdict(int), defaultdict(int)
    mark_ret_layers, mark_layers = defaultdict(set), defaultdict(set)
    for ci, (src, layer, prot, keys, phase, t, mk, rows) in enumerate(st.calls):
        dec = phase in ("step", "verify")
        for k in keys:
            hit = lru.access(k, ci, t, prot, None)
            if dec:
                mark_layers[mk].add(layer)
            if not hit and dec:
                if k in last_pos:
                    ret += 1
                    ret_tok.append(t - last_pos[k])
                    ret_stack.append(stacks[layer].sum(lseq[layer]) - stacks[layer].sum(last_seq[k] + 1))
                    per_layer_ret[layer] += 1
                    per_key_ret[k] += 1
                    mark_ret_layers[mk].add(layer)
                else:
                    first += 1
            if k in last_pos and t > last_pos[k]:
                hist[k].append(t - last_pos[k])
            if k in last_seq:
                stacks[layer].add(last_seq[k], -1)
            last_seq[k] = lseq[layer]
            stacks[layer].add(lseq[layer], 1)
            lseq[layer] += 1
            last_pos[k] = t
    out = dict(tag=os.path.basename(path).split(".trace")[0], capacity=cap,
               decode_misses=first + ret, first_use=first, returns=ret,
               return_share=ret / max(1, first + ret))
    out["reuse_tokens_pct"] = {str(q): pct(ret_tok, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["reuse_stack_pct"] = {str(q): pct(ret_stack, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    lv = sorted(per_layer_ret.values(), reverse=True)
    out["layer_top10_share"] = sum(lv[:10]) / max(1, ret)
    out["layer_ret_min_max"] = (min(lv) if lv else 0, max(lv) if lv else 0)
    kv = sorted(per_key_ret.values(), reverse=True)
    nk = len(kv)
    out["keys_returning"] = nk
    out["key_top10pct_share"] = sum(kv[:max(1, nk // 10)]) / max(1, ret)
    out["returns_per_returning_key_mean"] = ret / max(1, nk)
    xs, ys, within2 = [], [], 0
    for k, ds in hist.items():
        for a, b in zip(ds, ds[1:]):
            xs.append(math.log(a))
            ys.append(math.log(b))
            if b <= 2 * a and a <= 2 * b:
                within2 += 1
    out["consecutive_pairs"] = len(xs)
    out["spearman_log_next_vs_prev"] = spearman(xs, ys) if len(xs) > 10 else None
    out["next_within_2x_prev"] = within2 / max(1, len(xs))
    sh = ys[:]
    random.Random(0).shuffle(sh)
    out["next_within_2x_random"] = sum(1 for a, b in zip(xs, sh) if abs(a - b) <= math.log(2)) / max(1, len(xs))
    cond_n = cond_hit = base_n = base_hit = 0
    for mk, layers in mark_layers.items():
        rl = mark_ret_layers.get(mk, set())
        for L in layers:
            if L + 1 in layers:
                base_n += 1
                base_hit += (L + 1) in rl
                if L in rl:
                    cond_n += 1
                    cond_hit += (L + 1) in rl
    out["p_ret_next_layer_given_ret"] = cond_hit / max(1, cond_n)
    out["p_ret_next_layer_base"] = base_hit / max(1, base_n)
    return out, st, cap


def partner_corr(st, cap):
    lru = sim.LRU(cap)
    partners, seen = {}, set()
    cond_n = cond_hit = base_n = base_hit = 0
    for ci, (src, layer, prot, keys, phase, t, mk, rows) in enumerate(st.calls):
        rets = set()
        for k in keys:
            if not lru.access(k, ci, t, prot, None) and k in seen:
                rets.add(k)
        if src == "s" and phase == "step":
            ks = set(keys)
            for k in rets:
                for pk in partners.get(k, ()):
                    if pk in ks:
                        cond_n += 1
                        cond_hit += pk in rets
            for k in keys:
                base_n += 1
                base_hit += k in rets
        seen.update(keys)
        if src == "s":
            for k in keys:
                partners[k] = [x for x in keys if x != k]
    return dict(p_partner_returns_given_return_and_reselected=cond_hit / max(1, cond_n),
                partner_cases=cond_n,
                p_selected_expert_is_return_miss=base_hit / max(1, base_n))


def main():
    rec = {}
    for l in open(sys.argv[1]):
        r = json.loads(l)
        rec[r["tag"]] = r
    res = {}
    for path in sys.argv[3:]:
        tag = os.path.basename(path).split(".trace")[0]
        o, st, cap = analyse(path, rec[tag]["prompt_tokens"])
        o.update(partner_corr(st, cap))
        res[tag] = o
        print(tag, "done", file=sys.stderr)
        json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
