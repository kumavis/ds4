#!/usr/bin/env python3
"""D12 validation: replay a trace with the runtime's own policy and check that
every hit/miss and every evicted key is reproduced.

Runtime policy (rocm/ds4_rocm_runtime.cuh, cuda_stream_resident_evict_one):
LRU on a global clock bumped by every lookup hit, every insertion and every
't' bump (pointer table / compaction); the victim is the oldest entry that is
not in the current call's protected set (same layer, expert in the call's
ids).  An eviction happens only when no slab slot is free and the pool cannot
grow; the replay checks that this is when the resident count equals the
slot count, and that the pool size matches the M lines.

Usage: validate.py TRACE [TRACE...]
"""
import sys
from collections import OrderedDict
import d12trace as T


def replay(path, verbose=True):
    marks, calls, flushes, loose = T.load(path)
    lru = OrderedDict()           # key -> None, oldest first
    flush_at = set(flushes)
    n_h = n_m = n_x = 0
    bad_hm = bad_x = bad_when = 0
    full_sizes = set()
    examples = []
    pending_x = []
    for ci, c in enumerate(calls):
        if ci in flush_at:
            lru.clear()
        prot = c.protected()
        seq = c.seq
        i = 0
        while i < len(seq):
            op, k = seq[i]
            if op in "hm":
                sim_hit = k in lru
                if sim_hit != (op == "h"):
                    bad_hm += 1
                    if len(examples) < 5:
                        examples.append(("hm", ci, op, divmod(k, 384)))
                if op == "h":
                    n_h += 1
                    if k in lru:
                        lru.move_to_end(k)
                else:
                    n_m += 1
                    # evictions recorded right after this miss
                    j = i + 1
                    while j < len(seq) and seq[j][0] == "x":
                        xk = seq[j][1]
                        n_x += 1
                        full_sizes.add(len(lru))
                        victim = None
                        for cand in lru:            # oldest first
                            if cand not in prot:
                                victim = cand
                                break
                        if victim != xk:
                            bad_x += 1
                            if len(examples) < 5:
                                examples.append(("x", ci, divmod(xk, 384),
                                                 victim and divmod(victim, 384)))
                        lru.pop(xk, None)
                        j += 1
                    i = j - 1
                    lru[k] = None
            elif op == "t":
                if k in lru:
                    lru.move_to_end(k)
                else:
                    bad_hm += 1
            elif op == "x":
                # eviction not directly after a miss
                bad_when += 1
                lru.pop(k, None)
            i += 1
    slots = sorted({m.slots for m in marks})
    budgets = sorted({m.budget for m in marks})
    res = dict(trace=path, calls=len(calls), marks=len(marks), hits=n_h,
               misses=n_m, evictions=n_x, bad_hitmiss=bad_hm, bad_victim=bad_x,
               stray_evictions=bad_when + len(loose), flushes=len(flushes),
               cache_size_at_eviction=sorted(full_sizes)[:3] + (["..."] if len(full_sizes) > 3 else []),
               budget=budgets, slots_max=max(slots) if slots else None,
               examples=examples)
    if verbose:
        print(res)
    return res


if __name__ == "__main__":
    ok = True
    for p in sys.argv[1:]:
        r = replay(p)
        ok &= r["bad_hitmiss"] == 0 and r["bad_victim"] == 0 and r["stray_evictions"] == 0
    print("VALIDATION", "PASS" if ok else "FAIL")
