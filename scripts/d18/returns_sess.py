#!/usr/bin/env python3
"""D18: reuse distance of returning experts in a session trace (decode misses
on an expert looked up before), as D15 measured for single prompts.
Distances: decode tokens between the previous lookup and this one (prompt
steps and sweeps excluded from the count), and token positions (includes
the prompt tokens appended in between).
Usage: returns_sess.py TRACE [TRACE ...]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sess  # noqa: E402


def pct(v, qs=(0.25, 0.5, 0.75, 0.9)):
    v = sorted(v)
    return [v[min(len(v) - 1, int(q * len(v)))] for q in qs] if v else []


for path in sys.argv[1:]:
    tr = sess.Trace(path)
    dec_idx = {}
    n = 0
    for i, M in enumerate(tr.marks):
        if M.phase == "dec":
            n += 1
        dec_idx[i] = n
    last_d, last_p = {}, {}
    dd, dp, same_turn = [], [], 0
    last_turn = {}
    for ci in range(len(tr.src)):
        mk = tr.mk[ci]
        M = tr.marks[mk] if mk >= 0 else None
        d = dec_idx.get(mk, 0)
        for k, hit in tr.look[ci]:
            if tr.phase[ci] == "dec" and not hit and k in last_d:
                dd.append(d - last_d[k])
                dp.append(M.pos - last_p[k])
                same_turn += last_turn.get(k) == M.turn
            last_d[k], last_p[k] = d, (M.pos if M else 0)
            last_turn[k] = M.turn if M else -1
    print(os.path.basename(path), "returns", len(dd),
          "decode-token distance p25/50/75/90", pct(dd),
          "position distance p25/50/75/90", pct(dp),
          "same turn share %.2f" % (same_turn / max(1, len(dd))))
