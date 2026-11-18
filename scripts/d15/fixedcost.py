#!/usr/bin/env python3
"""D15: separate compute from read cost in the off/on pairs (estimate).
Per run: ms/token = 1000/tok/s; misses per generated token = decode misses
over (prompt + generated) steps scaled to the generated steps is not
separable, so use the pair difference: d(ms/token) - 1.44 x d(misses)/N,
N = generated tokens (the prompt part is the same in both runs up to the
first differing token).  1.44 ms per miss (D4)."""
import json, sys
R = {json.loads(l)["tag"]: json.loads(l) for l in open(sys.argv[1])}
print("| config | ms/token off | ms/token on | misses off | misses on | d ms/token | read part (1.44 ms x d misses/N) | compute part (estimate) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for n in ["mx-code", "mx-prose", "mx-hash", "mx-long", "q2-code"]:
    rows = []
    for i in (1, 2):
        a, b = R[f"d15-q8-{n}-off-r{i}"], R[f"d15-q8-{n}-on-r{i}"]
        N = a.get("token_count") or a["ntok_max"]
        ma, mb = 1000 / a["decode_tps"], 1000 / b["decode_tps"]
        rd = 1.44 * (b["decode_misses"] - a["decode_misses"]) / N
        rows.append((ma, mb, a["decode_misses"], b["decode_misses"], mb - ma, rd, mb - ma - rd))
    m = [sum(x[j] for x in rows) / 2 for j in range(7)]
    print(f"| {n} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.0f} | {m[3]:.0f} | {m[4]:+.2f} | {m[5]:+.2f} | {m[6]:+.2f} |")
