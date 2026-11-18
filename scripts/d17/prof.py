"""Per-sweep breakdown from DS4_METAL_GRAPH_PREFILL_PROFILE lines in a server log.

Line: ds4: V4.1 prefill layer=L pos=P rows=R encoder_resident=E map=.. engram=.. encode=.. drain=.. seed=.. ms
  map    wait for the layer's expert loader (read-bound time not hidden by compute)
  engram+encode+drain   the layer's compute (encode enqueues, drain waits for the GPU)
  seed   end-of-layer cache seeding. On ROCm the encode step does not wait for
         the GPU (drain is 0): the layer's GPU time surfaces at the seed's
         synchronous read-back, so compute is counted as engram+encode+drain+seed
         (the seed's own copies are ~2 GB device-to-device, ~10 ms; estimate).
Union sweeps (D17, DS4_ROCM_V41_UNION_SWEEP) also print
  ds4: V4.1 union sweep rows=R layers=.. union_experts=.. read_experts=.. load_ms=.. ok=1
and their union load time sits inside encode (it runs between router and MoE).

  prof.py SERVER_LOG
"""
import re, statistics, sys
from collections import OrderedDict

rx = re.compile(r"V4\.1 prefill layer=(\d+) pos=(\d+) rows=(\d+) encoder_resident=\d+ map=([\d.]+) "
                r"engram=([\d.]+) encode=([\d.]+) drain=([\d.]+) seed=([\d.]+) ms")
ux = re.compile(r"V4\.1 union sweep rows=(\d+) layers=(\d+) union_experts=(\d+) read_experts=(\d+) load_ms=([\d.]+)")
sweeps = OrderedDict()
unions = []
for line in open(sys.argv[1], errors="replace"):
    m = rx.search(line)
    if m:
        L, pos, rows = int(m[1]), int(m[2]), int(m[3])
        mp, en, ec, dr, sd = map(float, m.groups()[3:])
        key = (pos, rows)
        if L == 0 and key in sweeps and len(sweeps[key]) >= 40:
            key = (pos, rows, len(sweeps))
        sweeps.setdefault(key, []).append((L, mp, en + ec + dr + sd, 0.0, ec, dr))
        continue
    u = ux.search(line)
    if u:
        unions.append(tuple(float(x) for x in u.groups()))
print("| sweep pos | rows | layer records | wall s (sum) | loader wait s | compute s (incl. seed) | compute ms/layer | wait ms/layer |")
print("|---|---|---|---|---|---|---|---|")
for key, ls in sweeps.items():
    mp = sum(x[1] for x in ls) / 1000; cp = sum(x[2] for x in ls) / 1000; sd = sum(x[3] for x in ls) / 1000
    print(f"| {key[0]} | {key[1]} | {len(ls)} | {mp + cp:.1f} | {mp:.1f} | {cp:.1f} | "
          f"{1000 * cp / len(ls):.0f} | {1000 * mp / len(ls):.0f} |")
if unions:
    print("\n| union sweep rows | layers | union experts | read experts | load s |")
    print("|---|---|---|---|---|")
    for r, l, ux_, rd, ms in unions:
        print(f"| {r:.0f} | {l:.0f} | {ux_:.0f} | {rd:.0f} | {ms / 1000:.1f} |")
