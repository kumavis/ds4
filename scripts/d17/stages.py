"""Per-sweep stage totals from DS4_METAL_V41_STAGE_PROFILE (synchronized stages)
plus the layer profile's map (loader wait) and seed, from a server log.

  stages.py SERVER_LOG
"""
import re, sys
from collections import OrderedDict

sx = re.compile(r"V4\.1 stage layer=(\d+) pos=(\d+) rows=(\d+) (.+?)=([\d.]+) ms")
lx = re.compile(r"V4\.1 prefill layer=(\d+) pos=(\d+) rows=(\d+) encoder_resident=\d+ map=([\d.]+) "
                r"engram=([\d.]+) encode=([\d.]+) drain=([\d.]+) seed=([\d.]+) ms")
sw = OrderedDict()
labels = []
for line in open(sys.argv[1], errors="replace"):
    m = sx.search(line)
    if m:
        key = (int(m[2]), int(m[3]))
        d = sw.setdefault(key, {})
        d[m[4]] = d.get(m[4], 0.0) + float(m[5])
        if m[4] not in labels:
            labels.append(m[4])
        continue
    m = lx.search(line)
    if m:
        key = (int(m[2]), int(m[3]))
        d = sw.setdefault(key, {})
        d["map (loader wait)"] = d.get("map (loader wait)", 0.0) + float(m[4])
        d["seed"] = d.get("seed", 0.0) + float(m[8])
cols = ["map (loader wait)"] + labels + ["seed"]
print("| pos | rows | " + " | ".join(cols) + " | total s |")
print("|---|---|" + "---|" * (len(cols) + 1))
for (pos, rows), d in sw.items():
    if sum(d.values()) < 1000:
        continue
    print(f"| {pos} | {rows} | " + " | ".join(f"{d.get(c, 0) / 1000:.1f}" for c in cols) + f" | {sum(d.values()) / 1000:.1f} |")
print("(seconds per sweep, summed over 40 layers)")
