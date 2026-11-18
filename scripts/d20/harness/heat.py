"""Read throughput against drive temperature, from run sys.log files (2 s samples).

Only samples with heavy reads count (both drives busy, sum >= 4000 MB/s:
prefill sweeps, which are read-bound). For each nvme0 (system drive, replica)
temperature bucket: samples, mean MB/s of each drive, and the nvme0 share.
If the hot drive throttled, its MB/s and its share would fall at the top bucket.

  heat.py SYS_LOG [SYS_LOG ...]
"""
import re, statistics, sys

rows = []
for p in sys.argv[1:]:
    for line in open(p, errors="replace"):
        d = dict(re.findall(r"(\w+)=(\S+)", line))
        try:
            rows.append((int(d["nvme0_c"]), int(d["nvme1_c"]), int(d["nvme0_mbs"]), int(d["nvme1_mbs"])))
        except (KeyError, ValueError):
            pass
heavy = [r for r in rows if r[2] + r[3] >= 4000 and r[2] > 0 and r[3] > 0]
print(f"samples {len(rows)}, heavy-read samples {len(heavy)}")
print("| nvme0 C | samples | nvme0 MB/s | nvme1 MB/s | total MB/s | nvme0 share | nvme1 C (mean) |")
print("|---|---|---|---|---|---|---|")
for lo, hi in ((0, 80), (80, 84), (84, 86), (86, 88), (88, 100)):
    b = [r for r in heavy if lo <= r[0] < hi]
    if not b:
        continue
    m0 = statistics.mean(r[2] for r in b); m1 = statistics.mean(r[3] for r in b)
    print(f"| {lo}-{hi - 1} | {len(b)} | {m0:.0f} | {m1:.0f} | {m0 + m1:.0f} | {m0 / (m0 + m1):.3f} | {statistics.mean(r[1] for r in b):.1f} |")
