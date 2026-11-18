"""Misses per decode step by position in the reply, averaged over turns that
followed a layer sweep (new tokens >= 256), from D8_STEP lines.

  recovery.py RUN_DIR [RUN_DIR ...]
"""
import json, os, re, statistics, sys

BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
acc = {b: [] for b in BUCKETS}
turns = 0
for rd in sys.argv[1:]:
    log = os.path.join(rd, "server.log")
    for l in open(os.path.join(rd, "turns.jsonl")):
        t = json.loads(l)
        u = t.get("usage") or {}
        new = (u.get("prompt_tokens") or 0) - ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if new < 256 or not t.get("log_off"):
            continue
        with open(log, "rb") as f:
            f.seek(t["log_off"][0])
            seg = f.read(t["log_off"][1] - t["log_off"][0]).decode("utf-8", "replace")
        mis = [int(m) for m in re.findall(r"D8_STEP pos=\d+ tok=\d+ dspark=\d+ ms=[\d.]+ pend_ms=[\d.]+ slots=\d+ misses=(\d+)", seg)]
        if len(mis) < 64:
            continue
        turns += 1
        for lo, hi in BUCKETS:
            if len(mis) >= hi:
                acc[(lo, hi)].append(statistics.mean(mis[lo:hi]))
print(f"turns after a sweep with >= 64 reply tokens: {turns}")
print("| reply tokens | turns | misses per token (mean over turns) |")
print("|---|---|---|")
for b in BUCKETS:
    if acc[b]:
        print(f"| {b[0]}-{b[1] - 1} | {len(acc[b])} | {statistics.mean(acc[b]):.1f} |")
