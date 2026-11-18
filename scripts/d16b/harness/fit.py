"""Prefill time of a turn against its new tokens (plain long session, live reuse)."""
import json, statistics
rows = [json.loads(l) for l in open("/data/ds4-bench/d14-merged.jsonl")]
pts = [(r["new_tokens"], r["prefill_s"], r["prompt_tokens"]) for r in rows
       if r["tag"] == "plain" and r.get("new_tokens") and r.get("prefill_s") and (r.get("reused_tokens") or 0) > 0]
bins = [(0, 64), (64, 512), (512, 2048), (2048, 4096), (4096, 8192), (8192, 40000)]
print("| new tokens | turns | prefill s (median) | tok/s (median) |")
print("|---|---|---|---|")
for lo, hi in bins:
    b = [p for p in pts if lo <= p[0] < hi]
    if b:
        print(f"| {lo}-{hi} | {len(b)} | {statistics.median(p[1] for p in b):.1f} | {statistics.median(p[0] / p[1] for p in b):.1f} |")
big = [p for p in pts if p[0] > 300]
n = len(big); mx = sum(p[0] for p in big) / n; my = sum(p[1] for p in big) / n
sl = sum((p[0] - mx) * (p[1] - my) for p in big) / sum((p[0] - mx) ** 2 for p in big)
print(f"fit (new > 300, n={n}): prefill_s = {my - sl * mx:.1f} + {sl * 1000:.2f} s per 1k new tokens")
tot_pre = sum(r["prefill_s"] or 0 for r in rows if r["tag"] == "plain")
tot_dec = sum(r["decode_s"] or 0 for r in rows if r["tag"] == "plain")
print(f"plain session totals: prefill {tot_pre/60:.1f} min, decode {tot_dec/60:.1f} min, share prefill {tot_pre/(tot_pre+tot_dec):.0%}")
