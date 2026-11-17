"""D9: results table from results.jsonl for tags matching a prefix.

usage: python3 results.py PREFIX [PREFIX...]
"""
import json
import sys

RES = "/data/ds4-bench/results.jsonl"
pref = sys.argv[1:]
recs = {}
for line in open(RES):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if any(r.get("tag", "").startswith(p) for p in pref):
        recs[r["tag"]] = r
cols = ["decode_tps", "dspark_cycles", "dspark_accepted", "decode_misses", "div_first_div",
        "div_common_prefix", "div_tf_agree", "div_tf_flips", "div_bitexact_frac",
        "avail_min_mb", "nvme_temp_max_c", "exit_code"]
print("| tag | " + " | ".join(cols) + " |")
print("|---|" + "---:|" * len(cols))
for t, r in recs.items():
    vals = []
    for c in cols:
        v = r.get(c)
        vals.append("-" if v is None else (f"{v:.4f}" if isinstance(v, float) and c.startswith("div") else str(v)))
    print(f"| {t} | " + " | ".join(vals) + " |")
