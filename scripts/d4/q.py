#!/usr/bin/env python3
# Query results.jsonl: q.py REGEX [fields...]
import json, re, sys
pat = re.compile(sys.argv[1])
fields = sys.argv[2:] or ["decode_tps","decode_misses","prefill_misses","ntok_max","avail_min_mb","gtt_max_mb","planned_gib","ssd_temp_max_c","exit_code","prompt_tokens","ctx"]
for l in open("/data/ds4-bench/results.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if pat.search(r.get("tag","")):
        print(r["tag"], *[f"{f}={r.get(f)}" for f in fields])
