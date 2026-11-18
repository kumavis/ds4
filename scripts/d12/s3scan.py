#!/usr/bin/env python3
"""D12: S3-FIFO small-queue share and slot-count robustness scan.
Usage: s3scan.py RESULTS.jsonl TRACE.gz ..."""
import json, os, sys
import sim
rec = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rec[r["tag"]] = r
for path in sys.argv[2:]:
    tag = os.path.basename(path).split(".trace")[0]
    st = sim.Stream(path, rec[tag]["prompt_tokens"])
    cap0 = st.capacity_runtime
    out = []
    for cap in (cap0 - 600, cap0, cap0 + 500):
        base = sim.summarize(st, sim.replay(st, sim.LRU(cap)))
        for sm in (0.05, 0.1, 0.2, 0.3):
            r = sim.summarize(st, sim.replay(st, sim.S3FIFO(cap, sm)))
            cells = []
            for b in ("1-128", "129-512", "513-2048"):
                if b in r["buckets"]:
                    a, z = r["buckets"][b]["per_token"], base["buckets"][b]["per_token"]
                    cells.append(f"{b}:{100*(a-z)/z:+.1f}")
            out.append(f"  cap {cap} small {sm}: " + " ".join(cells))
    print(tag.replace("d12-tr-", ""))
    print("\n".join(out))
