"""Merge the long-session turn records into two series (plain, dspark) for table.py buckets.
DSpark A is taken from its re-parse (reparsed.jsonl: the DSpark round lines were
parsed after a parser fix); every other row comes from sessions.jsonl unchanged."""
import glob, json
out = open("/data/ds4-bench/d14-merged.jsonl", "w")
src = {"d14-s1-mx-plain-long-a": "plain", "d14-s1-mx-plain-long-b": "plain",
       "d14-s1-mx-dspark-long-b": "dspark", "d14-s1-q2-plain-32k": "q2-plain",
       "d14-s1-mx-plain-32k-pc8192": "plain-pc8192", "d14-s1-mx-plain-32k-pc4096": "plain-pc4096"}
for l in open("/data/ds4-bench/sessions.jsonl"):
    r = json.loads(l)
    if r.get("kind") == "turn" and r.get("tag") in src:
        r["tag"] = src[r["tag"]]
        out.write(json.dumps(r) + "\n")
for p in glob.glob("/data/ds4-bench/runs/*-d14-s1-mx-dspark-long-a/reparsed.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r.get("kind") == "turn":
            r["tag"] = "dspark"
            out.write(json.dumps(r) + "\n")
out.close()
print("ok")
