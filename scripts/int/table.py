#!/usr/bin/env python3
"""Markdown tables of the kumanix-int runs from results.jsonl (latest record
per tag). Usage: table.py [--results FILE] PREFIX...   (tag prefixes)

misses/tok: stream-cache misses after prefill per generated token =
(decode_misses + prefill_misses - prefill misses of the matching plain run)
/ ntok_max. The record's prefill_misses counts every batch (multi-row) call,
so on DSpark runs it holds prefill plus verify rows; plain runs only prefill.
For plain runs it is decode_misses / ntok_max."""
import json, sys

path = "/data/ds4-bench/results.jsonl"
argv = sys.argv[1:]
if argv and argv[0] == "--results":
    path, argv = argv[1], argv[2:]
recs = {}
for line in open(path):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    t = r.get("tag") or ""
    if any(t.startswith(p) for p in argv):
        recs[t] = r


def plain_of(t):
    for a, b in (("-old-", "-plain-"), ("-online-", "-plain-")):
        if a in t:
            return t.replace(a, b)
    if t.endswith("-tf"):
        return t[:-3] + "-plain-r1"
    if t.startswith("int-q2-code-"):
        return "int-q2-code-plain"
    return None


def f(v, fmt):
    return "-" if v is None else format(v, fmt)


print("| tag | prefill tok/s | decode tok/s | misses/tok | DSpark rounds / accepted drafts | div_tf_agree | div_first_div | ssd max C | exit |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for t in sorted(recs, key=lambda k: (recs[k]["ts"])):
    r = recs[t]
    n = r.get("ntok_max") or 1
    dm, pm = r.get("decode_misses"), r.get("prefill_misses")
    mpt = None
    if dm is not None:
        if r.get("dspark"):
            p = recs.get(plain_of(t) or "")
            if p is not None and pm is not None and p.get("prefill_misses") is not None:
                mpt = (dm + pm - p["prefill_misses"]) / n
        else:
            mpt = dm / n
    ds = "-"
    if r.get("dspark"):
        ds = f"{f(r.get('dspark_cycles'), 'd')} / {f(r.get('dspark_accepted'), 'd')}"
    fd = r.get("div_first_div")
    fd = "-" if r.get("ref_tag") is None else ("none" if fd is None else str(fd))
    print(f"| {t} | {f(r.get('prefill_tps'), '.2f')} | {f(r.get('decode_tps'), '.2f')} | "
          f"{f(mpt, '.1f')} | {ds} | {f(r.get('div_tf_agree'), '.4f')} | {fd} | "
          f"{f(r.get('ssd_temp_max_c'), 'd')} | {r.get('exit_code')} |")
