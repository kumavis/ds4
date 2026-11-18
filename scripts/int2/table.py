#!/usr/bin/env python3
"""Markdown table of kumanix-int2 runs from results.jsonl (latest record per
tag, in time order). Usage: table.py [--results FILE] PREFIX...

misses/tok: decode_misses / ntok_max for plain runs. For DSpark runs
(decode_misses + prefill_misses - prefill misses of the matching plain run) /
ntok_max, because the record's prefill_misses counts every multi-row call,
verify rows included (as scripts/int/table.py). dyn slots: the memory-sized
cache's dynamic slots from the run log ("-" under legacy sizing)."""
import json, os, re, sys

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
    m = re.match(r"int2-mx-(\w+)-dsp-(r\d|pf)$", t)
    if m:
        return f"int2-mx-{m.group(1)}-plain-r1"
    if t == "int2-q2-code-dsp":
        return "int2-q2-code-plain"
    if t.startswith("int2-gate-mx-code-exact"):
        return "int2-gate-mx-code"
    return None


def slots(r):
    d = r.get("run_dir")
    try:
        txt = open(os.path.join(d, "run.log"), errors="replace").read()
    except (OSError, TypeError):
        return None
    m = re.findall(r"dynamic slots (\d+)", txt)
    return int(m[-1]) if m else None


def f(v, fmt):
    return "-" if v is None else format(v, fmt)


print("| tag | prefill tok/s | decode tok/s | misses/tok | dyn slots | DSpark rounds / accepted | avail min MiB | NVMe max C | exit |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for t in sorted(recs, key=lambda k: recs[k]["ts"]):
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
    print(f"| {t} | {f(r.get('prefill_tps'), '.2f')} | {f(r.get('decode_tps'), '.2f')} | "
          f"{f(mpt, '.1f')} | {f(slots(r), 'd')} | {ds} | {f(r.get('avail_min_mb'), 'd')} | "
          f"{f(r.get('nvme_temp_max_c'), 'd')} | {r.get('exit_code')} |")
