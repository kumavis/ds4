#!/usr/bin/env python3
"""Markdown table of kumanix-int3 runs from results.jsonl (latest record per
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
    m = re.match(r"int3-mx-(\w+)-dsp-(?:on|off)-(r\d)$", t)
    if m:
        return f"int3-mx-{m.group(1)}-plain-off-{m.group(2)}"
    if t.startswith("int3-q2-code-dsp-"):
        return "int3-q2-code-plain-off"
    if t.startswith("int3-gate-pf-mx-code-exact"):
        return "int3-gate-mx-code"
    return None


def runlog(r):
    try:
        return open(os.path.join(r.get("run_dir"), "run.log"), errors="replace").read()
    except (OSError, TypeError):
        return ""


def slots(r):
    m = re.findall(r"dynamic slots (\d+)", runlog(r))
    return int(m[-1]) if m else None


def d5(r):
    txt = runlog(r)
    i = re.findall(r"D5 prefetch: diag.*? issued=(\d+)", txt)
    h = re.findall(r"D5 thermal: .*? held=(\d+) .*?engaged=(\d+)", txt)
    if not i:
        return "-"
    return f"{i[-1]} / {h[-1][0]} ({h[-1][1]})" if h else f"{i[-1]} / -"


def f(v, fmt):
    return "-" if v is None else format(v, fmt)


print("| tag | prefill tok/s | decode tok/s | misses/tok | dyn slots | DSpark rounds / accepted | avail min MiB | NVMe max C | D5 issued / held (engaged) | exit |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
          f"{f(r.get('nvme_temp_max_c'), 'd')} | {d5(r)} | {r.get('exit_code')} |")
