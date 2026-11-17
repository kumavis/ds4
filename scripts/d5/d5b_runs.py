"""D5b run table: per-token decode breakdown and D5/DSpark stats from run logs,
joined with results.jsonl.  Usage: python3 d5b_runs.py RESULTS.jsonl RUNS_DIR TAG_PREFIX...

Per plain run (perf token lines): ms/token, misses, demand read wait, rest.
Per run: the D5 prefetch line (issued, adopted, dropped, speculative GiB,
demand sets started with speculative chunks in flight), DSpark cycles /
accepted / batch misses, and the record's div_* logits comparison."""
import glob
import json
import os
import re
import sys

res, runs_dir, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
recs = {}
for line in open(res):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if any(r.get("tag", "").startswith(p) for p in prefixes):
        recs[r["tag"]] = r


def num(pat, s, cast=float):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else None


print("| tag | D5 env | tok/s | ms/tok | demand wait ms/tok | rest ms/tok | issued | adopted ready+inflight | dropped | spec GiB | demand sets w/ spec in flight | DSpark cycles/accepted | batch misses | div bitexact / first div | nvme max °C |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag, r in recs.items():
    d = r.get("run_dir") or ""
    try:
        s = open(os.path.join(d, "run.log"), errors="replace").read()
    except OSError:
        s = ""
    rows = [tuple(map(float, m.groups())) for m in re.finditer(
        r"perf token \d+ pos \d+ ms ([0-9.]+) slots \d+ misses (\d+) wait_ms ([0-9.]+)", s)]
    if rows:
        n = len(rows)
        ms = sum(x[0] for x in rows) / n
        w = sum(x[2] for x in rows) / n
        bd = f"{ms:.1f} | {w:.1f} | {ms - w:.1f}"
    else:
        bd = "- | - | -"
    env = r.get("env", {})
    d5 = " ".join(f"{k.replace('DS4_ROCM_D5_', '')}={v}" for k, v in env.items()
                  if k.startswith("DS4_ROCM_D5_") or k == "DS4_ROCM_STREAM_READ_WORKERS")
    pf = re.search(r"D5 prefetch: (.*)", s)
    if pf:
        p = pf.group(1)
        iss = num(r"issued=(\d+)", p, int)
        ar = num(r"adopted_ready=(\d+)", p, int)
        ai = num(r"adopted_inflight=(\d+)", p, int)
        dr = (num(r"dropped_ready=(\d+)", p, int) or 0) + (num(r"dropped_loading=(\d+)", p, int) or 0)
        gib = num(r"spec_read=([0-9.]+)", p)
        ds = num(r"demand_sets=(\d+)", p, int)
        dw = num(r"with_spec_inflight=(\d+)", p, int)
        d5c = f"{iss} | {ar}+{ai} | {dr} | {gib} | {dw}/{ds}"
    else:
        d5c = "- | - | - | - | -"
    cyc = num(r"DSpark stats cycles=(\d+)", s, int)
    acc = num(r"accepted_draft=(\d+)", s, int)
    bm = num(r"batch calls=\d+ unique=\d+ hits=\d+ misses=(\d+)", s, int)
    dv = r.get("div_bitexact_frac")
    fd = r.get("div_first_div")
    print(f"| {tag} | {d5 or '-'} | {r.get('decode_tps')} | {bd} | {d5c} | "
          f"{cyc if cyc is not None else '-'}/{acc if acc is not None else '-'} | {bm if bm is not None else '-'} | "
          f"{dv if dv is not None else '-'} / {fd if fd is not None else '-'} | {r.get('nvme_temp_max_c', r.get('ssd_temp_max_c'))} |")
