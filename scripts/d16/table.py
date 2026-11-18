#!/usr/bin/env python3
"""D16 tables from results.jsonl (latest record per tag) plus each run log.
Usage: table.py PREFIX...
Columns: decode tok/s; misses per token (plain: decode_misses / ntok_max;
DSpark: verify batch misses per verified row from the D16 line); output class
and ref_match; divergence against the record's DS4_REF_TAG run (first
differing generated token, common prefix / compared, argmax agreement and
largest logit difference over the common prefix); positional token match
fraction (plain runs: perf-token ids against the reference run, same index);
D16 swap statistics over committed rows (share of row-layers whose chosen set
differs from the unbiased top-6, swapped picks per row-layer, dropped
unbiased ranks 4/5/6 (1-3 are protected with keep 3), mean drift = weight
share of the dropped exact picks); MemAvailable minimum and hottest NVMe."""
import json, os, re, sys

recs = {}
for line in open("/data/ds4-bench/results.jsonl"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    t = r.get("tag") or ""
    if any(t.startswith(p) for p in sys.argv[1:]):
        recs[t] = r


def log(r):
    try:
        return open(os.path.join(r["run_dir"], "run.log"), errors="replace").read()
    except Exception:
        return ""


TOK = re.compile(r"perf token (\d+) pos \d+ .* id (\d+)")


def toks(txt):
    return [int(m.group(2)) for m in TOK.finditer(txt)]


def f(v, fmt):
    return "-" if v is None else format(v, fmt)


print("| tag | class | tok/s | misses/tok (plain) or per verified row (DSpark) | DSpark rounds / accepted | first div | common / compared | token match | prefix argmax agree | logit maxabs (prefix) | swapped row-layers | swapped picks / row-layer | dropped rank 4/5/6 | drift | avail min MiB | NVMe max C |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for t in sorted(recs, key=lambda k: recs[k]["ts"]):
    r = recs[t]
    txt = log(r)
    n = r.get("ntok_max") or 1
    if r.get("dspark"):
        m = re.findall(r"verify_batch_misses=\d+ \(([0-9.]+) per verified row\)", txt)
        mpt = float(m[-1]) if m else None
        ds = "%s / %s" % (r.get("dspark_cycles"), r.get("dspark_accepted"))
    else:
        mpt = r["decode_misses"] / n if r.get("decode_misses") is not None else None
        ds = "-"
    kind = r"verify \(committed rows\)" if r.get("dspark") else "decode"
    sw = re.findall(r"D16 route prior " + kind + r": .*? swapped_row_layers=\d+ \(([0-9.]+)\) "
                    r"swapped_picks=\d+ \(([0-9.]+) per row-layer\) by_unbiased_rank=([0-9/]+) "
                    r"drift_mean=([0-9.]+)", txt)
    if sw:
        s1, s2, ranks, dr = sw[-1]
        rk = "/".join(ranks.split("/")[3:])
    else:
        s1 = s2 = rk = dr = "-"
    ref = r.get("ref_tag")
    match = "-"
    if ref and ref in recs and not r.get("dspark"):
        a, b = toks(txt), toks(log(recs[ref]))
        k = min(len(a), len(b))
        if k:
            match = "%.3f" % (sum(1 for i in range(k) if a[i] == b[i]) / k)
    fd = "-" if not ref else ("none" if r.get("div_first_div") is None else str(r.get("div_first_div")))
    cc = "-" if not ref else "%s / %s" % (r.get("div_common_prefix"), r.get("div_compared"))
    rm = "" if r.get("ref_match") is None else " (ref_match %s)" % r.get("ref_match")
    print("| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        t, r.get("output_class"), rm, f(r.get("decode_tps"), ".2f"), f(mpt, ".2f"), ds, fd, cc, match,
        f(r.get("div_tf_agree"), ".4f") if ref else "-",
        f(r.get("div_logit_maxabs"), ".3f") if ref else "-",
        s1, s2, rk, dr, f(r.get("avail_min_mb"), "d"), f(r.get("nvme_temp_max_c"), "d")))
