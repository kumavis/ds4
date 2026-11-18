#!/usr/bin/env python3
"""D13: one table row per run, and off/on pair means.

Usage: runinfo.py RESULTS.jsonl RUNS_DIR TAG_REGEX
RUNS_DIR holds /data/ds4-bench/runs/<UTC>-<tag>/run.log copies (same layout).

misses/tok: plain = mean of per-step D8_STEP misses (decode only, as in D4);
DSpark = (selected + batch misses) / tokens generated (includes the misses of a
token-major prompt, which is identical off and on).
budget = dynamic slots (the exit line; the lent 768 are already handed back at
close); peak = the most resident experts seen ("max=").
"""
import hashlib, json, os, re, sys
from collections import defaultdict

res, runs, pat = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
recs = {}
for line in open(res):
    r = json.loads(line)
    if pat.search(r["tag"]):
        recs[r["tag"]] = r  # latest record per tag

STAT = re.compile(r"selected calls=\d+ slots=\d+ hits=\d+ misses=(\d+); batch calls=\d+ unique=\d+ "
                  r"hits=\d+ misses=(\d+).*?resident current=\d+/[\d.]+ GiB max=(\d+)/[\d.]+ GiB budget=(\d+)")


def info(tag, r):
    log = os.path.join(runs, os.path.basename(r["run_dir"]), "run.log")
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    toks = re.findall(r"perf token (\d+) pos \d+ .* id (\d+)", txt)
    dig = None
    if toks:
        s = "".join(f"{a} {b}\n" for a, b in toks)
        dig = f"{hashlib.md5(s.encode()).hexdigest()[:12]}/{len(toks)}"
    st = STAT.findall(txt)
    sel, bat, peak, bud = (int(x) for x in st[-1]) if st else (None,) * 4
    lent = len(re.findall(r"D13 staging lent", txt))
    if r.get("dspark"):
        n = r.get("ntok_max") or 512
        mpt = (sel + bat) / n if sel is not None else None
    else:
        steps = [int(x) for x in re.findall(r"D8_STEP pos=\d+ .*? misses=(\d+)", txt)]
        mpt = sum(steps) / len(steps) if steps else None
    return dict(tag=tag, tps=r.get("decode_tps"), mpt=mpt, budget=bud, peak=peak, lent=lent,
                rounds=r.get("dspark_cycles"), acc=r.get("dspark_accepted"), dig=dig,
                ref_match=r.get("ref_match"), avail=r.get("avail_min_mb"),
                nvme=r.get("nvme_temp_max_c"), exit=r.get("exit_code"),
                bitexact=r.get("div_bitexact_frac"), first_div=r.get("div_first_div"),
                compared=r.get("div_compared"), ts=r["ts"])


rows = [info(t, r) for t, r in sorted(recs.items(), key=lambda kv: kv[1]["ts"])]
f = lambda v, p=2: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
print("| tag | tok/s | misses/tok | dyn slots | peak resident | rounds/acc | digest | ref_match | avail_min_mb | nvme max C | exit |")
print("|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|")
for x in rows:
    ra = f"{x['rounds']}/{x['acc']}" if x["rounds"] is not None else "-"
    print(f"| {x['tag']} | {f(x['tps'])} | {f(x['mpt'])} | {f(x['budget'])} | {f(x['peak'])} | {ra} | "
          f"{x['dig'] or '-'} | {f(x['ref_match'])} | {f(x['avail'])} | {f(x['nvme'])} | {f(x['exit'])} |")

# off/on pairs: tag ...-off-rN / ...-on-rN
pairs = defaultdict(lambda: {"off": [], "on": []})
for x in rows:
    m = re.match(r"(d13-.*)-(off|on)(?:-r\d)?$", x["tag"])
    if m and not x["tag"].startswith(("d13-fact", "d13-smoke", "d13-gate")):
        pairs[m.group(1)][m.group(2)].append(x)
print()
print("| line | off tok/s (runs = mean) | on tok/s (runs = mean) | change | misses/tok off -> on | peak resident off -> on |")
print("|---|---|---|---:|---|---|")
for k, v in pairs.items():
    if not v["off"] or not v["on"]:
        continue
    mo = sum(x["tps"] for x in v["off"]) / len(v["off"])
    mn = sum(x["tps"] for x in v["on"]) / len(v["on"])
    so = ", ".join(f(x["tps"]) for x in v["off"])
    sn = ", ".join(f(x["tps"]) for x in v["on"])
    mpo = sum(x["mpt"] for x in v["off"]) / len(v["off"])
    mpn = sum(x["mpt"] for x in v["on"]) / len(v["on"])
    po = "/".join(str(x["peak"]) for x in v["off"])
    pn = "/".join(str(x["peak"]) for x in v["on"])
    print(f"| {k} | {so} = {mo:.3f} | {sn} = {mn:.3f} | {100 * (mn / mo - 1):+.1f}% | {mpo:.2f} -> {mpn:.2f} | {po} -> {pn} |")
