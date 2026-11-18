#!/usr/bin/env python3
"""D15 part B: tables from the d15-q8-* records and run logs.

Usage: partb.py RESULTS.jsonl RUNS_DIR
RUNS_DIR holds <run_dir basename>/run.log copies of the d15 runs.
"""
import json
import os
import re
import sys

TOK = re.compile(r"perf token (\d+) pos \d+ .* id (\d+)")
R = {}
for l in open(sys.argv[1]):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if r["tag"].startswith("d15-q8-"):
        R[r["tag"]] = r
RUNS = sys.argv[2]


def log(t):
    p = os.path.join(RUNS, os.path.basename(R[t]["run_dir"]), "run.log")
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def toks(t):
    return [int(m.group(2)) for m in TOK.finditer(log(t))]


def mean(v):
    return sum(v) / len(v)


print("## Every run\n")
print("| tag | tok/s | output_class | ref_match | ref_line | token digest | rounds/accepted | decode+batch misses | MemAvailable min (MiB) | GTT max (MiB) | NVMe max (°C) |")
print("|---|---:|---|---|---|---|---|---|---:|---:|---:|")
for t, r in R.items():
    print(f"| {t} | {r.get('decode_tps')} | {r.get('output_class')} | {r.get('ref_match')} | {r.get('ref_line')} | "
          f"{r.get('token_digest')}/{r.get('token_count')} | {r.get('dspark_cycles')}/{r.get('dspark_accepted')} | "
          f"{r.get('decode_misses')}+{r.get('prefill_misses')} | {r.get('avail_min_mb')} | {r.get('gtt_max_mb')} | {r.get('nvme_temp_max_c')} |")

print("\n## Labels\n")
on = [t for t in R if "-on" in t or "-tf-" in t or "-fr-" in t]
off = [t for t in R if t not in on]
bad_on = [t for t in on if R[t].get("output_class") != "fp-order"]
bad_off = [t for t in off if R[t].get("output_class") == "fp-order"]
print(f"* switch-on runs: {len(on)}, labelled fp-order: {len(on) - len(bad_on)}; not fp-order: {bad_on or 'none'}")
print(f"* switch-off runs labelled fp-order: {bad_off or 'none'}")
print(f"* ref_match on switch-on runs: {sorted(set(str(R[t].get('ref_match')) for t in on))}; "
      f"on switch-off runs: {sorted(set(str(R[t].get('ref_match')) for t in off))}")
for t in on + off:
    print(f"    {t}: class={R[t].get('output_class')} ref_match={R[t].get('ref_match')} ref_line={R[t].get('ref_line')}")

print("\n## Speed, off vs on (same binary, two runs each, interleaved)\n")
print("| config | off r1 | off r2 | off mean | on r1 | on r2 | on mean | change | MemAvail min off/on | NVMe max off/on | first differing token, on vs off (r1, r2) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for n in ["mx-code", "mx-prose", "mx-hash", "mx-long", "dsp-code", "dsp-json", "q2-code"]:
    tags = {s: [f"d15-q8-{n}-{s}-r{i}" for i in (1, 2)] for s in ("off", "on")}
    if not all(t in R for v in tags.values() for t in v):
        print(f"| {n} | missing |")
        continue
    v = {s: [R[t]["decode_tps"] for t in tags[s]] for s in tags}
    mo, mn = mean(v["off"]), mean(v["on"])
    fd = []
    for i in range(2):
        a, b = toks(tags["off"][i]), toks(tags["on"][i])
        d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if not a and not b:
            r0, r1 = R[tags["off"][i]], R[tags["on"][i]]
            fd.append(f"no per-token ids; rounds/accepted {r0.get('dspark_cycles')}/{r0.get('dspark_accepted')} vs "
                      f"{r1.get('dspark_cycles')}/{r1.get('dspark_accepted')}")
            continue
        fd.append("none" if d is None and len(a) == len(b) else str(d if d is not None else min(len(a), len(b))))
    am = "/".join(str(min(R[t]["avail_min_mb"] for t in tags[s])) for s in ("off", "on"))
    tm = "/".join(str(max(R[t]["nvme_temp_max_c"] for t in tags[s])) for s in ("off", "on"))
    print(f"| {n} | {v['off'][0]:.2f} | {v['off'][1]:.2f} | {mo:.3f} | {v['on'][0]:.2f} | {v['on'][1]:.2f} | {mn:.3f} | "
          f"{100*(mn-mo)/mo:+.1f}% | {am} | {tm} | {', '.join(fd)} |")

print("\n## Divergence against the off reference (record fields from diverge.py)\n")
keys = ["ref_tag", "div_forced", "div_tf_positions", "div_tf_agree", "div_tf_flips", "div_flip_margin_max",
        "div_flip_margin_med", "div_logit_maxabs", "div_logit_meanabs", "div_logit_p99", "div_bitexact_frac",
        "div_first_div", "div_common_prefix", "div_compared", "div_error"]
for t in [x for x in R if "-tf-" in x or "-fr-" in x or "-ref-" in x]:
    print(f"* `{t}` ({R[t].get('decode_tps')} tok/s): " + ", ".join(f"{k}={R[t].get(k)}" for k in keys))

print("\n## C1 coarse profile\n")
for t in ("d15-q8-prof-off", "d15-q8-prof-on"):
    if t in R:
        for line in log(t).splitlines():
            if line.startswith("C1 ") and "C1L" not in line:
                print(f"    {t}: {line}")
