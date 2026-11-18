"""Small-append table from probe sessions (driver step `probe`).

  probes.py RESULTS TAG_PREFIX SETTING [SETTING ...]

For tags TAG_PREFIX-SETTING: one row per probe (in order), per setting the
time to first token, the experts read before the first token (NVMe bytes /
17.93 MiB, includes both drives) and token identity against the twin
(twin_identical_tokens / twin_compared_tokens).
"""
import json, sys

results, prefix, settings = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {s: [] for s in settings}
for l in open(results):
    r = json.loads(l)
    if r.get("kind") != "turn":
        continue
    for s in settings:
        if r.get("tag") == f"{prefix}-{s}" and r.get("probe_kind") == "probe":
            rows[s].append(r)
n = max(len(v) for v in rows.values())
hdr = "| probe | new tokens |" + "".join(f" {s}: TTFT s | {s}: experts read | {s}: identical |" for s in settings)
print(hdr)
print("|" + "---|" * (2 + 3 * len(settings)))
for i in range(n):
    first = next(rows[s][i] for s in settings if i < len(rows[s]))
    line = f"| {i + 1} | {first.get('new_tokens')} |"
    for s in settings:
        if i >= len(rows[s]):
            line += " - | - | - |"
            continue
        r = rows[s][i]
        mib = r.get("est_read_mib_prefill")
        ex = f"{mib / 17.93:.0f}" if mib is not None else "-"
        ident = (f"{r['twin_identical_tokens']}/{r['twin_compared_tokens']}"
                 if r.get("twin_compared_tokens") is not None else "(twin)")
        line += f" {r.get('ttft_s'):.1f} | {ex} | {ident} |"
    print(line)
