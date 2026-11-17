"""D5 results table from /data/ds4-bench/results.jsonl and the queue logs.

Usage: python3 results.py RESULTS.jsonl QUEUE.log [QUEUE.log ...] TAG_PREFIX [...]
Prints one markdown row per record whose tag starts with a prefix (last
record per tag), with the token-ID digest the queue logged for it, and the
on/off means per prompt."""
import json
import re
import sys

args = sys.argv[1:]
res_path = args[0]
logs = [a for a in args[1:] if a.endswith(".log")]
prefixes = [a for a in args[1:] if not a.endswith(".log")]
digest = {}
for lp in logs:
    for line in open(lp):
        m = re.search(r"end (\S+) .* tps=(\S+) digest=(\S+)", line)
        if m:
            digest[m.group(1)] = m.group(3)
recs = {}
for line in open(res_path):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if any(r.get("tag", "").startswith(p) for p in prefixes):
        recs[r["tag"]] = r
print("| tag | commit | D5 env | decode tok/s | prefill tok/s | decode misses | read GiB | read wait s | SSD max °C | avail min MB | digest / lines |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for tag, r in recs.items():
    env = r.get("env", {})
    d5 = " ".join(f"{k.replace('DS4_ROCM_D5_', '')}={v}" for k, v in env.items() if k.startswith("DS4_ROCM_D5_"))
    print(f"| {tag} | {r.get('ds4_commit', '')[:7]}{'*' if r.get('ds4_dirty') else ''} | {d5 or '-'} | "
          f"{r.get('decode_tps')} | {r.get('prefill_tps')} | {r.get('decode_misses')} | {r.get('read_gib')} | "
          f"{(r.get('read_wait_ms') or 0) / 1000:.1f} | {r.get('ssd_temp_max_c')} | {r.get('avail_min_mb')} | "
          f"{digest.get(tag, '?')} |")
print()
groups = {}
for tag, r in recs.items():
    m = re.match(r"(d5[a-z]*-[a-z0-9]+-[a-z]+)-(on|off)-r\d$", tag)
    if m:
        groups.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(r["decode_tps"])
if groups:
    print("| prompt | off tok/s (runs) | on tok/s (runs) | on/off |")
    print("|---|---|---|---|")
    for g, d in groups.items():
        off = d.get("off", [])
        on = d.get("on", [])
        if off and on:
            mo, mn = sum(off) / len(off), sum(on) / len(on)
            print(f"| {g} | {mo:.2f} ({', '.join(map(str, off))}) | {mn:.2f} ({', '.join(map(str, on))}) | {(mn / mo - 1) * 100:+.1f}% |")
