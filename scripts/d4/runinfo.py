#!/usr/bin/env python3
"""D4: per-run table from results.jsonl plus the run logs.

runinfo.py REGEX [--csv]
Columns: tag, dynamic slots configured (from 'SSD expert storage ... (N experts)'),
resident peak slots (stream cache stats), decode tok/s, decode ms/token,
decode misses per generated token, avail_min_mb, gtt_max_mb, planned GiB,
ssd max C, exit, token-ID digest/lines (plain runs).
Misses per token: plain runs = sum of D8_STEP misses (decode steps only) /
steps; DSpark runs = (selected + batch misses - prompt misses) / ntok, where the
prompt's misses are total misses minus the D8_STEP decode misses of ... not
available, so for DSpark the column is (decode_misses + prefill_misses -
P) / ntok_max with P taken from --prompt-misses (default: the plain code run
at the nearest slot count, see curve.py)."""
import hashlib, json, re, sys

def load(pat):
    out = []
    rx = re.compile(pat)
    for l in open("/data/ds4-bench/results.jsonl"):
        try: r = json.loads(l)
        except Exception: continue
        if rx.search(r.get("tag", "")): out.append(r)
    return out

def parse_log(path):
    info = {"dyn": None, "res": None, "budget": None, "steps": 0, "step_misses": 0,
            "step_ms": 0.0, "digest": None, "lines": 0, "memsized": None, "sizing": None,
            "steps_late": 0, "misses_late": 0, "ms_late": 0.0}
    ids = []
    try: f = open(path, errors="replace")
    except OSError: return info
    for line in f:
        m = re.search(r"dynamic \((\d+) experts\)", line)
        if m and info["dyn"] is None: info["dyn"] = int(m.group(1))
        m = re.search(r"resident current=\d+/[\d.]+ GiB max=(\d+)/[\d.]+ GiB budget=(\d+)", line)
        if m: info["res"], info["budget"] = int(m.group(1)), int(m.group(2))
        if "memory-sized SSD cache" in line: info["sizing"] = "memory"
        for m in re.finditer(r"D8_STEP pos=(\d+) tok=\d+ dspark=(\d) ms=([\d.]+) .*?misses=(\d+)", line):
            info["steps"] += 1; info["step_misses"] += int(m.group(4)); info["step_ms"] += float(m.group(3))
            if info["steps"] > 64:
                info["steps_late"] += 1; info["misses_late"] += int(m.group(4)); info["ms_late"] += float(m.group(3))
        m = re.search(r"perf token (\d+) pos \d+ .* id (\d+)", line)
        if m: ids.append(f"{m.group(1)} {m.group(2)}")
    if ids:
        info["digest"] = hashlib.md5(("\n".join(ids) + "\n").encode()).hexdigest()[:12]
        info["lines"] = len(ids)
    return info

def rows(pat):
    for r in load(pat):
        i = parse_log(r["run_dir"] + "/run.log")
        yield r, i

if __name__ == "__main__":
    pat = sys.argv[1]
    print("| tag | sizing | dyn slots | resident peak | decode tok/s | ms/tok | misses/tok (decode) | misses/tok after 64 | ms/tok after 64 | avail_min_mb | gtt_max_mb | planned GiB | ssd max C | exit | digest / lines |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r, i in rows(pat):
        env = r.get("env") or {}
        sizing = env.get("DS4_ROCM_STREAM_CACHE_SIZING") or i["sizing"] or "legacy"
        tps = r.get("decode_tps")
        if r.get("dspark"):
            mpt = "-"  # computed in curve.py
        else:
            mpt = f'{i["step_misses"]/i["steps"]:.1f}' if i["steps"] else "-"
        ml = f'{i["misses_late"]/i["steps_late"]:.1f}' if i["steps_late"] and not r.get("dspark") else "-"
        msl = f'{i["ms_late"]/i["steps_late"]:.1f}' if i["steps_late"] and not r.get("dspark") else "-"
        dg = f'{i["digest"]} / {i["lines"]}' if i["digest"] else "-"
        print(f'| {r["tag"]} | {sizing} | {i["dyn"]} | {i["res"]} | {tps} | {1000/tps if tps else 0:.1f} | {mpt} | {ml} | {msl} | {r.get("avail_min_mb")} | {r.get("gtt_max_mb")} | {r.get("planned_gib")} | {r.get("ssd_temp_max_c")} | {r.get("exit_code")} | {dg} |')
