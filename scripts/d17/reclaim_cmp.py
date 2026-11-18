"""Partial staging reclaim: per turn of a probe run, union reads, misses per token
in the first 16 and first 64 reply tokens, and the partial-reclaim counters.

  reclaim_cmp.py RUN_DIR [RUN_DIR ...]
"""
import json, os, re, statistics, sys

ux = re.compile(r"union sweep rows=(\d+) layers=(\d+) union_experts=(\d+) read_experts=(\d+) load_ms=([\d.]+)")
px = re.compile(r"partial reclaim lent_resident=(\d+) evicted=(\d+) held=(\d+) violations=(\d+)")
dx = re.compile(r"D8_STEP pos=\d+ tok=\d+ dspark=\d+ ms=[\d.]+ pend_ms=[\d.]+ slots=\d+ misses=(\d+)")
rx = re.compile(r"D13 staging reclaimed for a layer sweep: evicted (\d+) entries")
for rd in sys.argv[1:]:
    print(f"\n{os.path.basename(rd)}")
    print("| turn | new tokens | TTFT s | union rows | union experts | read experts | reads/layer | D13 full-reclaim evicted | lent resident at start | partial evicted | held | violations | misses/tok first 16 | first 64 |")
    print("|" + "---|" * 14)
    log = os.path.join(rd, "server.log")
    for l in open(os.path.join(rd, "turns.jsonl")):
        t = json.loads(l)
        with open(log, "rb") as f:
            f.seek(t["log_off"][0])
            seg = f.read(t["log_off"][1] - t["log_off"][0]).decode("utf-8", "replace")
        u = ux.findall(seg)
        p = px.findall(seg)
        full = rx.findall(seg)
        mis = [int(m) for m in dx.findall(seg)]
        usage = t.get("usage") or {}
        new = (usage.get("prompt_tokens") or 0) - ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        ttft = (t["t_first"] - t["t_send"]) if t.get("t_first") else None
        ur = ",".join(x[0] for x in u) or "-"
        ue = sum(int(x[2]) for x in u) if u else None
        re_ = sum(int(x[3]) for x in u) if u else None
        lay = sum(int(x[1]) for x in u) if u else 0
        pr = p[-1] if p else None
        f16 = f"{statistics.mean(mis[:16]):.1f}" if len(mis) >= 16 else "-"
        f64 = f"{statistics.mean(mis[:64]):.1f}" if len(mis) >= 64 else "-"
        print(f"| {t['turn']} | {new} | {ttft:.1f} | {ur} | {ue if ue is not None else '-'} | {re_ if re_ is not None else '-'} | "
              f"{(re_ / lay) if lay else 0:.0f} | {','.join(full) or '-'} | {pr[0] if pr else '-'} | {pr[1] if pr else '-'} | "
              f"{pr[2] if pr else '-'} | {pr[3] if pr else '-'} | {f16} | {f64} |")
