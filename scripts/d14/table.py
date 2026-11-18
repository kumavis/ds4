#!/usr/bin/env python3
"""D14 tables from /data/ds4-bench/sessions.jsonl (stdlib only).

  table.py [--results FILE] turns TAG [TAG ...]     one row per turn
  table.py [--results FILE] buckets TAG [TAG ...]   turns grouped by context fill
  table.py [--results FILE] scenarios [TAG-PREFIX]  one row per scenario record
"""
import argparse, json, statistics, sys

BUCKETS = [(0, 12000, "<12k"), (12000, 24000, "~16k"), (24000, 48000, "~32k"), (48000, 80000, "~64k"),
           (80000, 115000, "~100k"), (115000, 150000, "~128k"), (150000, 1 << 30, ">150k")]


def load(path):
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def f(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def cmd_turns(recs, tags):
    print("| tag | step | s | turn | prompt | reused | new | TTFT s | prefill s | prefill t/s | reply | decode t/s | miss/tok (first16/rest) | est miss/tok | DSpark acc/round | kv events | avail min MiB | nvme max C |")
    print("|" + "---|" * 18)
    for r in recs:
        if r.get("kind") != "turn" or r.get("tag") not in tags:
            continue
        kv = ",".join(f"{e['type']}{'@' + str(e.get('tokens')) if e.get('tokens') else ''}"
                      f"{'(' + str(round(e['ms'])) + 'ms)' if e.get('ms') else ''}" for e in (r.get("kv_events") or []))
        miss = f"{f(r.get('misses_per_tok'))} ({f(r.get('misses_first16'))}/{f(r.get('misses_rest'))})" if r.get("misses_per_tok") is not None else "-"
        ds = f"{f(r.get('dspark_avg_accept'), 2)}/{r.get('dspark_rounds')}" if r.get("dspark_rounds") else "-"
        print(f"| {r['tag']} | {r.get('step')} | {r.get('session')} | {r.get('turn')} | {f(r.get('prompt_tokens'))} | "
              f"{f(r.get('reused_tokens'))} | {f(r.get('new_tokens'))} | {f(r.get('ttft_s'))} | {f(r.get('prefill_s'))} | "
              f"{f(r.get('prefill_tps'))} | {f(r.get('reply_tokens'))} | {f(r.get('decode_tps'), 2)} | {miss} | "
              f"{f(r.get('est_misses_per_tok'))} | {ds} | {kv or '-'} | {f(r.get('avail_min_mb'))} | {f(r.get('nvme_max_c'))} |")


def cmd_buckets(recs, tags):
    print("| tag | fill | turns | prompt range | TTFT s (med) | new tok (med) | prefill t/s (med) | decode t/s (med) | decode after 16 (med) | miss/tok first16 | miss/tok rest | est miss/tok | DSpark avg accept | avail min MiB |")
    print("|" + "---|" * 14)
    for tag in tags:
        rows = [r for r in recs if r.get("kind") == "turn" and r.get("tag") == tag and r.get("prompt_tokens")]
        for lo, hi, name in BUCKETS:
            b = [r for r in rows if lo <= r["prompt_tokens"] < hi]
            if not b:
                continue
            dec = [r for r in b if (r.get("reply_tokens") or 0) >= 32]
            print(f"| {tag} | {name} | {len(b)} | {min(r['prompt_tokens'] for r in b)}-{max(r['prompt_tokens'] for r in b)} | "
                  f"{f(med([r.get('ttft_s') for r in b]))} | {f(med([r.get('new_tokens') for r in b]))} | "
                  f"{f(med([r.get('prefill_tps') for r in b]))} | {f(med([r.get('decode_tps') for r in dec]), 2)} | "
                  f"{f(med([r.get('decode_tps_after16') for r in dec]), 2)} | {f(med([r.get('misses_first16') for r in dec]))} | "
                  f"{f(med([r.get('misses_rest') for r in dec]))} | {f(med([r.get('est_misses_per_tok') for r in dec]))} | "
                  f"{f(med([r.get('dspark_avg_accept') for r in dec]), 2)} | "
                  f"{f(min([r.get('avail_min_mb') for r in b if r.get('avail_min_mb')], default=None))} |")


def cmd_scen(recs, prefix):
    print("| tag | ctx | sessions | dspark | kv dir | planned GiB | cache slots | cache GiB | listening | startup s | turns | max ctx | exit | stop | avail min MiB | nvme max C | errors |")
    print("|" + "---|" * 17)
    for r in recs:
        if r.get("kind") != "scenario" or not r.get("tag", "").startswith(prefix):
            continue
        st = r.get("stop") or {}
        print(f"| {r['tag']} | {r.get('ctx')} | {r.get('resident_sessions')} | {r.get('dspark')} | {r.get('kv_disk_dir') or '-'} | "
              f"{f(r.get('planned_gib'), 2)} | {f(r.get('cache_slots'))} | {f(r.get('cache_gib'), 2)} | {r.get('listening')} | "
              f"{f(r.get('startup_s'))} | {r.get('turns')} | {f(r.get('max_context'))} | {r.get('exit_code')} | "
              f"{st.get('result')} {f(st.get('exit_wait_s'))}s | {f(r.get('avail_min_mb'))} | {f(r.get('nvme_max_c'))} | "
              f"{'; '.join(r.get('startup_errors') or [])[:200] or '-'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/ds4-bench/sessions.jsonl")
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    recs = load(a.results)
    if a.cmd == "turns":
        cmd_turns(recs, set(a.args))
    elif a.cmd == "buckets":
        cmd_buckets(recs, a.args)
    else:
        cmd_scen(recs, a.args[0] if a.args else "d14-")


if __name__ == "__main__":
    main()
