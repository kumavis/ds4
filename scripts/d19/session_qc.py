"""Compare two D19 session runs on matched turns. Use --overlap for an explicitly partial twin."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(tokens):
    return sha256(",".join(tokens).encode()).hexdigest()[:16]


def read(records, tag, runs):
    group = [row for row in records if row.get("tag") == tag]
    totals = [row for row in group if row["kind"] == "scenario"]
    require(len(totals) == 1, f"{tag}: missing/duplicate scenario")
    total = totals[0]
    turns = [row for row in group if row["kind"] == "turn"]
    require(total["turns"] == len(turns) > 0 and len({r["turn"] for r in turns}) == len(turns), f"{tag}: count")
    run = runs / Path(total["run_dir"]).name
    raw = [json.loads(line) for line in (run/"turns.jsonl").read_text().splitlines()]
    require(len(raw) == len(turns) and total["exit_code"] == 0 and not total["driver_exception"] and
            total["ready"] and total["stop"]["result"] == "stopped" and
            re.search(r"^EXIT=0 wall_s=\d+ driver_rc=0$", (run/"run.log").read_text(), re.M), f"{tag}: incomplete driver")
    with (run/"server.log").open("rb") as log:
        for r, t in zip(turns, raw):
            require(r["turn"] == t["turn"] and r["reply_sha"] == t["reply_sha"] and r["http_status"] == 200 and not r["error"],
                    f"{tag}: turn {r['turn']} HTTP/visible reply")
            a,b = t["log_off"]
            log.seek(a)
            ids = re.findall(rb"D8_STEP [^\n]*?\btok=(\d+)", log.read(b-a))
            require(r["d8_steps"] == len(ids) and (digest([x.decode() for x in ids]) if ids else None) == r["reply_tok_sha"],
                    f"{tag}: turn {r['turn']} raw D8 token digest")
    hashes = [r["reply_sha"] for r in turns]
    require(total["reply_shas"] == hashes and total["replies_digest"] == digest(hashes), f"{tag}: reply summary digest")
    require(total["max_context"] == max(r["context_after"] for r in turns), f"{tag}: fill")
    require(total["avail_min_mb"] >= 12288 and 0 < total["nvme_max_c"] < 89, f"{tag}: telemetry")
    require(len(total["steps"]) == 1, f"{tag}: expected one grow step")
    return total, turns


def compare(records, runs, left, right, overlap):
    sa, a = read(records, left, runs)
    sb, b = read(records, right, runs)
    for k in ("payload_sha256", "dspark", "ctx", "quant", "scenario"):
        require(sa[k] == sb[k], f"unmatched setup: {k}")
    require(all(s["steps"][0]["end"] == "target" for s in (sa,sb)) or overlap, "partial session; request --overlap explicitly")
    require(len(a) == len(b) or overlap, "unequal lengths")
    n = min(len(a),len(b))
    for x,y in zip(a[:n],b[:n]):
        for k in ("turn", "prompt_tokens", "event", "reply_tokens", "reply_sha", "reply_tok_sha",
                  "d8_steps", "dspark_rounds", "dspark_accepted"):
            require(x[k] == y[k], f"turn {x['turn']}: {k} differs; matched timing would be confounded")
    out = {"matched_turns": n, "dspark":sa["dspark"]}
    for tag, s, rows in ((left,sa,a),(right,sb,b)):
        matched = rows[:n]
        tok = sum(t["reply_tokens"] for t in matched)
        sec = sum(t["decode_s"] for t in matched)
        out[tag] = dict(turns=len(rows), end=s["steps"][0]["end"], max_context=s["max_context"],
                        dynamic_cache_slots=s["cache_slots"], kv_disk_dir=s["kv_disk_dir"],
                        first_turn=matched[0]["turn"], last_turn=matched[-1]["turn"], matched_reply_tokens=tok,
                        matched_d8_steps=sum(t["d8_steps"] for t in matched), decode_s=round(sec,3), decode_tps=tok/sec,
                        ttft_sum_s=round(sum(t["ttft_s"] for t in matched),3), memory_min_mib=s["avail_min_mb"],
                        nvme_max_c=s["nvme_max_c"], first_kv_hit_tokens=rows[0]["kv_hit_tokens"])
    out["decode_gain_pct"] = 100*(out[right]["decode_tps"]/out[left]["decode_tps"]-1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sessions",type=Path)
    p.add_argument("runs",type=Path)
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--overlap",action="store_true")
    a=p.parse_args()
    compare([json.loads(l) for l in a.sessions.read_text().splitlines()],a.runs,a.left,a.right,a.overlap)
