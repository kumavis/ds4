"""Compare raw D19 session evidence and timings for matched turns."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(values):
    return sha256(",".join(values).encode()).hexdigest()[:16]


def validate_turn(record, raw, transcript_turn, log, previous_end, log_size, dspark):
    turn = record["turn"]
    require(turn == raw["turn"] == transcript_turn["turn"], f"turn {turn}: turn ID")
    require(raw["http_status"] == record["http_status"] == 200 and not raw["error"] and not record["error"],
            f"turn {turn}: HTTP status or error")
    require(raw["finish"] == record["finish"] and raw["finish"] in {"stop", "tool_calls", "length"},
            f"turn {turn}: finish reason")

    usage = raw["usage"]
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    require(prompt == record["prompt_tokens"] == transcript_turn["prompt_tokens"], f"turn {turn}: prompt usage")
    require(completion == record["reply_tokens"] == transcript_turn["completion_tokens"] and completion > 0,
            f"turn {turn}: completion usage")
    require(prompt + completion == usage["total_tokens"] == record["context_after"], f"turn {turn}: total usage")
    text_hash = sha256(transcript_turn["reply"].encode()).hexdigest()[:16]
    require(text_hash == transcript_turn["reply_sha"] == raw["reply_sha"] == record["reply_sha"],
            f"turn {turn}: visible reply digest")

    start, end = raw["log_off"]
    require(0 <= previous_end <= start < end <= log_size, f"turn {turn}: invalid/overlapping log offsets")
    log.seek(start)
    segment = log.read(end - start)
    steps = re.findall(rb"D8_STEP\s+pos=(\d+)\s+tok=(\d+)\b", segment)
    require(len(steps) == segment.count(b"D8_STEP ") == record["d8_steps"], f"turn {turn}: malformed D8 steps")
    positions = [int(pos) for pos, _ in steps]
    if dspark:
        require(all(prompt <= pos < prompt + completion for pos in positions) and positions == sorted(set(positions)),
                f"turn {turn}: invalid DSpark step positions")
    else:
        require(positions == list(range(prompt, prompt + completion)), f"turn {turn}: incomplete plain D8 positions")
    token_hash = digest([token.decode() for _, token in steps]) if steps else None
    require(token_hash == record["reply_tok_sha"], f"turn {turn}: raw D8 token digest")
    return end, [(int(pos), int(token)) for pos, token in steps]


def read(records, tag, runs):
    group = [row for row in records if row.get("tag") == tag]
    summaries = [row for row in group if row["kind"] == "scenario"]
    require(len(summaries) == 1, f"{tag}: missing/duplicate scenario")
    summary = summaries[0]
    turns = [row.copy() for row in group if row["kind"] == "turn"]
    require(summary["turns"] == len(turns) > 0, f"{tag}: invalid count")
    turn_ids = [turn["turn"] for turn in turns]
    require(turn_ids == list(range(turn_ids[0], turn_ids[0] + len(turns))), f"{tag}: nonconsecutive turns")

    directory = runs / Path(summary["run_dir"]).name
    raw_turns = [json.loads(line) for line in (directory / "turns.jsonl").read_text().splitlines()]
    driver = json.loads((directory / "driver.json").read_text())
    stop = driver["stop"]
    run_log = (directory / "run.log").read_text()
    require(len(raw_turns) == len(turns) and summary["exit_code"] == 0 and not summary["driver_exception"],
            f"{tag}: incomplete session records")
    require(driver["ready"] and summary["ready"] and stop == summary["stop"] and stop["result"] == "stopped" and
            stop["comm"] == "ds4-server" and stop["sigterm_ts"] and
            re.search(r"^EXIT=0 wall_s=\d+ driver_rc=0$", run_log, re.M) and
            f"sending one SIGTERM to {stop['pid']}" in run_log and "after SIGTERM" in run_log,
            f"{tag}: clean stop unverified")
    require(len(summary["steps"]) == 1 and summary["steps"] == driver["steps"], f"{tag}: driver step mismatch")
    label = summary["steps"][0]["label"]
    transcript = json.loads((directory / f"transcript-{label}.json").read_text())
    transcript_turns = transcript["turns"]
    require([t["turn"] for t in transcript_turns] == list(range(turn_ids[-1] + 1)), f"{tag}: transcript chronology")

    server_log = directory / "server.log"
    log_size = server_log.stat().st_size
    previous_end = 0
    with server_log.open("rb") as log:
        for record, raw in zip(turns, raw_turns):
            previous_end, record["raw_d8"] = validate_turn(record, raw, transcript_turns[record["turn"]], log, previous_end, log_size,
                                         summary["dspark"])
    hashes = [turn["reply_sha"] for turn in turns]
    require(summary["reply_shas"] == hashes and summary["replies_digest"] == digest(hashes),
            f"{tag}: scenario reply digest")
    require(summary["max_context"] == max(turn["context_after"] for turn in turns), f"{tag}: maximum fill")
    require(summary["avail_min_mb"] >= 12288 and 0 < summary["nvme_max_c"] < 89, f"{tag}: telemetry")
    return summary, turns


def arm_metrics(summary, turns, count):
    matched = turns[:count]
    tokens = sum(turn["reply_tokens"] for turn in matched)
    steps = sum(turn["d8_steps"] for turn in matched)
    seconds = sum(turn["decode_s"] for turn in matched)
    return {
        "turns": len(turns), "end": summary["steps"][0]["end"], "max_context": summary["max_context"],
        "dynamic_cache_slots": summary["cache_slots"], "kv_disk_dir": summary["kv_disk_dir"],
        "first_turn": matched[0]["turn"], "last_turn": matched[-1]["turn"], "matched_reply_tokens": tokens,
        "matched_d8_steps": steps, "token_id_coverage": "complete" if steps == tokens else "partial",
        "decode_s": round(seconds, 3), "decode_tps": tokens / seconds,
        "ttft_sum_s": round(sum(turn["ttft_s"] for turn in matched), 3),
        "memory_min_mib": summary["avail_min_mb"], "nvme_max_c": summary["nvme_max_c"],
        "first_kv_hit_tokens": turns[0]["kv_hit_tokens"],
    }


def compare(records, runs, left, right, overlap):
    left_summary, left_turns = read(records, left, runs)
    right_summary, right_turns = read(records, right, runs)
    for key in ("payload_sha256", "dspark", "ctx", "quant", "scenario"):
        require(left_summary[key] == right_summary[key], f"unmatched setup: {key}")
    allowed = {"target", "mem-guard"} if overlap else {"target"}
    for summary in (left_summary, right_summary):
        require(summary["steps"][0]["end"] in allowed, f"unexpected stop: {summary['steps'][0]['end']}")
    require(len(left_turns) == len(right_turns) or overlap, "unequal session lengths")
    count = min(len(left_turns), len(right_turns))
    for first, second in zip(left_turns, right_turns):
        for key in ("turn", "prompt_tokens", "event", "reply_tokens", "reply_sha", "reply_tok_sha",
                    "raw_d8", "d8_steps", "dspark_rounds", "dspark_accepted"):
            require(first[key] == second[key], f"turn {first['turn']}: {key} differs")
    output = {
        "matched_turns": count, "dspark": left_summary["dspark"],
        left: arm_metrics(left_summary, left_turns, count), right: arm_metrics(right_summary, right_turns, count),
    }
    output["decode_gain_pct"] = 100 * (output[right]["decode_tps"] / output[left]["decode_tps"] - 1)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--overlap", action="store_true", help="allow only target and memory-guard terminations")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.sessions.read_text().splitlines()]
    compare(records, args.runs, args.left, args.right, args.overlap)
