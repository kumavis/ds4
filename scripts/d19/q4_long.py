"""D19 matched cold-refill sessions. Invoke through q4.sh, including --from TAG."""
from datetime import datetime
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import q4_check
from q4_check import require
import session_qc

ROOT = q4_check.ROOT
CHECKOUT = Path("/data/ds4-d19")
PREFIX = "d19-q4long"
ORDER = [("lru", "a"), ("s3fifo", "a"), ("s3fifo", "b"), ("lru", "b")]
SLOTS = 4500  # Historical memory-sized plain halves had 4591 and 4576 dynamic slots.
KV_MIB = 32768
BINARY_SHA = "b4c2692ed0914e9e22597e6ad0dd09b652b9f9b0c257016e2fba429ca8135884"
PAYLOAD_SHA = "6a55a1f3280c"


def tag(policy, leg):
    return f"{PREFIX}-{policy}-{leg}"


def kvdir(policy):
    return ROOT / "d19-kv" / f"q4long-{policy}"


def tracepath(name):
    return ROOT / "d19-traces" / (name + ".trace")


def load_records():
    return [json.loads(l) for l in (ROOT / "sessions.jsonl").read_text().splitlines()]


def chronology(initial, restored):
    initial_summary, initial_turns, initial_transcript, initial_directory = initial
    restored_summary, restored_turns, restored_transcript, _ = restored
    first = restored_turns[0]
    require(datetime.fromisoformat(initial_summary["ts_end"]) <= datetime.fromisoformat(restored_summary["ts"]),
            "overlapping A/B timestamps")
    require(first["turn"] == initial_turns[-1]["turn"] + 1 and first["norm"] is True, "wrong first continued turn")
    require(restored_transcript["continued_from"] == str(initial_directory / "transcript-grow.json") and
            restored_transcript["turns"][:len(initial_turns)] == initial_transcript["turns"],
            "restored transcript does not reproduce A")
    persisted = initial_summary["max_context"]
    require(first["kv_hit_tokens"] == persisted and first["reused_tokens"] >= persisted,
            "B did not restore A's final persisted prefix")
    require(any(turn["prompt_tokens"] >= 100000 and turn["reply_tokens"] > 0 for turn in restored_turns[1:]),
            "no completed warm turn started at 100000+")


def spec_for(initial):
    if initial is None:
        return "grow:target=66000:label=grow"
    return f"grow:target=106000:from={initial[3]}:label=grow"


def inspect(records, policy, leg, initial=None):
    name = tag(policy, leg)
    summary, turns = session_qc.read(records, name, ROOT / "runs")
    directory = Path(summary["run_dir"])
    transcript = json.loads((directory / "transcript-grow.json").read_text())
    require((leg == "a") == (initial is None), "incorrect preceding session")
    if leg == "a":
        require(turns[0]["turn"] == 0, "initial session omits starting turns")
    require([(step["spec"], step["end"]) for step in summary["steps"]] == [(spec_for(initial), "target")],
            "wrong step or stop reason")
    require(summary["ctx"] == 163840 and summary["payload_sha256"] == PAYLOAD_SHA and not summary["dspark"] and
            summary["quant"] == "mxfp4" and summary["kv_disk_dir"] == str(kvdir(policy)), "session setup")
    flags = summary["server_flags"]
    require(flags[flags.index("--ssd-streaming-cache-experts") + 1] == str(SLOTS) and
            flags[flags.index("--kv-disk-space-mb") + 1] == str(KV_MIB), "slot or KV limits")
    env = summary["env"]
    require(env["DS4_ROCM_STREAM_SEED_POLICY"] == "cold" and env["DS4_ROCM_STREAM_EVICT_POLICY"] == policy and
            env["DS4_D12_CACHE_TRACE"] == str(tracepath(name)), "policy or trace setup")
    require(any(int(position) == summary["max_context"] for _, position in summary["shutdown_persist"]),
            "final persisted prefix missing")
    evidence = summary, turns, transcript, directory
    if initial is not None:
        chronology(initial, evidence)
    return evidence


def validate_trace(name, policy):
    raw = tracepath(name)
    compressed = Path(str(raw) + ".gz")
    require(raw.exists() != compressed.exists(), "expected one trace artifact")
    if raw.exists():
        subprocess.run(["gzip", "--", str(raw)], check=True)
    with gzip.open(compressed, "rt") as stream:
        mark = next(line for line in stream if line.startswith("M ")).split()
    require(int(mark[4]) == SLOTS + 768 and int(mark[6]) == 0, "incorrect initial budget or residual cache")
    command = [sys.executable, "-B", str(CHECKOUT / "scripts/d19/replay.py")]
    if policy == "lru":
        command.append("--lru")
    subprocess.run([*command, str(compressed)], check=True)


def warm_comparison(records):
    metrics = {}
    for policy in ("lru", "s3fifo"):
        _, turns = session_qc.read(records, tag(policy, "b"), ROOT / "runs")
        warm = [turn for turn in turns[1:] if turn["prompt_tokens"] >= 100000]
        require(warm, "no warm reply with 100000 input tokens")
        tokens = sum(turn["reply_tokens"] for turn in warm)
        metrics[policy] = {
            "turn_ids": [turn["turn"] for turn in warm], "reply_tokens": tokens,
            **session_qc.pooled_decode(warm),
            "ttft_sum_s": sum(turn["ttft_s"] for turn in warm), "restored_first_turn_ttft_s": turns[0]["ttft_s"],
        }
    require(metrics["lru"]["turn_ids"] == metrics["s3fifo"]["turn_ids"], "different 100k turn sets")
    print("Matched warm turns starting at 100000+:", json.dumps(metrics, sort_keys=True))


def run(policy, leg):
    name = tag(policy, leg)
    q4_check.fresh(name)
    cache = kvdir(policy)
    cache.mkdir(parents=True, exist_ok=True)
    initial = None
    if leg == "a":
        require(not any(cache.iterdir()), "A requires an empty policy-specific KV directory")
    else:
        initial = inspect(load_records(), policy, "a")
        require(any(cache.glob("*.kv")), "missing A checkpoints")
        validate_trace(tag(policy, "a"), policy)
    # Historical raw A/B traces used 38.7/25.1 MB and KV about 1.8 GiB.
    required_bytes = (KV_MIB + 1024) * 2**20
    for directory in (cache, ROOT / "d19-traces", ROOT / "runs"):
        require(shutil.disk_usage(directory).free >= required_bytes, f"insufficient space on {directory}")
    binary = CHECKOUT / "ds4-server"
    require(sha256(binary.read_bytes()).hexdigest() == BINARY_SHA, "server binary differs from pinned D19")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("DS4_", "D14_", "HSA_"))}
    for key in ("DS4_MODEL", "DS4_ROCM_STREAM_REPLICA"):
        env[key] = os.environ[key]
    env.update(
        DS4_DIR=str(CHECKOUT), D14_PAYLOAD=str(ROOT / "d14-payload-v2.json"), D14_BUDGET_MIN="100",
        DS4_ROCM_STREAM_SEED_POLICY="cold", DS4_ROCM_STREAM_EVICT_POLICY=policy,
        DS4_D12_CACHE_TRACE=str(tracepath(name)), DS4_NOTE=f"D19 matched long trace {policy} {leg}; {SLOTS} dynamic slots",
    )
    require(not (ROOT / "PAUSED").exists(), "PAUSED")
    flags = f"--ssd-streaming-cache-experts {SLOTS} --kv-disk-dir {cache} --kv-disk-space-mb {KV_MIB}"
    command = [str(CHECKOUT / "scripts/d19/run.sh"), name, "s1-long", "163840", flags, spec_for(initial)]
    subprocess.run(command, env=env, check=True)
    records = load_records()
    inspect(records, policy, leg, initial)
    validate_trace(name, policy)
    other = "s3fifo" if policy == "lru" else "lru"
    if any(row.get("tag") == tag(other, leg) for row in records):
        session_qc.compare(records, ROOT / "runs", tag("lru", leg), tag("s3fifo", leg), False)
        if leg == "b":
            warm_comparison(records)


if __name__ == "__main__":
    names = [tag(*pair) for pair in ORDER]
    require(len(sys.argv) <= 2 and (len(sys.argv) == 1 or sys.argv[1] in names), f"expected one of {names}")
    require(os.environ.get("INVOCATION_ID"), "use a dedicated systemd unit via q4.sh")
    first = names.index(sys.argv[1]) if len(sys.argv) == 2 else 0
    for policy, leg in ORDER[first:]:
        run(policy, leg)
