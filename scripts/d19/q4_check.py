"""D19 queue 4 record gates. Runs only on the CPU, between GPU clients."""
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path("/data/ds4-bench")


def require(value, message):
    if not value:
        raise ValueError(message)


def records(name, tag):
    with (ROOT / name).open() as stream:
        return [r for line in stream if (r := json.loads(line)).get("tag") == tag]


def telemetry(row, temp):
    for key in ("avail_min_mb", temp):
        require(isinstance(row.get(key), (float, int)) and math.isfinite(row[key]), f"missing {key}")
    require(row["avail_min_mb"] >= 12288, f"MemAvailable {row['avail_min_mb']} MiB")
    require(0 < row[temp] < 89, f"NVMe {row[temp]} C")


def fresh(tag):
    require(not list((ROOT / "runs").glob("*-" + tag)), f"run directory already exists for {tag}")
    for name in ("results.jsonl", "sessions.jsonl"):
        require(not records(name, tag), f"{tag} already recorded in {name}")
    trace = ROOT / "d19-traces" / (tag + ".trace")
    require(not trace.exists() and not trace.with_suffix(".trace.gz").exists(), f"trace already exists for {tag}")


def check(tag, spec=None):
    paths = list((ROOT / "runs").glob("*-" + tag))
    require(len(paths) == 1, f"expected unique run directory for {tag}")
    run = paths[0]
    log = (run / "run.log").read_text()
    if spec is None:
        found = records("results.jsonl", tag)
        require(len(found) == 1, f"missing or duplicated result for {tag}")
        row = found[0]
        telemetry(row, "nvme_temp_max_c")
        require(re.search(r"^EXIT=0 wall_s=\d+$", log, re.M), "raw benchmark exit")
    else:
        found = records("sessions.jsonl", tag)
        summaries = [r for r in found if r.get("kind") == "scenario"]
        turns = [r for r in found if r.get("kind") == "turn"]
        require(len(summaries) == 1, "missing or duplicated scenario")
        row = summaries[0]
        telemetry(row, "nvme_max_c")
        require(re.search(r"^EXIT=0 wall_s=\d+ driver_rc=0$", log, re.M), "server or driver exit")
        require(row["ready"] and not row["driver_exception"] and row["stop"]["result"] == "stopped", "driver failed")
        require(len(turns) == row["turns"] > 0 and row["max_context"] > 0, "incomplete turn records")
        require(len(row["steps"]) == 1 and row["steps"][0]["spec"] == spec and row["steps"][0]["end"] == "target",
                f"requested target was not reached: {row['steps']}")
        if "-trace-" in tag:
            require((ROOT / "d19-traces" / (tag + ".trace")).stat().st_size > 0, "missing trace")
        if tag.endswith("-trace-a"):
            transcript = json.loads((run / "transcript-grow.json").read_text())
            require(len(transcript["turns"]) == len(turns), "incomplete restore transcript")
            require(any(int(pos) >= row["max_context"] - 1 for _, pos in row["shutdown_persist"]), "missing final KV save")
        if tag.endswith("-trace-b"):
            first = min(turns, key=lambda r: r["turn"])
            require((first.get("kv_hit_tokens") or 0) > 0, "first turn did not restore disk KV")
    require(row["exit_code"] == 0 and row["run_dir"] == str(run), "record exit or run_dir mismatch")
    print(run)


if __name__ == "__main__":
    action, tag, *args = sys.argv[1:]
    if action == "fresh":
        fresh(tag)
    elif action == "check":
        check(tag, *args)
    else:
        raise SystemExit("expected fresh or check")
