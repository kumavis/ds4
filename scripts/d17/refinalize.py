#!/usr/bin/env python3
"""Re-run driver.py finalize for a run directory whose finalize failed, with
the same arguments and DS4_*/HSA_*/D17_* environment run.sh used (read back
from run.log). Appends to /data/ds4-bench/sessions.jsonl unless --results.

  refinalize.py RUN_DIR SCENARIO [--results FILE]
"""
import datetime as dt, os, re, subprocess, sys

run_dir, scenario = sys.argv[1], sys.argv[2]
results = sys.argv[sys.argv.index("--results") + 1] if "--results" in sys.argv else "/data/ds4-bench/sessions.jsonl"
log = open(os.path.join(run_dir, "run.log")).read()
m = re.search(r"^run: tag=(\S+) scenario=\S+ start=(\S+) ds4_dir=(\S+) commit=(\S*) dirty=(\S+)", log, re.M)
tag, start_iso, ds4_dir, commit, dirty = m.groups()
start = int(dt.datetime.fromisoformat(start_iso).timestamp())
wall = int(re.search(r"^EXIT=-?\d+ wall_s=(\d+)", log, re.M)[1])
model = re.search(r"^run: model=(\S+) payload=(\S+)", log, re.M)
model_path, payload = model[1], model[2]
cmd = re.search(r"^run: cmd=(.*)$", log, re.M)[1].split()
flags = cmd[cmd.index("-m") + 2:]
ctx = flags[flags.index("--ctx") + 1]
envline = re.search(r"^run: env=(.*)$", log, re.M)[1]
env = dict(os.environ)
for k, v in re.findall(r"(?:^| )((?:DS4|HSA|D17|D14)_[A-Z0-9_]+)=(.*?)(?= (?:DS4|HSA|D17|D14)_[A-Z0-9_]+=|$)", envline):
    env[k] = v
note = env.get("DS4_NOTE", "")
here = os.path.dirname(os.path.abspath(__file__))
args = ["python3", os.path.join(here, "driver.py"), "finalize", "--run-dir", run_dir, "--tag", tag,
        "--scenario", scenario, "--note", note, "--ctx", ctx, "--model", model_path, "--payload", payload,
        "--ds4-dir", ds4_dir, "--ds4-commit", commit, "--ds4-dirty", dirty,
        "--start", str(start), "--end", str(start + wall), "--results", results, "--"] + flags
sys.exit(subprocess.call(args, env=env))
