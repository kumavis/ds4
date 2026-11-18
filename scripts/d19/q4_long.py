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


def chronology(a, b):
    sa, ta, xa, pa = a
    sb, tb, xb, _ = b
    require(datetime.fromisoformat(sa["ts_end"]) <= datetime.fromisoformat(sb["ts"]), "overlapping A/B timestamps")
    require(tb[0]["turn"] == ta[-1]["turn"]+1 and tb[0]["norm"] is True, "wrong first continued turn")
    require(xb["continued_from"] == str(pa/"transcript-grow.json") and xb["turns"][:len(ta)] == xa["turns"],
            "restored transcript does not reproduce A")
    require(tb[0]["kv_hit_tokens"] == sa["max_context"] and tb[0]["reused_tokens"] >= sa["max_context"],
            "B did not restore A's final persisted prefix")
    require(any(t["prompt_tokens"] >= 100000 and t["reply_tokens"] > 0 for t in tb), "no completed turn started at 100000+")


def inspect(records, policy, leg, a=None):
    name=tag(policy,leg)
    summary, turns=session_qc.read(records,name,ROOT/"runs")
    path=Path(summary["run_dir"])
    tr=json.loads((path/"transcript-grow.json").read_text())
    begin=0 if leg=="a" else a[1][-1]["turn"]+1
    require([r["turn"] for r in turns] == list(range(begin,begin+len(turns))), "missing or repeated turn")
    require(len(tr["turns"]) == begin+len(turns), "transcript size")
    for r in turns:
        t=tr["turns"][r["turn"]]
        require(t["turn"]==r["turn"] and t["reply_sha"]==r["reply_sha"] and
                sha256(t["reply"].encode()).hexdigest()[:16]==r["reply_sha"], "transcript reply")
    want="grow:target=66000:label=grow" if leg=="a" else f"grow:target=106000:from={a[3]}:label=grow"
    require([(s["spec"],s["end"]) for s in summary["steps"]] == [(want,"target")], "wrong step or stop reason")
    require(summary["ctx"]==163840 and summary["payload_sha256"]==PAYLOAD_SHA and
            not summary["dspark"] and summary["quant"]=="mxfp4" and summary["kv_disk_dir"]==str(kvdir(policy)), "session setup")
    flags=summary["server_flags"]
    require(flags[flags.index("--ssd-streaming-cache-experts")+1]==str(SLOTS) and
            flags[flags.index("--kv-disk-space-mb")+1]==str(KV_MIB), "slot or KV limits")
    env=summary["env"]
    require(env["DS4_ROCM_STREAM_SEED_POLICY"]=="cold" and env["DS4_ROCM_STREAM_EVICT_POLICY"]==policy and
            env["DS4_D12_CACHE_TRACE"]==str(tracepath(name)), "policy or trace setup")
    require(any(int(n)==summary["max_context"] for _,n in summary["shutdown_persist"]), "final persisted prefix missing")
    result=summary,turns,tr,path
    if leg=="b":
        chronology(a,result)
    return result


def validate_trace(name, policy):
    raw=tracepath(name)
    compressed=Path(str(raw)+".gz")
    require(raw.exists() != compressed.exists(), "expected one trace artifact")
    if raw.exists():
        subprocess.run(["gzip","--",str(raw)],check=True)
    with gzip.open(compressed,"rt") as stream:
        mark=next(line for line in stream if line.startswith("M ")).split()
    require(int(mark[4])==SLOTS+768 and int(mark[6])==0, "incorrect initial budget or residual cache")
    cmd=[sys.executable,"-B",str(CHECKOUT/"scripts/d19/replay.py")]
    if policy=="lru":
        cmd.append("--lru")
    subprocess.run([*cmd,str(compressed)],check=True)


def run(policy,leg):
    name=tag(policy,leg)
    q4_check.fresh(name)
    cache=kvdir(policy)
    cache.mkdir(parents=True,exist_ok=True)
    refs=load_records()
    a=None
    if leg=="a":
        require(not any(cache.iterdir()), "A requires an empty policy-specific KV directory")
        spec="grow:target=66000:label=grow"
    else:
        a=inspect(refs,policy,"a")
        require(any(cache.glob("*.kv")), "missing A checkpoints")
        validate_trace(tag(policy,"a"),policy)
        spec=f"grow:target=106000:from={a[3]}:label=grow"
    # Allow the full configured KV budget, plus 1 GiB for logs/trace/gzip.
    # Historical uncompressed A/B traces used 38.7/25.1 MB with ~1.8 GiB KV.
    for volume in (cache,ROOT/"d19-traces",ROOT/"runs"):
        require(shutil.disk_usage(volume).free >= (KV_MIB+1024)*2**20, f"insufficient space on {volume}")
    binary=CHECKOUT/"ds4-server"
    require(sha256(binary.read_bytes()).hexdigest()==BINARY_SHA, "server binary differs from pinned D19")
    env={k:v for k,v in os.environ.items() if not k.startswith(("DS4_","D14_","HSA_"))}
    # Parent q4.sh sources the same model and replica settings as the short arms.
    for k in ("DS4_MODEL","DS4_ROCM_STREAM_REPLICA"):
        env[k]=os.environ[k]
    env.update(DS4_DIR=str(CHECKOUT),D14_PAYLOAD=str(ROOT/"d14-payload-v2.json"),
               D14_BUDGET_MIN="100",DS4_ROCM_STREAM_SEED_POLICY="cold",DS4_ROCM_STREAM_EVICT_POLICY=policy,
               DS4_D12_CACHE_TRACE=str(tracepath(name)),DS4_NOTE=f"D19 matched long trace {policy} {leg}; 4500 dynamic slots")
    require(not (ROOT/"PAUSED").exists(), "PAUSED")
    flags=f"--ssd-streaming-cache-experts {SLOTS} --kv-disk-dir {cache} --kv-disk-space-mb {KV_MIB}"
    subprocess.run([str(CHECKOUT/"scripts/d19/run.sh"),name,"s1-long","163840",flags,spec],env=env,check=True)
    data=load_records()
    inspect(data,policy,leg,a)
    validate_trace(name,policy)
    other="s3fifo" if policy=="lru" else "lru"
    if any(r.get("tag")==tag(other,leg) for r in data):
        session_qc.compare(data,ROOT/"runs",tag("lru",leg),tag("s3fifo",leg),False)
        if leg=="b":
            metrics={}
            for arm in ("lru","s3fifo"):
                arm_rows=[r for r in data if r.get("tag")==tag(arm,"b") and r.get("kind")=="turn"]
                completed=[r for r in arm_rows[1:] if r["prompt_tokens"] >= 100000]
                require(completed, "no warm reply with 100000 input tokens")
                tokens=sum(r["reply_tokens"] for r in completed)
                seconds=sum(r["decode_s"] for r in completed)
                metrics[arm]=dict(turn_ids=[r["turn"] for r in completed],reply_tokens=tokens,
                                  decode_s=seconds,decode_tps=tokens/seconds,
                                  ttft_sum_s=sum(r["ttft_s"] for r in completed),
                                  restored_first_turn_ttft_s=arm_rows[0]["ttft_s"])
            require(metrics["lru"]["turn_ids"]==metrics["s3fifo"]["turn_ids"],"different 100k turn sets")
            print("Matched warm turns starting at 100000+:", json.dumps(metrics,sort_keys=True))


if __name__=="__main__":
    names=[tag(*pair) for pair in ORDER]
    require(len(sys.argv)<=2 and (len(sys.argv)==1 or sys.argv[1] in names), f"expected one of {names}")
    require(os.environ.get("INVOCATION_ID"), "use a dedicated systemd unit via q4.sh")
    first=names.index(sys.argv[1]) if len(sys.argv)==2 else 0
    for policy,leg in ORDER[first:]:
        run(policy,leg)
