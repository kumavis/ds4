#!/usr/bin/env python3
"""D14 driver: replays agent sessions against ds4-server and records each turn.

Subcommands (stdlib only, runs on the box inside nix-shell):

  driver.py run --run-dir DIR --port P --payload FILE --step SPEC [--step SPEC ...]
      Waits for the server to listen, runs the steps in order, then stops the
      server cleanly (see `stop`). Writes DIR/turns.jsonl (one raw row per turn)
      and DIR/transcript-<step>.json (messages incl. the model's replies).
  driver.py stop --run-dir DIR --port P
      The clean stop: wait until no request is in flight (no TCP connection on
      the port, the server log quiet for 5 s, GPU busy < 25% for 3 samples),
      check that the PID in DIR/server.pid is ds4-server, send ONE SIGTERM,
      wait for the process to exit. Never a second signal, never SIGKILL.
  driver.py finalize --run-dir DIR --tag TAG ... --results FILE
      After the server exited: parse server.log + sys.log + turns.jsonl into
      one record per turn and one scenario record, append to FILE.

Step specs (key=value joined by ':'):
  grow:target=32768[:session=0][:max_turns=N][:label=L]
      one session from the system prompt; each turn adds the next payload event
      and the model's own reply; stop before prompt+event+reply exceeds target
  resume:from=RUN_DIR:label=L:at=TOKENS[:turns=N][:salt=1]
      replay the transcript of RUN_DIR/transcript-L.json up to the turn whose
      prompt was nearest TOKENS (its user message, no reply), then N-1 further
      turns with the same events (feeding back the new replies); salt=1 puts a
      line in front of the system prompt so nothing can be reused (cold prefill)
  multi:n=2:target=16384[:stagger_s=0]
      n sessions (payload sessions 0..n-1, shared system prompt) concurrently
"""
import argparse, datetime as dt, fcntl, glob, hashlib, http.client, json, os, platform, re, signal
import socket, statistics, subprocess, sys, threading, time

MIB = 1048576.0
MEM_FLOOR_MB = 12288


# ------------------------------------------------------------------ system
def mem_available_mb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) // 1024
    return None


def nvme_read_sectors():
    tot = 0
    with open("/proc/diskstats") as f:
        for l in f:
            p = l.split()
            if re.fullmatch(r"nvme\d+n\d+", p[2]):
                tot += int(p[5])
    return tot


def gpu_busy():
    for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            return int(open(p).read().strip())
        except OSError:
            pass
    return None


def port_connections(port):
    """Established TCP connections to the server port (either side)."""
    n = 0
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for l in open(path).readlines()[1:]:
                p = l.split()
                lport = int(p[1].split(":")[1], 16)
                rport = int(p[2].split(":")[1], 16)
                if p[3] == "01" and (lport == port or rport == port):
                    n += 1
        except OSError:
            pass
    return n


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_from(path, off, end=None):
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read() if end is None else f.read(end - off)
    return data.decode("utf-8", "replace")


def now_iso(t=None):
    return dt.datetime.fromtimestamp(t if t is not None else time.time()).astimezone().isoformat(timespec="seconds")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ HTTP
class TurnResult:
    pass


def chat(port, messages, max_tokens, on_token=None):
    body = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "stream": True,
                       "stream_options": {"include_usage": True}, "temperature": 0,
                       "max_tokens": max_tokens, "think": False}).encode()
    r = TurnResult()
    r.t_send = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json", "Connection": "close"})
    resp = conn.getresponse()
    r.http_status = resp.status
    r.text, r.reasoning, r.chunk_times, r.usage, r.finish, r.error = [], [], [], None, None, None
    r.t_first = None
    if resp.status != 200:
        r.error = resp.read().decode("utf-8", "replace")[:2000]
        conn.close()
        r.t_end = time.time()
        return r
    buf = b""
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        if ev.get("error"):
            r.error = json.dumps(ev["error"])[:2000]
        if ev.get("usage"):
            r.usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            c = d.get("content")
            rc = d.get("reasoning_content") or d.get("reasoning")
            t = time.time()
            if c or rc:
                if r.t_first is None:
                    r.t_first = t
                r.chunk_times.append(t)
                if c:
                    r.text.append(c)
                if rc:
                    r.reasoning.append(rc)
                if on_token:
                    on_token(len(r.chunk_times), t)
            if ch.get("finish_reason"):
                r.finish = ch["finish_reason"]
    conn.close()
    r.t_end = time.time()
    r.text = "".join(r.text)
    r.reasoning = "".join(r.reasoning)
    return r


def wait_ready(port, pid_file, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pid = read_pid(pid_file)
        if pid and not os.path.exists(f"/proc/{pid}"):
            return None  # server exited (e.g. refused admission)
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/v1/models", headers={"Connection": "close"})
            if c.getresponse().status == 200:
                c.close()
                return time.time() - t0
        except OSError:
            pass
        time.sleep(2)
    return None


def read_pid(pid_file):
    try:
        return int(open(pid_file).read().strip())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ clean stop
def clean_stop(run_dir, port, max_wait_s=7200):
    """One SIGTERM to an idle ds4-server; wait for it to exit. Returns a dict."""
    pid_file = os.path.join(run_dir, "server.pid")
    slog = os.path.join(run_dir, "server.log")
    pid = read_pid(pid_file)
    out = {"pid": pid}
    if not pid or not os.path.exists(f"/proc/{pid}"):
        out["result"] = "not-running"
        return out
    comm = open(f"/proc/{pid}/comm").read().strip()
    out["comm"] = comm
    if comm != "ds4-server":
        out["result"] = f"refused: pid {pid} is {comm}, not ds4-server"
        say("stop:", out["result"])
        return out
    # idle: no connection, log quiet 5 s, GPU busy < 25% three times in a row
    t0 = time.time()
    calm, last_size, last_change = 0, log_size(slog), time.time()
    while True:
        s = log_size(slog)
        if s != last_size:
            last_size, last_change = s, time.time()
        busy = gpu_busy()
        conns = port_connections(port)
        quiet = time.time() - last_change >= 5
        if conns == 0 and quiet and (busy is None or busy < 25):
            calm += 1
        else:
            calm = 0
        if calm >= 3:
            break
        if time.time() - t0 > max_wait_s:
            out["result"] = "not-idle: gave up waiting, server left running"
            say("stop:", out["result"])
            return out
        time.sleep(1)
    out["idle_wait_s"] = round(time.time() - t0, 1)
    out["log_off_at_stop"] = log_size(slog)
    out["sigterm_ts"] = now_iso()
    say(f"stop: server idle (no connection, log quiet, gpu {busy}%); sending one SIGTERM to {pid}")
    t1 = time.time()
    os.kill(pid, signal.SIGTERM)
    while os.path.exists(f"/proc/{pid}"):
        try:
            st = open(f"/proc/{pid}/stat").read().split()[2]
            if st == "Z":
                break
        except OSError:
            break
        time.sleep(0.2)
    out["exit_wait_s"] = round(time.time() - t1, 2)
    out["result"] = "stopped"
    say(f"stop: ds4-server exited {out['exit_wait_s']} s after SIGTERM")
    return out


# ------------------------------------------------------------------ turns
class Recorder:
    def __init__(self, run_dir, port):
        self.run_dir, self.port = run_dir, port
        self.slog = os.path.join(run_dir, "server.log")
        self.lock = threading.Lock()
        self.path = os.path.join(run_dir, "turns.jsonl")
        self.mem_min = None
        self.stop_flag = False
        self.t_ready = None

    def mem_ok(self):
        m = mem_available_mb()
        return m is None or m >= MEM_FLOOR_MB + 512

    def write(self, row):
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def turn(self, step, session, idx, event_i, messages, max_tokens, extra=None):
        early = {}
        sect0 = nvme_read_sectors()

        def on_tok(n, t):
            if n == 1:
                early["s_first"] = nvme_read_sectors()
            elif n == 17:
                early["s_17"] = nvme_read_sectors()
                early["t_17"] = t

        off0 = log_size(self.slog)
        mem0 = mem_available_mb()
        r = chat(self.port, messages, max_tokens, on_tok)
        sect_end = nvme_read_sectors()
        # wait (bounded) for the server's end-of-request log line
        t_wait = time.time()
        while time.time() - t_wait < 120:
            seg = read_from(self.slog, off0)
            if re.search(r"ds4-server: chat ctx=\S+ gen=\d+.* finish=", seg):
                break
            time.sleep(0.5)
        time.sleep(0.5)
        off1 = log_size(self.slog)
        row = {"step": step, "session": session, "turn": idx, "event": event_i,
               "t_send": r.t_send, "t_first": r.t_first, "t_end": r.t_end, "t_17": early.get("t_17"),
               "http_status": r.http_status, "error": r.error, "finish": r.finish, "usage": r.usage,
               "max_tokens": max_tokens, "chunks": len(r.chunk_times),
               "chunk_times": [round(t - r.t_send, 4) for t in r.chunk_times],
               "reply_chars": len(r.text), "reasoning_chars": len(r.reasoning),
               "reply_sha": hashlib.sha256(r.text.encode()).hexdigest()[:16],
               "sectors": {"start": sect0, "first": early.get("s_first"), "tok17": early.get("s_17"),
                           "end": sect_end},
               "log_off": [off0, off1], "mem_start_mb": mem0, "mem_end_mb": mem_available_mb(),
               "messages": len(messages), "messages_chars": sum(len(m["content"]) for m in messages)}
        if extra:
            row.update(extra)
        self.write(row)
        u = r.usage or {}
        ptd = u.get("prompt_tokens_details") or {}
        ttft = (r.t_first - r.t_send) if r.t_first else None
        dec = (len(r.chunk_times) - 1) / (r.t_end - r.t_first) if r.t_first and r.t_end > r.t_first else None
        say(f"{step} s{session} t{idx}: prompt={u.get('prompt_tokens')} cached={ptd.get('cached_tokens')} "
            f"gen={u.get('completion_tokens')} ttft={ttft and round(ttft, 1)}s "
            f"chunks/s={dec and round(dec, 2)} finish={r.finish} err={r.error and r.error[:200]}")
        return r, row


def load_payload(path):
    return json.load(open(path))


def step_args(spec):
    kind, _, rest = spec.partition(":")
    kv = dict(p.split("=", 1) for p in rest.split(":") if p) if rest else {}
    return kind, kv


def save_transcript(run_dir, label, transcript):
    with open(os.path.join(run_dir, f"transcript-{label}.json"), "w") as f:
        json.dump(transcript, f, ensure_ascii=False)


def guard(rec, ctx, budget_s, prompt_tokens, est_next, max_tokens, target):
    if rec.stop_flag:
        return "stop-flag"
    if not rec.mem_ok():
        return "mem-guard"
    if budget_s and rec.t_ready and time.time() - rec.t_ready > budget_s:
        return "time-budget"
    if prompt_tokens is not None and prompt_tokens + est_next + max_tokens > min(target, ctx - 64):
        return "target"
    return None


def run_grow(rec, payload, ctx, budget_s, step_label, session, target, max_turns, src=None):
    ev = payload["sessions"][session]["events"]
    messages = [{"role": "system", "content": payload["system"]}]
    transcript = {"system_sha": hashlib.sha256(payload["system"].encode()).hexdigest()[:16],
                  "session": session, "turns": []}
    prompt_tokens, reason = None, "events-exhausted"
    ratio = 0.85  # true tokens / est tokens (0.82 on the first S1 turn), learnt as we go
    first = 0
    if src:
        # continue a session saved by an earlier server session: the full
        # history incl. the model's replies, then the next payload events
        T = json.load(open(src))
        for t in T["turns"]:
            messages += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["reply"]}]
        transcript["turns"] = list(T["turns"])
        transcript["continued_from"] = src
        last = T["turns"][-1]
        prompt_tokens = last["prompt_tokens"] + (last.get("completion_tokens") or 0)
        ratio = T.get("ratio", 1.0)
        first = last["turn"] + 1
    for idx, e in enumerate(ev):
        if idx < first:
            continue
        if max_turns and idx >= first + max_turns:
            reason = "max-turns"
            break
        est_next = int(e["est_tokens"] * ratio) + 16
        g = guard(rec, ctx, budget_s, prompt_tokens if prompt_tokens is not None else 0,
                  est_next + (int(payload["system_est_tokens"] * ratio) if prompt_tokens is None else 0),
                  e["max_tokens"], target)
        if g:
            reason = g
            break
        messages.append({"role": "user", "content": e["text"]})
        r, row = rec.turn(step_label, session, idx, e["i"], messages, e["max_tokens"],
                          {"est_tokens": e["est_tokens"], "items": e["items"]})
        if r.error or not r.usage:
            reason = "error"
            messages.pop()
            break
        pt = r.usage["prompt_tokens"]
        if prompt_tokens is not None:
            grown = pt - prompt_tokens
            if e["est_tokens"] > 200 and grown > 0:
                ratio = 0.7 * ratio + 0.3 * (grown / e["est_tokens"])
        prompt_tokens = pt + (r.usage.get("completion_tokens") or 0)
        messages.append({"role": "assistant", "content": r.text})
        transcript["turns"].append({"turn": idx, "event": e["i"], "prompt_tokens": pt,
                                    "completion_tokens": r.usage.get("completion_tokens"),
                                    "max_tokens": e["max_tokens"], "user": e["text"], "reply": r.text,
                                    "reply_sha": row["reply_sha"]})
        transcript["ratio"] = ratio
        save_transcript(rec.run_dir, step_label if session == 0 else f"{step_label}-s{session}", transcript)
    say(f"{step_label}: ended ({reason}) at ~{prompt_tokens} tokens")
    return reason


def run_resume(rec, payload, ctx, budget_s, step_label, src_dir, src_label, at, turns, salt):
    tr = json.load(open(os.path.join(src_dir, f"transcript-{src_label}.json")))
    T = tr["turns"]
    k = min(range(len(T)), key=lambda j: abs(T[j]["prompt_tokens"] - at))
    system = payload["system"]
    if salt:
        system = f"Session {hashlib.sha256((step_label + str(at)).encode()).hexdigest()[:12]} (cold-prefill control).\n" + system
    messages = [{"role": "system", "content": system}]
    for j in range(k):
        messages += [{"role": "user", "content": T[j]["user"]}, {"role": "assistant", "content": T[j]["reply"]}]
    transcript = {"resumed_from": src_dir, "label": src_label, "turn": k, "turns": []}
    reason = "done"
    for n in range(turns):
        j = k + n
        if j >= len(T):
            reason = "transcript-end"
            break
        g = guard(rec, ctx, budget_s, None, 0, 0, 1 << 30)
        if g:
            reason = g
            break
        messages.append({"role": "user", "content": T[j]["user"]})
        r, row = rec.turn(step_label, tr.get("session", 0), j, T[j]["event"], messages, T[j]["max_tokens"],
                          {"resume_src_turn": j, "resume_src_reply_sha": T[j]["reply_sha"],
                           "resume_src_prompt_tokens": T[j]["prompt_tokens"], "salt": bool(salt)})
        if r.error or not r.usage:
            reason = "error"
            break
        messages.append({"role": "assistant", "content": r.text})
        transcript["turns"].append({"turn": j, "prompt_tokens": r.usage["prompt_tokens"], "reply_sha": row["reply_sha"]})
    save_transcript(rec.run_dir, step_label, transcript)
    say(f"{step_label}: ended ({reason})")
    return reason


def run_multi(rec, payload, ctx, budget_s, step_label, n, target, stagger_s, max_turns):
    reasons = {}

    def worker(s):
        time.sleep(stagger_s * s)
        reasons[s] = run_grow(rec, payload, ctx, budget_s, f"{step_label}", s, target, max_turns)

    th = [threading.Thread(target=worker, args=(s,)) for s in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return ",".join(f"s{s}:{reasons.get(s)}" for s in range(n))


def cmd_run(a):
    payload = load_payload(a.payload)
    rec = Recorder(a.run_dir, a.port)
    pid_file = os.path.join(a.run_dir, "server.pid")
    status = {"steps": []}
    say("waiting for the server to listen")
    ready = wait_ready(a.port, pid_file)
    if ready is None:
        status["ready"] = False
        say("server did not come up (exited or not listening)")
        json.dump(status, open(os.path.join(a.run_dir, "driver.json"), "w"), indent=1)
        # if it is still running (not listening), still stop it cleanly
        pid = read_pid(pid_file)
        if pid and os.path.exists(f"/proc/{pid}"):
            status["stop"] = clean_stop(a.run_dir, a.port)
            json.dump(status, open(os.path.join(a.run_dir, "driver.json"), "w"), indent=1)
        return 1
    rec.t_ready = time.time()
    status["ready"] = True
    status["ready_wait_s"] = round(ready, 1)
    status["t_ready"] = rec.t_ready
    say(f"server ready after {ready:.0f} s of waiting")
    budget = a.budget_min * 60 if a.budget_min else None
    try:
        for spec in a.step:
            kind, kv = step_args(spec)
            label = kv.get("label", kind)
            t0 = time.time()
            if kind == "grow":
                src = os.path.join(kv["from"], f"transcript-{kv.get('src_label', 'grow')}.json") if kv.get("from") else None
                why = run_grow(rec, payload, a.ctx, budget, label, int(kv.get("session", 0)),
                               int(kv["target"]), int(kv.get("max_turns", 0)), src)
            elif kind == "resume":
                why = run_resume(rec, payload, a.ctx, budget, label, kv["from"], kv.get("src_label", "grow"),
                                 int(kv["at"]), int(kv.get("turns", 1)), kv.get("salt") == "1")
            elif kind == "multi":
                why = run_multi(rec, payload, a.ctx, budget, label, int(kv["n"]), int(kv["target"]),
                                float(kv.get("stagger_s", 0)), int(kv.get("max_turns", 0)))
            elif kind == "ping":
                why = "done"
                for k in range(int(kv.get("n", 1))):
                    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
                    r, _ = rec.turn(label, 0, k, None, msgs, int(kv.get("max", 8)))
                    if r.error:
                        why = "error"
                        break
            elif kind == "idle":
                time.sleep(float(kv.get("s", 10)))
                why = "slept"
            else:
                why = f"unknown step {kind}"
            status["steps"].append({"spec": spec, "label": label, "end": why, "s": round(time.time() - t0, 1)})
            json.dump(status, open(os.path.join(a.run_dir, "driver.json"), "w"), indent=1)
            if why in ("mem-guard", "error", "time-budget"):
                say(f"stopping the scenario after step {label}: {why}")
                break
    except Exception as e:  # never leave the server running
        status["exception"] = repr(e)
        say("driver exception:", repr(e))
    finally:
        status["stop"] = clean_stop(a.run_dir, a.port)
        json.dump(status, open(os.path.join(a.run_dir, "driver.json"), "w"), indent=1)
    return 0 if not status.get("exception") else 1


# ------------------------------------------------------------------ parsing
def kvs(line):
    out = {}
    for k, v in re.findall(r"([A-Za-z0-9_]+)=([^\s;]+)", line):
        v = v.rstrip(",%")
        try:
            out[k] = float(v) if re.search(r"[.eE]", v) else int(v)
        except ValueError:
            out[k] = v
    return out


def parse_segment(seg, expert_mib):
    """Per-turn facts from the server log segment of one request."""
    d = {}
    lines = seg.splitlines()
    pre = [l for l in lines if " prefill " in l or re.search(r"chunk \d+/\d+ \(", l)]
    d["prefill_lines"] = len(pre)
    last = None
    for l in pre:
        m = re.search(r"chunk (\d+)/(\d+) \(([\d.]+)%\) chunk=([\d.]+) t/s avg=([\d.]+) t/s ([\d.]+)s", l)
        if m:
            last = m
    if last:
        d["prefill_log_done"], d["prefill_log_total"] = int(last[1]), int(last[2])
        d["prefill_log_avg_tps"], d["prefill_log_s"] = float(last[5]), float(last[6])
    m = None
    for l in lines:
        mm = re.search(r"ds4-server: chat ctx=(\S+) gen=(\d+).* finish=(\S+)(?: error=\"(.*?)\")? ([\d.]+)s", l)
        if mm:
            m = mm
    if m:
        d["log_ctx"], d["log_gen"], d["log_finish"], d["log_req_s"] = m[1], int(m[2]), m[3], float(m[5])
    dec = [l for l in lines if " decoding chunk=" in l]
    if dec:
        d["log_decode_avg_tps"] = kvs(dec[-1].split("decoding", 1)[1]).get("avg")
        mm = re.search(r"avg=([\d.]+) t/s", dec[-1])
        d["log_decode_avg_tps"] = float(mm[1]) if mm else None
    # KV cache events
    kv = []
    for l in lines:
        if "live kv cache miss" in l:
            kv.append({"type": "live-miss", **{k: v for k, v in kvs(l).items() if k in ("live", "prompt", "common", "reason")}})
        elif "kv cache stored" in l:
            x = kvs(l)
            ms = re.search(r"size=([\d.]+) MiB save=([\d.]+) ms", l)
            kv.append({"type": "store", "tokens": x.get("tokens"), "reason": x.get("reason"),
                       "size_mib": float(ms[1]) if ms else None, "ms": float(ms[2]) if ms else None})
        elif "kv cache hit" in l:
            x = kvs(l)
            ms = re.search(r"load=([\d.]+) ms", l)
            kv.append({"type": "hit", "tokens": x.get("tokens"), "ms": float(ms[1]) if ms else None})
        elif "kv cache skipped" in l or "kv cache evicted" in l or "kv cache load failed" in l:
            kv.append({"type": "other", "line": l.split("ds4-server: ", 1)[-1][:300]})
        elif "persisting resident KV" in l:
            kv.append({"type": "shutdown-persist", **kvs(l)})
    d["kv_events"] = kv
    # D8 per decode step (plain decode path, also batched serial fallback)
    steps = [kvs(l[l.index("D8_STEP"):]) for l in lines if "D8_STEP " in l]
    d["d8_steps"] = len(steps)
    if steps:
        mis = [s.get("misses", 0) for s in steps]
        d["d8_misses_total"] = sum(mis)
        d["d8_misses_per_step"] = round(sum(mis) / len(mis), 2)
        d["d8_misses_first16"] = round(statistics.mean(mis[:16]), 2)
        d["d8_misses_rest"] = round(statistics.mean(mis[16:]), 2) if len(mis) > 16 else None
        d["d8_step_ms_mean"] = round(statistics.mean(s.get("ms", 0) for s in steps), 2)
        d["d8_step_ms_first16"] = round(statistics.mean(s.get("ms", 0) for s in steps[:16]), 2)
        d["d8_step_ms_rest"] = round(statistics.mean(s.get("ms", 0) for s in steps[16:]), 2) if len(steps) > 16 else None
        d["d8_wait_ms_per_step"] = round(statistics.mean(s.get("wait_ms", 0) for s in steps), 2)
        d["d8_slots_per_step"] = round(statistics.mean(s.get("slots", 0) for s in steps), 1)
        d["d8_dspark_steps"] = sum(1 for s in steps if s.get("dspark") == 1)
        d["reply_tok_sha"] = hashlib.sha256(",".join(str(s.get("tok")) for s in steps).encode()).hexdigest()[:16]
    # DSpark rounds
    rounds = [l for l in lines if "V4.1 DSpark proposed=" in l]
    if rounds:
        ps = [kvs(l.split("DSpark", 1)[1]) for l in rounds]
        d["dspark_rounds"] = len(ps)
        d["dspark_proposed"] = sum(p.get("proposed", 0) for p in ps)
        d["dspark_accepted"] = sum(p.get("accepted", 0) for p in ps)
        d["dspark_accept_rate"] = round(d["dspark_accepted"] / d["dspark_proposed"], 4) if d["dspark_proposed"] else None
        d["dspark_avg_accept"] = round(d["dspark_accepted"] / len(ps), 3)
        dm = [float(m) for l in rounds for m in re.findall(r"draft=([\d.]+) ms", l)]
        vm = [float(m) for l in rounds for m in re.findall(r"verify=([\d.]+) ms", l)]
        d["dspark_draft_ms_mean"] = round(statistics.mean(dm), 2) if dm else None
        d["dspark_verify_ms_mean"] = round(statistics.mean(vm), 2) if vm else None
    return d


def parse_startup(text):
    d = {}
    m = re.search(r"= ([\d.]+) GiB planned", text)
    d["planned_gib"] = float(m[1]) if m else None
    m = re.search(r"dynamic slots (\d+) \(([\d.]+) GiB", text)
    d["cache_slots"], d["cache_gib"] = (int(m[1]), float(m[2])) if m else (None, None)
    m = re.search(r"\((\d+) experts, ([\d.]+) MiB each\)", text) or re.search(r"routed expert size: ([\d.]+) MiB", text)
    if m and m.lastindex == 2:
        d["expert_mib"] = float(m[2])
    elif m:
        d["expert_mib"] = float(m[1])
    else:
        d["expert_mib"] = None
    m = re.search(r"memory-sized SSD cache.*$", text, re.M)
    d["cache_sizing_line"] = m[0][:400] if m else None
    m = re.search(r"fixed = (.*)$", text, re.M)
    d["fixed_line"] = m[1][:400] if m else None
    m = re.search(r"ds4: memory: KV ([\d.]+) GiB \(raw ([\d.]+) \+ compressed ([\d.]+)\) \+ buffers ([\d.]+) GiB", text)
    if m:
        d["kv_gib"], d["kv_raw_gib"], d["kv_comp_gib"], d["buffers_gib"] = map(float, m.groups())
    m = re.search(r"context buffers ([\d.]+) MiB \(ctx=(\d+)", text)
    d["server_context_mib"] = float(m[1]) if m else None
    m = re.search(r"batched mode enabled resident_sessions=(\d+)", text)
    d["resident_sessions"] = int(m[1]) if m else 1
    d["mtp_disabled_batched"] = "speculative decoding is disabled while native session batching" in text
    m = re.search(r"KV disk cache (.*)$", text, re.M)
    d["kv_disk_line"] = m[1][:400] if m else None
    d["listening"] = "listening on http" in text
    m = re.search(r"^ds4: MTP support model loaded: (\S+)", text, re.M)
    d["mtp_loaded"] = bool(m)
    m = re.search(r"^ds4: DSpark verify setting: (\S+)", text, re.M)
    d["dspark_verify"] = m[1] if m else None
    errs = [l for l in text.splitlines() if re.search(r"error|fail|refus|not enough|exceed|cannot|unable|insufficient", l, re.I)
            and not l.startswith("D8_STEP") and "kv cache" not in l and "failures=0" not in l and "fallbacks=0" not in l]
    d["startup_errors"] = errs[-5:] or None
    return d


def parse_end(text):
    d = {}
    m = re.search(r"stream cache stats \w+: (.*)$", text, re.M)
    if m:
        parts = {p.strip().split(" ", 1)[0]: p.strip() for p in m[1].split(";")}
        sel, bat = kvs(parts.get("selected", "")), kvs(parts.get("batch", ""))
        d.update({"decode_hits": sel.get("hits"), "decode_misses": sel.get("misses"),
                  "prefill_unique": bat.get("unique"), "prefill_hits": bat.get("hits"), "prefill_misses": bat.get("misses")})
        m2 = re.search(r"evictions=(\d+) evicted=([\d.]+) GiB", m[1])
        d["evictions"], d["evicted_gib"] = (int(m2[1]), float(m2[2])) if m2 else (None, None)
    m = re.search(r"stream read profile (.*)$", text, re.M)
    if m:
        p = kvs(m[1])
        m2 = re.search(r"bytes=([\d.]+) GiB", m[1])
        d.update({"read_jobs": p.get("jobs"), "read_gib": float(m2[1]) if m2 else None,
                  "read_wait_ms": p.get("wait")})
    m = re.search(r"^ds4: DSpark stats (.*)$", text, re.M)
    d["dspark_stats"] = kvs(m[1]) if m else None
    d["shutdown_logged"] = "shutdown requested, draining requests" in text
    d["shutdown_persist"] = re.findall(r"persisting resident KV cache before shutdown slot=(\d+) tokens=(\d+)", text) or None
    return d


def sys_rows(run_dir):
    rows = []
    p = os.path.join(run_dir, "sys.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            d = dict(re.findall(r"(\w+)=(\S+)", line))
            if d:
                d["_t"] = d.get("epoch")
                rows.append(d)
    return rows


def numcol(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (KeyError, ValueError):
            pass
    return out


def window_stats(rows, t0, t1):
    w = [r for r in rows if r.get("epoch") and t0 - 1 <= float(r["epoch"]) <= t1 + 1]
    av, nv, gb = numcol(w, "avail_mb"), numcol(w, "nvme_max_c"), numcol(w, "gpu_busy")
    return {"avail_min_mb": min(av) if av else None, "nvme_max_c": max(nv) if nv else None,
            "gpu_busy_median": statistics.median(gb) if gb else None, "sys_samples": len(w)}


def git(d, *args):
    try:
        return subprocess.run(["git", "-C", d, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def cmd_finalize(a):
    run_dir = a.run_dir
    slog = os.path.join(run_dir, "server.log")
    text = open(slog, errors="replace").read() if os.path.exists(slog) else ""
    st = parse_startup(text)
    end = parse_end(text)
    rows = sys_rows(run_dir)
    turns = [json.loads(l) for l in open(os.path.join(run_dir, "turns.jsonl"))] if os.path.exists(os.path.join(run_dir, "turns.jsonl")) else []
    drv = json.load(open(os.path.join(run_dir, "driver.json"))) if os.path.exists(os.path.join(run_dir, "driver.json")) else {}
    env = {k: v for k, v in os.environ.items() if re.match(r"(DS4_|HSA_)", k) and k not in ("HSA_PATH", "DS4_NOTE")}
    flags = a.flags
    model = os.path.basename(os.path.realpath(a.model)) if a.model else None
    quant = "mxfp4" if model and "MXFP4" in model.upper() else ("q2" if model and ("Q2_K" in model or "IQ2" in model) else "other")
    def flag(name):
        return flags[flags.index(name) + 1] if name in flags and flags.index(name) + 1 < len(flags) else None
    base = {"schema": 1, "tag": a.tag, "scenario": a.scenario, "note": a.note or None,
            "host": socket.gethostname(), "kernel": platform.release(),
            "ds4_dir": a.ds4_dir, "ds4_commit": a.ds4_commit or None,
            "ds4_dirty": {"true": True, "false": False}.get(a.ds4_dirty),
            "model": model, "quant": quant,
            "mtp_model": os.path.basename(flag("--mtp-model")) if flag("--mtp-model") else None,
            "dspark": "--dspark" in flags, "ctx": a.ctx,
            "batched_sessions": int(flag("--batched-session")) if flag("--batched-session") else None,
            "kv_disk_dir": flag("--kv-disk-dir"), "server_flags": flags, "env": env,
            "payload": a.payload, "payload_sha256": hashlib.sha256(open(a.payload, "rb").read()).hexdigest()[:12] if a.payload and os.path.exists(a.payload) else None,
            "bench_commit": git(os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"),
            "run_dir": os.path.abspath(run_dir)}
    out = []
    emib = st.get("expert_mib") or a.expert_mib
    for t in turns:
        u = t.get("usage") or {}
        ptd = u.get("prompt_tokens_details") or {}
        pt, cached = u.get("prompt_tokens"), ptd.get("cached_tokens")
        seg = read_from(slog, t["log_off"][0], t["log_off"][1]) if t.get("log_off") else ""
        p = parse_segment(seg, emib)
        ttft = round(t["t_first"] - t["t_send"], 3) if t.get("t_first") else None
        ct = t.get("chunk_times") or []
        gen = u.get("completion_tokens")
        decode_s = (t["t_end"] - t["t_first"]) if t.get("t_first") else None
        dec_tps = round((gen - 1) / decode_s, 3) if gen and gen > 1 and decode_s and decode_s > 0 else None
        late_tps = None
        if t.get("t_17") and gen and gen > 32 and t["t_end"] > t["t_17"]:
            late_tps = round((len(ct) - 17) / (t["t_end"] - t["t_17"]), 3) if len(ct) > 17 else None
        new = (pt - cached) if pt is not None and cached is not None else None
        pre_s = p.get("prefill_log_s")
        sec = t.get("sectors") or {}
        def est_miss(a_, b_, ntok):
            if a_ is None or b_ is None or not emib or not ntok:
                return None
            return round((b_ - a_) * 512 / MIB / emib / ntok, 2)
        rec = dict(base)
        rec.update({
            "kind": "turn", "ts": now_iso(t["t_send"]), "step": t["step"], "session": t["session"],
            "turn": t["turn"], "event": t.get("event"), "items": t.get("items"),
            "prompt_tokens": pt, "reused_tokens": cached, "new_tokens": new,
            "cache_write_tokens": ptd.get("cache_write_tokens"),
            "context_after": (pt + gen) if pt is not None and gen is not None else None,
            "ttft_s": ttft, "prefill_s": pre_s,
            "prefill_tps": round(new / pre_s, 2) if new and pre_s else None,
            "prefill_tps_ttft": round(new / ttft, 2) if new and ttft else None,
            "prefill_log_avg_tps": p.get("prefill_log_avg_tps"),
            "reply_tokens": gen, "max_tokens": t.get("max_tokens"), "finish": t.get("finish"),
            "decode_s": round(decode_s, 3) if decode_s else None, "decode_tps": dec_tps,
            "decode_tps_after16": late_tps, "log_decode_avg_tps": p.get("log_decode_avg_tps"),
            "request_s": round(t["t_end"] - t["t_send"], 3), "log_req_s": p.get("log_req_s"),
            "reply_sha": t.get("reply_sha"), "reply_tok_sha": p.get("reply_tok_sha"),
            "reply_chars": t.get("reply_chars"), "reasoning_chars": t.get("reasoning_chars"),
            "kv_events": p.get("kv_events"),
            "kv_store_ms": round(sum(e.get("ms") or 0 for e in p["kv_events"] if e["type"] == "store"), 1),
            "kv_store_mib": round(sum(e.get("size_mib") or 0 for e in p["kv_events"] if e["type"] == "store"), 1),
            "kv_load_ms": round(sum(e.get("ms") or 0 for e in p["kv_events"] if e["type"] == "hit"), 1),
            "kv_hit_tokens": max([e.get("tokens") or 0 for e in p["kv_events"] if e["type"] == "hit"], default=None),
            "d8_steps": p.get("d8_steps"), "misses_per_tok": p.get("d8_misses_per_step"),
            "misses_first16": p.get("d8_misses_first16"), "misses_rest": p.get("d8_misses_rest"),
            "step_ms_mean": p.get("d8_step_ms_mean"), "step_ms_first16": p.get("d8_step_ms_first16"),
            "step_ms_rest": p.get("d8_step_ms_rest"), "read_wait_ms_per_tok": p.get("d8_wait_ms_per_step"),
            "slots_per_tok": p.get("d8_slots_per_step"),
            "est_read_mib_prefill": round(((sec.get("first") or 0) - sec["start"]) * 512 / MIB, 1) if sec.get("first") else None,
            "est_read_mib_decode": round((sec["end"] - sec["first"]) * 512 / MIB, 1) if sec.get("first") else None,
            "est_misses_per_tok": est_miss(sec.get("first"), sec.get("end"), (gen - 1) if gen else None),
            "est_misses_first16": est_miss(sec.get("first"), sec.get("tok17"), 16),
            "est_misses_rest": est_miss(sec.get("tok17"), sec.get("end"), (gen - 17) if gen and gen > 17 else None),
            "dspark_rounds": p.get("dspark_rounds"), "dspark_proposed": p.get("dspark_proposed"),
            "dspark_accepted": p.get("dspark_accepted"), "dspark_accept_rate": p.get("dspark_accept_rate"),
            "dspark_avg_accept": p.get("dspark_avg_accept"), "dspark_draft_ms": p.get("dspark_draft_ms_mean"),
            "dspark_verify_ms": p.get("dspark_verify_ms_mean"),
            "mem_start_mb": t.get("mem_start_mb"), "mem_end_mb": t.get("mem_end_mb"),
            "http_status": t.get("http_status"), "error": t.get("error"),
            "resume_src_turn": t.get("resume_src_turn"), "resume_src_reply_sha": t.get("resume_src_reply_sha"),
            "resume_match": (t.get("reply_sha") == t.get("resume_src_reply_sha")) if t.get("resume_src_reply_sha") else None,
            "salt": t.get("salt"), "est_tokens": t.get("est_tokens"),
            "chunk_times": ct,
        })
        rec.update(window_stats(rows, t["t_send"], t["t_end"]))
        out.append(rec)
    # scenario summary
    allsys = window_stats(rows, 0, 1e12)
    exit_line = re.search(r"^EXIT=(-?\d+) wall_s=(\d+)", open(os.path.join(run_dir, "run.log")).read(), re.M) if os.path.exists(os.path.join(run_dir, "run.log")) else None
    summ = dict(base)
    summ.update({"kind": "scenario", "ts": a.start and now_iso(float(a.start)), "ts_end": a.end and now_iso(float(a.end)),
                 "exit_code": int(exit_line[1]) if exit_line else None, "wall_s": int(exit_line[2]) if exit_line else None,
                 "steps": drv.get("steps"), "ready": drv.get("ready"),
                 "startup_s": round(drv["t_ready"] - float(a.start), 1) if drv.get("t_ready") and a.start else None,
                 "stop": drv.get("stop"), "driver_exception": drv.get("exception"),
                 "turns": len(out), "max_context": max([r["context_after"] or 0 for r in out], default=None),
                 "reply_shas": [r["reply_sha"] for r in out],
                 "replies_digest": hashlib.sha256(",".join(r["reply_sha"] for r in out).encode()).hexdigest()[:16] if out else None})
    summ.update(st)
    summ.update(end)
    summ.update(allsys)
    out.append(summ)
    lines = [json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in out]
    if a.results == "-":
        print("\n".join(lines))
    else:
        with open(a.results, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("\n".join(lines) + "\n")
        print(f"finalize: {len(out) - 1} turn rows + 1 scenario row appended to {a.results}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--payload", required=True)
    r.add_argument("--ctx", type=int, required=True)
    r.add_argument("--budget-min", type=float, default=0)
    r.add_argument("--step", action="append", default=[])
    s = sub.add_parser("stop")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--port", type=int, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--tag", required=True)
    f.add_argument("--scenario", default=None)
    f.add_argument("--note", default=None)
    f.add_argument("--ctx", type=int)
    f.add_argument("--model")
    f.add_argument("--payload")
    f.add_argument("--ds4-dir")
    f.add_argument("--ds4-commit")
    f.add_argument("--ds4-dirty")
    f.add_argument("--start")
    f.add_argument("--end")
    f.add_argument("--expert-mib", type=float, default=None)
    f.add_argument("--results", default="/data/ds4-bench/sessions.jsonl")
    f.add_argument("flags", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "stop":
        out = clean_stop(a.run_dir, a.port)
        print(json.dumps(out))
        return 0
    return cmd_finalize(a)


if __name__ == "__main__":
    sys.exit(main())
