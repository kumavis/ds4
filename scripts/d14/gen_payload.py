#!/usr/bin/env python3
"""D14 payload generator: a deterministic coding-agent session (stdlib only).

  gen_payload.py --src /data/ds4-d14 --seed 14 --sessions 4 --out payload.json

Output (JSON):
  system        the shared system prompt: agent instructions, ~20 JSON tool
                schemas, repository context (target ~6.5k tokens)
  sessions[s]   {"id", "seed", "events": [...]} ; session s uses seed+s, so
                sessions share the system prompt and diverge afterwards
  event         {"i", "items": [kind,...], "text", "max_tokens", "est_tokens"}
                one user-side message per model turn: a user instruction, or
                1-3 tool results (read_file / grep / build / test / git / diff),
                sometimes an instruction followed by a tool result
Every byte comes from this script or from the source files under --src
(the public ds4 tree); nothing private, no network. Same seed and same --src
tree give byte-identical output (sha256 printed).

Token estimates use chars/3.1 for code-like text and chars/4.0 for prose; they
only steer the mix, the driver reads the true counts from the server.
"""
import argparse, difflib, hashlib, json, os, random, re, subprocess, sys

CODE_CPT = 3.1
PROSE_CPT = 4.0

# --------------------------------------------------------------------------
# System prompt (written for this benchmark)
# --------------------------------------------------------------------------
INSTRUCTIONS = """You are Forge, an autonomous software engineering agent working inside a single \
source repository on a Linux workstation. You help the user understand, change, build and test \
the code. You work in turns: the user (or the harness on the user's behalf) sends you an \
instruction or the results of tools you asked for, and you answer with your analysis and the next \
action you want to take.

# How you work

1. Read before you write. Before you change a file, read the parts of it you will touch and the \
callers of every function whose behaviour you change. Do not guess at the contents of a file you \
have not seen in this session.
2. Keep changes small and reviewable. Prefer one focused change per turn. Never reformat code you \
are not otherwise changing, never rename public symbols without being asked, and keep the existing \
style of the file (indentation, brace placement, naming, comment style).
3. Build and test after every change. A change is not done until the build is clean (no new \
warnings) and the relevant tests pass. If a test fails, read the failure output carefully and fix \
the cause, not the symptom.
4. Be precise about what you know. When you state a fact about the code, cite the file and line. \
When you are not sure, say so and say what you would read or run to find out.
5. Respect the user's environment. Never run destructive commands (rm -rf, git reset --hard, \
git push --force, dd, mkfs) unless the user explicitly asked for exactly that command. Never \
install packages or fetch anything from the network. Never print or copy secrets, keys or tokens.
6. Long-running processes are dangerous on this machine: never kill, interrupt or time out a \
process that is using the GPU; wait for it to finish.

# Output format

Answer in plain prose with short paragraphs. Use fenced code blocks for code, commands and diffs. \
When you want to call a tool, end your answer with exactly one fenced block tagged `tool` that \
contains a single JSON object of the form {"name": "<tool name>", "arguments": {...}} matching \
one of the schemas below. You may call at most three tools per turn; put each call in its own \
`tool` block. The harness runs the calls in order and returns each result in a <tool_result> \
element whose `name` and `call_id` attributes identify the call. Tool results can be long; read \
them fully before you answer. If a result is truncated, the element ends with a note saying how \
many lines were omitted.

Keep each answer focused: summarise what the results show, state what you conclude, and state \
the next step. Do not repeat large parts of a tool result back to the user. If the user asks a \
direct question, answer it first.

# Conventions in this repository

- C11 with GNU extensions, built with the project's Makefile; GPU code is HIP/ROCm (`.cu`, \
`.cuh`) and Metal (`.m`, `metal/`). Tests live in `tests/` and are built by `make test-*` targets.
- Errors are reported through an `err` buffer and a length (`char *err, size_t errlen`) and a \
non-zero return code; do not call `exit()` from library code.
- Environment variables that change behaviour are named `DS4_*`, are read once, and must be \
documented next to the code that reads them.
- Logging goes to stderr with a `ds4:` or `ds4-server:` prefix and key=value fields so that \
benchmark scripts can parse it. Do not change the format of an existing log line without \
updating its parsers.
- Every change to model output must be measured: greedy decoding of a fixed prompt must give the \
same token IDs before and after a change that is meant to be exact.

# Tools
"""


def P(t, d, **kw):
    out = {"type": t, "description": d}
    out.update(kw)
    return out


TOOLS = [
    ("read_file", "Read a text file from the repository. Returns the requested line range with "
     "line numbers (like `cat -n`). Use offset and limit for large files; results longer than "
     "4,000 lines are truncated.",
     {"path": P("string", "Path relative to the repository root."),
      "offset": P("integer", "First line to return, 1-based. Default 1.", minimum=1),
      "limit": P("integer", "Maximum number of lines to return. Default 400.", minimum=1, maximum=4000)},
     ["path"]),
    ("write_file", "Create a file or replace its entire contents. Prefer edit_file for changes to "
     "existing files. Fails if the parent directory does not exist.",
     {"path": P("string", "Path relative to the repository root."),
      "content": P("string", "Complete new contents of the file."),
      "mode": P("string", "File mode for new files, octal string such as \"0644\".", default="0644")},
     ["path", "content"]),
    ("edit_file", "Replace one exact occurrence of a string in a file. The old string must match "
     "the file exactly, including whitespace, and must be unique unless replace_all is true.",
     {"path": P("string", "Path relative to the repository root."),
      "old_string": P("string", "Exact text to replace."),
      "new_string": P("string", "Replacement text."),
      "replace_all": P("boolean", "Replace every occurrence instead of exactly one.", default=False)},
     ["path", "old_string", "new_string"]),
    ("apply_patch", "Apply a unified diff to the working tree. The patch must apply cleanly with "
     "zero fuzz; on failure nothing is changed and the rejected hunks are returned.",
     {"patch": P("string", "Unified diff with a/ and b/ prefixes."),
      "check_only": P("boolean", "Only check whether the patch applies.", default=False)},
     ["patch"]),
    ("grep", "Search file contents with an extended regular expression. Returns matching lines "
     "as path:line:text, at most max_results lines.",
     {"pattern": P("string", "Extended regular expression (POSIX ERE)."),
      "path": P("string", "File or directory to search. Default: repository root."),
      "glob": P("string", "Only search files whose name matches this glob, e.g. \"*.c\"."),
      "ignore_case": P("boolean", "Case-insensitive match.", default=False),
      "context": P("integer", "Lines of context before and after each match.", minimum=0, maximum=20),
      "max_results": P("integer", "Maximum number of matching lines.", default=200)},
     ["pattern"]),
    ("glob", "List files whose path matches a glob pattern, sorted by modification time, newest "
     "first.",
     {"pattern": P("string", "Glob such as \"rocm/**/*.cuh\"."),
      "limit": P("integer", "Maximum number of paths.", default=500)},
     ["pattern"]),
    ("list_dir", "List a directory with sizes and modification times, like `ls -la`.",
     {"path": P("string", "Directory relative to the repository root."),
      "all": P("boolean", "Include dot files.", default=False)},
     ["path"]),
    ("run_command", "Run a shell command in the repository root with a time limit and return its "
     "exit status, stdout and stderr (each truncated to 4,000 lines). The command runs without "
     "network access. Do not use it for commands that use the GPU; use run_benchmark instead.",
     {"command": P("string", "Command line, run with /bin/sh -c."),
      "timeout_s": P("integer", "Time limit in seconds.", default=600, maximum=3600),
      "env": P("object", "Extra environment variables.", additionalProperties={"type": "string"}),
      "cwd": P("string", "Working directory relative to the repository root.")},
     ["command"]),
    ("build", "Build one or more Makefile targets and return the compiler output. Warnings are "
     "reported as errors for files you changed in this session.",
     {"targets": P("array", "Make targets, e.g. [\"strix-halo\"] or [\"test-session-state\"].",
                   items={"type": "string"}),
      "jobs": P("integer", "Parallel jobs (-j).", default=16),
      "clean": P("boolean", "Run `make clean` first.", default=False)},
     ["targets"]),
    ("run_tests", "Run test binaries or a make test target and return a summary with the output "
     "of failing tests.",
     {"target": P("string", "Make target or test binary path."),
      "filter": P("string", "Only run tests whose name contains this string."),
      "repeat": P("integer", "Run the selection this many times.", default=1, maximum=20)},
     ["target"]),
    ("run_benchmark", "Queue a bounded GPU benchmark run through the project's harness. The call "
     "waits for the GPU lock, runs the model once with the given flags, and returns the harness "
     "record (tokens per second, cache misses, read wait, memory, temperatures).",
     {"tag": P("string", "Record tag, [A-Za-z0-9._-]+."),
      "tokens": P("integer", "Tokens to generate.", minimum=1, maximum=4096),
      "prompt_file": P("string", "Prompt file on the benchmark host."),
      "flags": P("array", "Extra flags for the runtime.", items={"type": "string"}),
      "env": P("object", "DS4_* variables for the run.", additionalProperties={"type": "string"})},
     ["tag", "tokens", "prompt_file"]),
    ("git_status", "Show the working tree status (porcelain v2) and the current branch.",
     {"untracked": P("boolean", "Include untracked files.", default=True)}, []),
    ("git_diff", "Show changes as a unified diff.",
     {"revision": P("string", "Revision or range, e.g. \"HEAD~3..HEAD\". Default: working tree "
                    "against HEAD."),
      "paths": P("array", "Limit to these paths.", items={"type": "string"}),
      "stat": P("boolean", "Show a diffstat instead of the patch.", default=False),
      "context": P("integer", "Lines of context.", default=3)},
     []),
    ("git_log", "Show commit history.",
     {"revision": P("string", "Revision range."),
      "max_count": P("integer", "Maximum commits.", default=20),
      "paths": P("array", "Only commits touching these paths.", items={"type": "string"}),
      "format": P("string", "oneline, short or full.", enum=["oneline", "short", "full"])},
     []),
    ("git_commit", "Commit staged changes. Refuses if there is nothing staged or if the message "
     "does not follow the repository's `type(scope): summary` convention.",
     {"message": P("string", "Commit message."),
      "paths": P("array", "Paths to stage before committing.", items={"type": "string"}),
      "amend": P("boolean", "Amend the previous commit.", default=False)},
     ["message"]),
    ("symbol_search", "Find definitions and references of a C symbol using the project's ctags "
     "index.",
     {"symbol": P("string", "Function, type, macro or variable name."),
      "kind": P("string", "definition, reference or both.", enum=["definition", "reference", "both"]),
      "limit": P("integer", "Maximum results.", default=100)},
     ["symbol"]),
    ("read_log", "Read the tail or a byte range of a log file produced by a build, a test or a "
     "benchmark run.",
     {"path": P("string", "Log path."),
      "tail_lines": P("integer", "Return the last N lines.", default=200),
      "grep": P("string", "Only lines matching this regular expression.")},
     ["path"]),
    ("todo", "Maintain the task list for this session. The list is shown to the user.",
     {"action": P("string", "add, update, complete or list.", enum=["add", "update", "complete", "list"]),
      "id": P("integer", "Item id for update and complete."),
      "text": P("string", "Item text for add and update."),
      "status": P("string", "pending, in_progress or done.", enum=["pending", "in_progress", "done"])},
     ["action"]),
    ("ask_user", "Ask the user a question and wait for the answer. Use only when you cannot "
     "proceed safely without a decision.",
     {"question": P("string", "One question."),
      "options": P("array", "Suggested answers.", items={"type": "string"}, maxItems=4)},
     ["question"]),
    ("web_docs", "Look up a page in the offline documentation mirror (ROCm, HIP, C library, "
     "POSIX). No network access.",
     {"query": P("string", "Search terms."),
      "collection": P("string", "Mirror to search.", enum=["rocm", "hip", "libc", "posix", "gcc"]),
      "max_results": P("integer", "Maximum pages.", default=5)},
     ["query"]),
]


def tool_block():
    out = []
    for name, desc, props, req in TOOLS:
        schema = {"name": name, "description": desc,
                  "parameters": {"type": "object", "properties": props, "required": req,
                                 "additionalProperties": False}}
        out.append(json.dumps(schema, indent=2))
    return "\n\n".join(out)


def est(text, code):
    return int(len(text) / (CODE_CPT if code else PROSE_CPT))


def list_sources(src):
    files = []
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if not d.startswith(".") and d not in ("third_party", "licenses", "gguf"))
        if os.path.relpath(dp, src).startswith(os.path.join("scripts", "d14")):
            continue  # this benchmark's own files must not change the payload
        for f in sorted(fn):
            if f == "D14.md":
                continue
            if f.endswith((".c", ".h", ".cu", ".cuh", ".m", ".py", ".sh", ".md")) or f == "Makefile":
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) < 2000:
                        continue
                    with open(p, encoding="utf-8") as fh:
                        lines = fh.read().splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                files.append((os.path.relpath(p, src), lines))
    return files


def system_prompt(files, target_tokens):
    head = INSTRUCTIONS + "\n" + tool_block() + "\n\n# Repository context\n\n" \
        "Working directory: /work/ds4 (branch `feature/agent-bench`, clean at session start).\n" \
        "Files tracked in the repository (path, lines):\n"
    used = est(INSTRUCTIONS, False) + est(tool_block(), True) + 60
    lines = []
    for path, body in files:
        lines.append(f"  {path} ({len(body)} lines)")
        if used + est("\n".join(lines), True) >= target_tokens:
            break
    return head + "\n".join(lines) + "\n\nThe session starts now.\n"


# --------------------------------------------------------------------------
# Tool results
# --------------------------------------------------------------------------
def window(rng, body, tok_lo, tok_hi):
    """A contiguous line window of a file sized to tok_lo..tok_hi est tokens."""
    want = rng.randint(tok_lo, tok_hi)
    start = rng.randrange(0, max(1, len(body) - 20))
    out, chars = [], 0
    i = start
    while i < len(body) and chars / CODE_CPT < want:
        out.append(body[i])
        chars += len(body[i]) + 8
        i += 1
    return start + 1, out


def big_files(files, min_lines=300):
    return [f for f in files if len(f[1]) >= min_lines and f[0].endswith((".c", ".h", ".cu", ".cuh", ".m"))]


def r_read_file(rng, files, call):
    path, body = rng.choice(big_files(files))
    first, lines = window(rng, body, 500, 4000)
    numbered = "\n".join(f"{first + k:6d}\t{l}" for k, l in enumerate(lines))
    args = json.dumps({"path": path, "offset": first, "limit": len(lines)})
    return "read_file", args, numbered, True


IDENT = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]{3,})\s*\(")


def r_grep(rng, files, call):
    path, body = rng.choice(big_files(files))
    idents = sorted({m.group(1) for l in body for m in IDENT.finditer(l)})
    pat = rng.choice(idents) if idents else "return"
    hits = []
    for p, b in files:
        if not p.endswith((".c", ".h", ".cu", ".cuh")):
            continue
        for n, l in enumerate(b, 1):
            if pat in l:
                hits.append(f"{p}:{n}:{l.strip()}")
    cap = rng.randint(15, 160)
    more = len(hits) - cap
    hits = hits[:cap]
    text = "\n".join(hits) if hits else "(no matches)"
    if more > 0:
        text += f"\n[... {more} more matching lines omitted; narrow the pattern or the path]"
    return "grep", json.dumps({"pattern": pat, "max_results": cap}), text, True


def r_build(rng, files, call):
    srcs = [p for p, _ in files if p.endswith((".c", ".cu"))]
    rng.shuffle(srcs)
    n = rng.randint(6, 30)
    out = ["make -j16 strix-halo"]
    for p in srcs[:n]:
        obj = os.path.splitext(p)[0] + ".o"
        if p.endswith(".cu"):
            out.append(f"hipcc -O3 --offload-arch=gfx1151 -std=c++17 -DDS4_ROCM_BUILD -I. -c {p} -o {obj}")
        else:
            out.append(f"cc -O3 -std=gnu11 -Wall -Wextra -DDS4_ROCM_BUILD -I. -c {p} -o {obj}")
        if rng.random() < 0.18:
            body = dict(files)[p]
            ln = rng.randrange(1, len(body) + 1)
            col = rng.randint(5, 40)
            w = rng.choice(["unused variable 'tmp' [-Wunused-variable]",
                            "comparison of integer expressions of different signedness: 'int' and 'size_t' [-Wsign-compare]",
                            "'n' may be used uninitialized [-Wmaybe-uninitialized]",
                            "unused parameter 'ud' [-Wunused-parameter]"])
            out.append(f"{p}:{ln}:{col}: warning: {w}")
            out.append(f"  {ln:5d} | {body[ln - 1][:110]}")
            out.append("        | " + " " * min(col, 60) + "^")
    for b in ("ds4", "ds4-server", "ds4-bench", "ds4-eval", "ds4-agent"):
        out.append(f"cc -o {b} {b.replace('-', '_')}.o ds4.o ds4_rocm.o ds4_kvstore.o ds4_ssd.o -lamdhip64 -lhipblas -lpthread -lm")
    if rng.random() < 0.15:
        out.append("make: *** [Makefile:212: ds4_server.o] Error 1")
        status = 2
    else:
        status = 0
    text = f"exit status: {status}\n--- stdout+stderr ---\n" + "\n".join(out)
    return "build", json.dumps({"targets": ["strix-halo"], "jobs": 16}), text, True


def r_tests(rng, files, call):
    fns = sorted({m.group(1) for _, b in files for l in b[:4000] for m in IDENT.finditer(l)})
    picks = rng.sample(fns, min(len(fns), rng.randint(20, 120)))
    out, fails = [], 0
    for f in picks:
        ms = rng.uniform(0.1, 900.0)
        if rng.random() < 0.05:
            fails += 1
            out.append(f"[ FAIL ] test_{f} ({ms:.1f} ms)")
            out.append(f"    tests/test_{f.split('_')[0]}.c:{rng.randint(20, 900)}: assertion failed: "
                       f"{f}(&ctx, {rng.randint(0, 64)}) == {rng.randint(0, 4)}")
            out.append(f"    got {rng.randint(5, 4096)}; first differing token at position {rng.randint(0, 512)}")
        else:
            out.append(f"[  OK  ] test_{f} ({ms:.1f} ms)")
    out.append(f"{len(picks)} tests, {len(picks) - fails} passed, {fails} failed")
    return "run_tests", json.dumps({"target": "test-session-state"}), "\n".join(out), True


def r_git_log(rng, files, call):
    scopes = sorted({os.path.splitext(os.path.basename(p))[0].replace("ds4_", "") for p, _ in files})[:60]
    verbs = ["fix", "feat", "perf", "doc", "test", "refactor", "build"]
    what = ["handle short reads in the replica path", "keep the prompt tail resident through prefill",
            "count misses per decode step", "avoid a copy of the router logits", "log KV restore time",
            "align continued checkpoints to 2048 tokens", "size the cache from MemAvailable",
            "skip empty expert rows", "batch the shared-expert matvec", "trim staging slots after prefill",
            "report read wait per layer", "guard the GPU lock in the harness", "retry EINTR in accept",
            "free drafter buffers on session reset", "split long prefill into bounded chunks"]
    n = rng.randint(12, 60)
    out = []
    for _ in range(n):
        h = "%07x" % rng.getrandbits(28)
        out.append(f"{h} {rng.choice(verbs)}({rng.choice(scopes)}): {rng.choice(what)}")
    return "git_log", json.dumps({"max_count": n, "format": "oneline"}), "\n".join(out), False


def mutate(rng, lines):
    out = list(lines)
    for _ in range(rng.randint(2, 8)):
        if not out:
            break
        k = rng.randrange(len(out))
        r = rng.random()
        if r < 0.35:
            out.insert(k, " " * (len(out[k]) - len(out[k].lstrip())) + "/* D14: " +
                       rng.choice(["checked: the caller holds the lock here",
                                   "bounded by ctx_size, see the admission check",
                                   "keep in sync with the parser in record.py",
                                   "this runs once per decode step"]) + " */")
        elif r < 0.6:
            out[k] = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) * 2 or 1), out[k], count=1)
        elif r < 0.8:
            del out[k]
        else:
            out[k] = re.sub(r"\b(tokens|len|n|count|pos)\b", lambda m: m.group(1) + "_total", out[k], count=1)
    return out


def r_diff(rng, files, call):
    path, body = rng.choice(big_files(files))
    first, lines = window(rng, body, 300, 2000)
    new = mutate(rng, lines)
    d = difflib.unified_diff(lines, new, f"a/{path}", f"b/{path}", lineterm="", n=3)
    text = "\n".join(d)
    if not text:
        text = f"(no changes in {path})"
    text = text.replace("@@ -1", f"@@ -{first}", 1)
    return "git_diff", json.dumps({"paths": [path]}), text, True


def r_list_dir(rng, files, call):
    dirs = sorted({os.path.dirname(p) for p, _ in files})
    d = rng.choice(dirs)
    out = [f"total {rng.randint(100, 9000)}"]
    for p, b in files:
        if os.path.dirname(p) == d:
            size = sum(len(l) + 1 for l in b)
            out.append(f"-rw-r--r-- 1 dev users {size:9d} Nov {rng.randint(1, 18):2d} "
                       f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} {os.path.basename(p)}")
    return "list_dir", json.dumps({"path": d or "."}), "\n".join(out), True


RESULTS = [(r_read_file, 0.42), (r_grep, 0.14), (r_build, 0.10), (r_tests, 0.08),
           (r_diff, 0.16), (r_git_log, 0.04), (r_list_dir, 0.06)]

ASKS = [
    "Look at how {fn} in {path} is used and tell me whether it can be called while a prefill is running.",
    "Next: find every caller of {fn} and check that each one handles a non-zero return.",
    "The benchmark shows a slowdown after long prompts. Read {path} around {fn} and tell me what could grow with the context length.",
    "Please review the change you proposed for {fn}. Is it safe if two sessions share the same engine?",
    "Add a key=value log line in {fn} that reports the elapsed time in milliseconds, then build.",
    "Summarise in a few sentences what {path} is responsible for and which functions are its entry points.",
    "Check whether {fn} can overflow when the context is above 128k tokens.",
    "Run the session-state tests and tell me if anything regressed.",
    "Write a short plan for removing the extra copy in {fn}; do not change code yet.",
    "Is {fn} on the decode hot path? Give me the call chain from the server down to it.",
    "Compare the error handling in {fn} with the convention in the system prompt and list the differences.",
    "OK, that looks right. Continue with the next step of the plan.",
]


def instruction(rng, files):
    path, body = rng.choice(big_files(files))
    fns = sorted({m.group(1) for l in body for m in IDENT.finditer(l)}) or ["main"]
    t = rng.choice(ASKS).format(fn=rng.choice(fns), path=path)
    if rng.random() < 0.5:
        t += " " + rng.choice([
            "Keep the change minimal and do not touch the GPU kernels.",
            "Cite file and line for every claim.",
            "If you need more context, ask for the specific lines rather than whole files.",
            "We will benchmark this later, so note anything that could change the token output.",
            "Remember that the server log format is parsed by the harness."])
    return t


def make_event(rng, files, i, call_counter):
    r = rng.random()
    parts, kinds, est_tok = [], [], 0
    if r < 0.10:
        t = instruction(rng, files)
        parts.append(t); kinds.append("instruction"); est_tok += est(t, False)
    else:
        if r < 0.25:
            t = instruction(rng, files)
            parts.append(t); kinds.append("instruction"); est_tok += est(t, False)
            n = 1
        else:
            n = rng.choices([1, 2, 3], [0.45, 0.35, 0.20])[0]
        fns, ws = zip(*RESULTS)
        for _ in range(n):
            f = rng.choices(fns, ws)[0]
            call_counter[0] += 1
            name, args, body, code = f(rng, files, call_counter[0])
            block = (f'<tool_result name="{name}" call_id="call_{call_counter[0]:04d}" arguments=\'{args}\'>\n'
                     f"{body}\n</tool_result>")
            parts.append(block); kinds.append(name); est_tok += est(block, code)
    max_tokens = rng.choices([256, 64, 512], [0.70, 0.15, 0.15])[0]
    return {"i": i, "items": kinds, "text": "\n\n".join(parts), "max_tokens": max_tokens,
            "est_tokens": est_tok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="public source tree used for tool results")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--system-tokens", type=int, default=6500, help="target est tokens of the system prompt")
    ap.add_argument("--session-tokens", type=int, default=260000, help="est tokens of events per session")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = list_sources(a.src)
    if not files:
        sys.exit("no source files under --src")
    commit = None
    try:
        commit = subprocess.run(["git", "-C", a.src, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        pass
    system = system_prompt(files, a.system_tokens)
    sessions = []
    for s in range(a.sessions):
        rng = random.Random(a.seed * 1000 + s)
        events, total, counter = [], 0, [0]
        while total < a.session_tokens:
            ev = make_event(rng, files, len(events), counter)
            events.append(ev)
            total += ev["est_tokens"] + ev["max_tokens"]
        sessions.append({"id": s, "seed": a.seed * 1000 + s, "events": events})
    doc = {"version": 1, "generator": "d14 gen_payload.py", "seed": a.seed, "src": a.src,
           "src_commit": commit, "system": system, "system_est_tokens": est(system, True),
           "sessions": sessions}
    raw = json.dumps(doc, ensure_ascii=False, indent=0).encode()
    with open(a.out, "wb") as f:
        f.write(raw)
    print(f"wrote {a.out} sha256={hashlib.sha256(raw).hexdigest()[:12]} system_chars={len(system)} "
          f"system_est={doc['system_est_tokens']}")
    for s in sessions:
        ev = s["events"]
        sz = sorted(e["est_tokens"] for e in ev)
        print(f"session {s['id']}: events={len(ev)} est_tokens={sum(sz)} median={sz[len(sz)//2]} "
              f"max={sz[-1]} kinds={sum(len(e['items']) for e in ev)}")


if __name__ == "__main__":
    main()
