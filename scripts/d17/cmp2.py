"""Token identity per turn between two runs (D8_STEP token IDs), by step/turn.
  cmp2.py RUN_DIR_A RUN_DIR_B"""
import json, os, re, sys
def rows(d): return [json.loads(l) for l in open(os.path.join(d, "turns.jsonl"))]
def toks(d, off):
    with open(os.path.join(d, "server.log"), "rb") as f:
        f.seek(off[0]); seg = f.read(off[1] - off[0]).decode("utf-8", "replace")
    return [int(m) for m in re.findall(r"D8_STEP pos=\d+ tok=(\d+)", seg)]
a, b = sys.argv[1], sys.argv[2]
rb = {(r["step"], r["turn"]): r for r in rows(b)}
for r in rows(a):
    o = rb.get((r["step"], r["turn"]))
    if not o: continue
    ta, tb = toks(a, r["log_off"]), toks(b, o["log_off"])
    n = 0
    while n < min(len(ta), len(tb)) and ta[n] == tb[n]: n += 1
    print(r["step"], r["turn"], f"{n}/{min(len(ta), len(tb))}", "same text" if r["reply_sha"] == o["reply_sha"] else "text differs")
