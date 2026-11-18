"""D10 Part B: drafter tensor inventory from the DSpark GGUF table."""
import json, re
OUT = "/data/ds4-d9/scripts/d10/out"
d = json.load(open(f"{OUT}/dspark_table.json"))
GiB = 2**30
def use(n):
    if "_exps" in n: return ("GPU (resident image)", "only the selected experts: <=15 of 128 per stage per proposal (5 rows x top-3)")
    if "main_proj" in n: return ("GPU", "not by the proposal; by the per-token drafter KV bookkeeping (every committed token)")
    if "attn_kv.weight" in n: return ("GPU", "every proposal + every committed token (bookkeeping)")
    if "markov_w2" in n: return ("GPU (resident image)", "NOT read: the ROCm F32 path reads a transposed copy (extra 126 MiB allocation)")
    if "markov_w1" in n: return ("GPU", "one 1 KiB row per Markov step (5 per proposal)")
    if "confidence_head" in n: return ("GPU (+ 21 KiB copy in conf_proj)", "copy read per proposal")
    if "exp_probs_b_vl" in n: return ("GPU", "never (vision-token routing bias; text only)")
    return ("GPU", "every proposal")
groups = {}
for t in d["tensors"]:
    n = re.sub(r"mtp\.\d\.", "mtp.N.", t["name"])
    g = groups.setdefault(n, dict(n=0, bytes=0, dims=t["dims"], type=t["type"]))
    g["n"] += 1; g["bytes"] += t["bytes"]
print("| tensor (x stages) | shape | type | bytes each | total GiB | on GPU | read |")
print("|---|---|---|---:|---:|---|---|")
tot = 0
for n, g in sorted(groups.items(), key=lambda x: -x[1]["bytes"]):
    loc, rd = use(n)
    tot += g["bytes"]
    print(f"| {n} (x{g['n']}) | {'x'.join(map(str, g['dims']))} | {g['type']} | {g['bytes'] // g['n']:,} | {g['bytes'] / GiB:.4f} | {loc} | {rd} |")
print(f"| total | | | | {tot / GiB:.3f} | | |")
cat = {"routed experts": "_exps", "attention (q_a,q_b,kv,out_a,out_b)": "attn_(q_a|q_b|kv|output_a|output_b)\\.", "shared expert": "_shexp",
       "main_proj": "main_proj", "markov w1": "markov_w1", "markov w2": "markov_w2"}
for k, rx in cat.items():
    b = sum(t["bytes"] for t in d["tensors"] if re.search(rx, t["name"]))
    print(f"{k}: {b / GiB:.3f} GiB")
