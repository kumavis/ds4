"""D10 Part B: byte-compare drafter tensors against main-model tensors.

Run under the GPU lock (it reads tensor data from /data/ds4/gguf):
  flock /data/ds4-bench/gpu.lock nix-shell ... python3 bytecmp.py
Reads (total ~0.15 GB):
  1. the three drafter mtp.N.attn_kv F16 [5120,512] tensors (5.2 MB each) and
     the seven same-shape/type main tensors (blk.{2,8,..}.attn_compressor_*),
     whole, sha256;
  2. the first 64 KiB of every routed expert slice of the drafter's
     gate/up/down tensors (3 stages x 3 x 128) and of the main model's
     blk.37-39 (the drafter's target layers) gate/up/down (3 x 3 x 384).
"""
import hashlib
import json
import os

OUT = "/data/ds4-d9/scripts/d10/out"
DS = "/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf"
MX = "/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf"
PREFIX = 65536


def table(name):
    return {t["name"]: t for t in json.load(open(f"{OUT}/{name}"))["tensors"]}


def read(fd, off, n):
    return os.pread(fd, n, off)


ds, mx = table("dspark_table.json"), table("mxfp4_table.json")
fds, fmx = os.open(DS, os.O_RDONLY), os.open(MX, os.O_RDONLY)
total = 0
res = {"whole": {}, "expert_prefix_matches": [], "expert_prefixes": 0}
sig = lambda t: (tuple(t["dims"]), t["type"])
for n, t in ds.items():
    if not n.endswith("attn_kv.weight"):
        continue
    h = hashlib.sha256(read(fds, t["abs_off"], t["bytes"])).hexdigest()
    total += t["bytes"]
    res["whole"][n] = h
    for m, u in mx.items():
        if sig(u) == sig(t) and u["bytes"] == t["bytes"]:
            if m not in res["whole"]:
                res["whole"][m] = hashlib.sha256(read(fmx, u["abs_off"], u["bytes"])).hexdigest()
                total += u["bytes"]
main_h = {}
for L in (37, 38, 39):
    for part in ("gate", "up", "down"):
        u = mx[f"blk.{L}.ffn_{part}_exps.weight"]
        per = u["bytes"] // 384
        for e in range(384):
            main_h.setdefault(hashlib.sha256(read(fmx, u["abs_off"] + e * per, PREFIX)).digest(), []).append(
                f"blk.{L}.{part}.{e}")
            total += PREFIX
for s in (0, 1, 2):
    for part in ("gate", "up", "down"):
        t = ds[f"mtp.{s}.ffn_{part}_exps.weight"]
        per = t["bytes"] // 128
        for e in range(128):
            h = hashlib.sha256(read(fds, t["abs_off"] + e * per, PREFIX)).digest()
            total += PREFIX
            res["expert_prefixes"] += 1
            if h in main_h:
                res["expert_prefix_matches"].append((f"mtp.{s}.{part}.{e}", main_h[h]))
dsk = [k for k in res["whole"] if k.startswith("mtp.")]
res["attn_kv_equal_to_main"] = {k: [m for m, h in res["whole"].items() if not m.startswith("mtp.") and h == res["whole"][k]]
                                for k in dsk}
res["bytes_read"] = total
json.dump(res, open(f"{OUT}/bytecmp.json", "w"), indent=1)
print("bytes read", total, "attn_kv matches", res["attn_kv_equal_to_main"],
      "expert prefix matches", len(res["expert_prefix_matches"]), "of", res["expert_prefixes"])
