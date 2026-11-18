"""D11: option (1b-lru): no static hot set, an LRU of S drafter experts per
stage (resident), the rest read on demand at 2.9 ms. ESTIMATES."""
import sys
sys.path.insert(0, "/data/ds4-d11/scripts/d11"); sys.path.insert(0, "/data/ds4-d9/scripts/d9")
import importlib.util
spec = importlib.util.spec_from_file_location("reuse", "/data/ds4-d11/scripts/d11/reuse.py")
src = open("/data/ds4-d11/scripts/d11/reuse.py").read().replace("\nmain()\n", "\n")
ns = {}
exec(compile(src, "reuse", "exec"), ns)
from d9lib import record, run_dir
tags = {p: "d11-trace-mx-" + p for p in ns["P5"]}; tags["q2-code"] = "d11-trace-q2-code"
print("| run | baseline | " + " | ".join("LRU %d/stage" % s for s in (24, 32, 48, 64)) + " |")
print("|---|---:|" + "---:|" * 4)
for k, t in tags.items():
    v = ns["load"](t); tps = record(t)["decode_tps"]; cells = []
    for S in (24, 32, 48, 64):
        miss = sum(ns["lru_sim"](v, s, S)[1] for s in range(3))
        freed = (128 - S) * 3 * ns["GIB_PER_EXPERT"]
        tp = 511 / (511 / (tps + 0.12 * freed) + miss * 2.9 / 1000)
        cells.append("%.2f (%+.1f %%; %d misses, %.2f GiB)" % (tp, 100 * (tp / tps - 1), miss, freed))
    print("| %s | %.2f | " % (k, tps) + " | ".join(cells) + " |")
