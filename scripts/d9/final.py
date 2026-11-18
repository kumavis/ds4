"""D9: measured results summary (results.jsonl)."""
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import record, mean  # noqa: E402

P = ["code", "prose", "json", "translate", "list"]


def g(tag, f="decode_tps"):
    r = record(tag)
    return None if r is None else r.get(f)


def fm(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


print("| prompt | plain (int) | old r1 / r2 | model r1 / r2 | online r1 | model / old |")
print("|---|---:|---:|---:|---:|---:|")
ratios = []
for p in P:
    pl = mean([g(f"int-mx-{p}-plain-r1"), g(f"int-mx-{p}-plain-r2")])
    o = [g(f"d9-mx-{p}-old-r{i}") for i in (1, 2)]
    m = [g(f"d9-mx-{p}-model-r{i}") for i in (1, 2)]
    on = g(f"d9-mx-{p}-online-r1")
    om, mm = mean([x for x in o if x]), mean([x for x in m if x])
    ratios.append(mm / om)
    print(f"| {p} | {pl:.2f} | {fm(om)} ({fm(o[0])}, {fm(o[1])}) | {fm(mm)} ({fm(m[0])}, {fm(m[1])}) | {fm(on)} | {mm/om:.3f} |")
print(f"\nmean model/old ratio {mean(ratios):.3f}")
print("\n| tag | tok/s | rounds | accepted | misses | first_div | common_prefix | tf_agree | tf_flips | bitexact | avail_min_mb | nvme_max_c |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
tags = [f"d9-mx-{p}-{s}-r{i}" for i in (1, 2) for p in P for s in ("old", "model")] + \
       [f"d9-mx-{p}-online-r1" for p in P] + ["d9-q2-code-old", "d9-q2-code-model", "d9-q2-code-online",
                                               "d9-gate-model-exact-code", "d9-gate-model-exact-prose",
                                               "d9-b2-gate-mx-code", "d9-b2-gate-q2-code",
                                               "d9-probe-old-traj-mx-code", "d9-probe-old-traj-mx-translate"]
for t in tags:
    r = record(t)
    if not r:
        print(f"| {t} | missing |")
        continue
    print(f"| {t} | {fm(r.get('decode_tps'))} | {r.get('dspark_cycles') or '-'} | {r.get('dspark_accepted') if r.get('dspark_accepted') is not None else '-'} | "
          f"{r.get('decode_misses')} | {r.get('div_first_div') if r.get('div_first_div') is not None else ('none' if r.get('div_compared') else '-')} | "
          f"{r.get('div_common_prefix') or '-'} | {fm(r.get('div_tf_agree'), 4)} | {r.get('div_tf_flips') if r.get('div_tf_flips') is not None else '-'} | "
          f"{fm(r.get('div_bitexact_frac'), 4)} | {r.get('avail_min_mb')} | {r.get('nvme_temp_max_c')} |")
