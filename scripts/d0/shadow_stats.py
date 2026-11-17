#!/usr/bin/env python3
"""Summarise DS4_D0_SHADOW runs: batched vs exact verify rows on identical state.

Usage: shadow_stats.py RUN_DIR...   (reads "ds4: D0 row" lines from run.log)
Per run: rows, argmax flips, flip fraction, exact-row top-2 margin at flips,
logit max-abs diff (median / p90 / max), and the first differing tap
(component) histogram over rows.
"""
import collections, re, statistics, sys

ROW = re.compile(r"ds4: D0 row .*?logit_maxabs=(\S+) .*?flip=(\d) margin_s=(\S+) first=(\S+)@")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main(dirs):
    allm, allf, allmarg, small = [], 0, [], 0
    for d in dirs:
        rows = [ROW.search(l) for l in open(f"{d}/run.log", errors="replace")]
        rows = [m for m in rows if m]
        if not rows:
            continue
        mx = [float(m[1]) for m in rows]
        flips = [float(m[3]) for m in rows if m[2] == "1"]
        margins = [float(m[3]) for m in rows]
        first = collections.Counter(m[4] for m in rows)
        allm += mx; allf += len(flips); allmarg += flips
        small += sum(1 for x in margins if x < 0.25)
        print(f"{d.rsplit('/', 1)[-1]}: rows={len(rows)} flips={len(flips)} ({len(flips)/len(rows):.2%}) "
              f"flip_margins={sorted(round(x, 3) for x in flips)} maxabs med={statistics.median(mx):.3f} "
              f"p90={pct(mx, .9):.3f} max={max(mx):.3f} rows_margin<0.25={sum(1 for x in margins if x < .25)} "
              f"first={dict(first.most_common(4))}")
    if allm:
        print(f"TOTAL rows={len(allm)} flips={allf} ({allf/len(allm):.2%}) flip_margin_max="
              f"{max(allmarg) if allmarg else None} maxabs med={statistics.median(allm):.3f} "
              f"p90={pct(allm, .9):.3f} p99={pct(allm, .99):.3f} max={max(allm):.3f} rows_margin<0.25={small}")


if __name__ == "__main__":
    main(sys.argv[1:])
