"""D9: parse DSpark per-round traces from ds4 run logs.

Sources (stderr lines, possibly preceded on the same line by generated text):
  DS41_PHASE_COST  (DS4_ROCM_DSPARK_PHASE_TRACE=1)  one per round
  DS41_SCHED       (DS4_DSPARK_SCHED_TRACE=1, online/model only) one per round
  D8_STEP          (DS4_D8_STEP_TRACE=1) one per ds4_session_eval (plain step)
  D9_ROUND         (DS4_D9_ROUND_TRACE=1, d9 branch) one per round, all modes
"""
import glob
import json
import math
import os
import re

RUNS = "/data/ds4-bench/runs"
RESULTS = "/data/ds4-bench/results.jsonl"

_kv = re.compile(r"([A-Za-z_0-9]+)=(\S+)")


def _num(v):
    try:
        if v.startswith("0x") or v.startswith("-0x"):
            return float.fromhex(v)
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return float(v)
    except ValueError:
        return v


def _parse_line(rest):
    d = {}
    for k, v in _kv.findall(rest):
        if k in ("confidence", "conf", "drafts", "caps"):
            d[k] = [] if v == "-" else [_num(x) for x in v.split(",") if x]
        else:
            d[k] = _num(v)
    return d


def run_dir(tag):
    ds = sorted(glob.glob(f"{RUNS}/*-{tag}"))
    ds = [d for d in ds if os.path.basename(d).split("-", 1)[1] == tag]
    return ds[-1] if ds else None


def record(tag):
    rec = None
    with open(RESULTS) as f:
        for line in f:
            if f'"tag":"{tag}"' in line or f'"tag": "{tag}"' in line:
                rec = json.loads(line)
    return rec


def parse(path):
    """Return dict(phase=[...], sched=[...], step=[...], d9=[...], probe=[...])."""
    if os.path.isdir(path):
        path = os.path.join(path, "run.log")
    out = {"phase": [], "sched": [], "step": [], "d9": [], "text": None}
    tags = {"DS41_PHASE_COST v=1 ": "phase", "DS41_SCHED v=1 ": "sched",
            "D8_STEP ": "step", "D9_ROUND v=1 ": "d9"}
    with open(path, errors="replace") as f:
        for line in f:
            for t, key in tags.items():
                i = line.find(t)
                if i >= 0:
                    out[key].append(_parse_line(line[i + len(t):]))
                    break
            else:
                if line.startswith("ds4: DSpark stats"):
                    out["stats"] = _parse_line(line)
                elif line.startswith("ds4: DSpark sched final"):
                    out["sched_final"] = line.strip()
                elif line.startswith("ds4: prefill:"):
                    m = re.search(r"generation: ([0-9.]+)", line)
                    out["gen_tps"] = float(m.group(1)) if m else None
    return out


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def lstsq(X, y, w=None):
    """Weighted least squares without numpy. X: list of rows (no intercept
    column added). Returns (coef, r2, rmse)."""
    n, k = len(X), len(X[0])
    w = w or [1.0] * n
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi, wi in zip(X, y, w):
        for i in range(k):
            b[i] += wi * xi[i] * yi
            for j in range(k):
                A[i][j] += wi * xi[i] * xi[j]
    # Gaussian elimination with partial pivoting
    M = [A[i] + [b[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    coef = [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]
    pred = [sum(c * x for c, x in zip(coef, xi)) for xi in X]
    sw = sum(w)
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    ss_res = sum(wi * (yi - pi) ** 2 for wi, yi, pi in zip(w, y, pred))
    ss_tot = sum(wi * (yi - my) ** 2 for wi, yi in zip(w, y))
    return coef, (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), math.sqrt(ss_res / sw)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
