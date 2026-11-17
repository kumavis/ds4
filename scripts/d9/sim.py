"""D9 offline replay of DSpark scheduling policies on probe data.

A probe run (DS4_D9_PROBE=1) drafts at every position of the plain greedy
trajectory and logs the drafter's block (tokens + confidence logits).  From
it, A[p] = number of leading drafted tokens at p that equal the plain
trajectory's tokens p+1.., i.e. what a verify of width w at p accepts:
min(w, A[p]) (greedy, exact-verify semantics).

A policy is replayed position by position with a cost model:
  plain step                    P
  drafted round, width 0        D + P
  drafted round, width w >= 1   D + V(w + 1),  V(r) = a + b r + c * mu * r
(mu = expert misses per verified row of that prompt).
"""
import math
import sys

sys.path.insert(0, "/data/ds4-d9/scripts/d9")
from d9lib import parse, run_dir, sigmoid, lstsq, mean  # noqa: E402

BLOCK = 5


def load_probe(tag):
    r = parse(run_dir(tag))
    rounds = r["d9"]
    seq = {}
    info = {}
    for x in rounds:
        seq[x["pos"]] = x["seed"]
        info[x["pos"]] = x
    p0 = min(seq)
    n = len(seq)
    toks = [seq[p0 + i] for i in range(n)]
    A, conf, drafts, ok = [], [], [], []
    for i in range(n):
        x = info[p0 + i]
        if x["proposed"]:
            d = x["drafts"]
            a = 0
            while a < len(d) and i + 1 + a < n and d[a] == toks[i + 1 + a]:
                a += 1
            A.append(a)
            conf.append(x["conf"])
            drafts.append(d)
            ok.append(True)
        else:
            A.append(0)
            conf.append([float("-inf")] * BLOCK)
            drafts.append([])
            ok.append(False)
    plain = [info[p0 + i]["plain_ms"] for i in range(n)]
    pmiss = [info[p0 + i]["jobs"] / 3.0 for i in range(n)]
    pwait = [info[p0 + i]["wait_ms"] for i in range(n)]
    prop = [info[p0 + i]["prop_ms"] for i in range(n) if info[p0 + i]["proposed"]]
    return dict(tag=tag, p0=p0, n=n, toks=toks, A=A, conf=conf, drafts=drafts,
                draftable=ok, plain_ms=plain, plain_miss=pmiss, plain_wait=pwait,
                prop_ms=prop, gen_tps=r.get("gen_tps"))


# ---------------------------------------------------------------- cost model
class Cost:
    """Round costs in ms.  scale_wait scales the read-wait part (0 = no read
    wait, 0.5 = halved)."""

    def __init__(self, P, D, Va, Vb, Pw=0.0, Vwa=0.0, Vwb=0.0, scale_wait=1.0):
        self.P_nr, self.Pw = P - Pw, Pw
        self.Va_nr, self.Vb_nr = Va - Vwa, Vb - Vwb
        self.Vwa, self.Vwb = Vwa, Vwb
        self.D = D
        self.set_scale(scale_wait)

    def set_scale(self, s):
        self.s = s
        self.P = self.P_nr + s * self.Pw
        self.Va = self.Va_nr + s * self.Vwa
        self.Vb = self.Vb_nr + s * self.Vwb
        return self

    def V(self, rows):
        return self.Va + self.Vb * rows

    def round(self, proposed, w):
        if not proposed:
            return self.P
        return self.D + (self.P if w == 0 else self.V(w + 1))


# ---------------------------------------------------------------- policies
RESIDENT_COST = [100, 137, 161, 181, 215, 229]


def old_prefix_len(conf):
    survival, expected, selected = 1.0, 1.0, 0
    for i, c in enumerate(conf[:BLOCK]):
        if not math.isfinite(c):
            break
        survival *= sigmoid(c)
        cand = expected + survival
        cur, nxt = RESIDENT_COST[selected], RESIDENT_COST[i + 1] if i + 1 < len(RESIDENT_COST) else None
        if nxt is None or cand * cur <= expected * nxt:
            break
        expected, selected = cand, i + 1
    return selected


class OldPolicy:
    """ds41_draft_should_propose + ds41_draft_scheduled_prefix_len +
    ds41_draft_adapt (64-cycle backoff)."""
    name = "old"

    def __init__(self, backoff=True, min_width=0):
        self.skip_left = 0
        self.cyc = self.dr = self.acc = 0
        self.backoff, self.min_width = backoff, min_width

    def propose(self, p):
        if self.backoff and self.skip_left:
            self.skip_left -= 1
            return False
        return True

    def width(self, conf):
        return max(self.min_width, old_prefix_len(conf))

    def note(self, proposed, w, matched, kept):
        if not proposed or not self.backoff:
            return
        self.cyc += 1
        self.dr += w
        self.acc += matched
        if self.cyc < 4:
            return
        acceptance = self.acc / self.dr if self.dr else 0.0
        if self.cyc < 8 and acceptance >= 0.25:
            return
        if acceptance < 0.48 or self.acc < self.cyc:
            self.skip_left = 64
        self.cyc = self.dr = self.acc = 0


class FixedPolicy:
    def __init__(self, w):
        self.w = w
        self.name = f"fixed{w}"

    def propose(self, p):
        return True

    def width(self, conf):
        return self.w

    def note(self, *a):
        pass


class ModelPolicy:
    """Mirror of ds41_model_* in ds4.c (DS4_DSPARK_SCHED=model)."""
    name = "model"

    def __init__(self, P=118.0, D=20.0, VA=77.0, VB=49.0, kappa=1.1, cal_scale=1.0,
                 bias=(0, 0, 0, 0, 0), eta=0.0, halflife=1.0, gate=-0.1, skip=8,
                 skip_max=64, max_width=5, predicted=True):
        self.P, self.D, self.VA, self.VB = P, D, VA, VB
        self.kappa, self.cal_scale, self.eta = kappa, cal_scale, eta
        self.decay = 0.5 ** (1.0 / halflife)
        self.gate, self.skip, self.skip_max = gate, skip, max(skip, skip_max)
        self.max_width, self.predicted = max_width, predicted
        self.bias = list(bias) + [0.0] * (8 - len(bias))
        self.plain_left = 0
        self.fails = 0
        self.probe = False
        self.adv_v = self.adv_w = 0.0
        self.expected = 1.0
        self.conf = None

    def cost(self, k):
        return self.VA + self.VB * (k + 1) if k else self.P

    def accept_p(self, i, c):
        if not math.isfinite(c):
            return 0.0
        p = sigmoid(self.cal_scale * c + self.bias[i])
        return min(0.999, max(0.001, p))

    def propose(self, p):
        if self.plain_left:
            self.plain_left -= 1
            if not self.plain_left:
                self.probe = True
            return False
        return True

    def width(self, conf):
        rho = self.kappa / self.P
        survive, expected = 1.0, 1.0
        best_gain, best, best_e = 1.0 - rho * self.P, 0, 1.0
        for i in range(min(len(conf), self.max_width)):
            if not math.isfinite(conf[i]):
                break
            survive *= self.accept_p(i, conf[i])
            expected += survive
            gain = expected - rho * self.cost(i + 1)
            if gain > best_gain:
                best, best_gain, best_e = i + 1, gain, expected
        self.conf = conf
        return best

    def expected_for(self, conf, k):
        survive, e = 1.0, 1.0
        for i in range(k):
            survive *= self.accept_p(i, conf[i])
            e += survive
        return e

    def note(self, proposed, k, matched, kept):
        if not proposed:
            return
        conf = self.conf
        expected = self.expected_for(conf, k)
        if self.eta > 0:
            for i in range(min(k, matched + 1)):
                a = 1.0 if i < matched else 0.0
                p = self.accept_p(i, conf[i])
                self.bias[i] = max(-4.0, min(4.0, self.bias[i] + self.eta * (a - p)))
        tokens = expected if self.predicted else kept
        adv = (tokens - 1.0) - (self.D + self.cost(k) - self.P) / self.P
        if self.probe:
            self.probe = False
            self.adv_v = self.adv_w = 0.0
        self.adv_v = self.adv_v * self.decay + adv
        self.adv_w = self.adv_w * self.decay + 1.0
        if self.adv_v / self.adv_w < self.gate:
            s = self.skip
            for _ in range(self.fails):
                if s >= self.skip_max:
                    break
                s *= 2
            s = min(s, self.skip_max)
            self.plain_left = s
            self.fails += 1
            if not s:
                self.probe = True
        else:
            self.fails = 0


def replay(pr, pol, cost, ntok=None):
    """Replay policy pol on probe pr.  Returns dict with tok/s and counts."""
    n = pr["n"] if ntok is None else ntok
    p, t, rounds, drafted, accepted, plain = 0, 0.0, 0, 0, 0, 0
    widths = {}
    while p < n:
        draftable = pr["draftable"][p] and p + 1 < n
        proposed = draftable and pol.propose(p)
        w = 0
        if proposed:
            w = pol.width(pr["conf"][p])
            w = min(w, n - p - 1)
        matched = min(w, pr["A"][p])
        kept = 1 + matched
        t += cost.round(proposed, w)
        pol.note(proposed, w, matched, kept)
        rounds += 1
        if proposed and w:
            drafted += 1
            accepted += matched
        else:
            plain += 1
        widths[w if proposed else -1] = widths.get(w if proposed else -1, 0) + 1
        p += kept
    return dict(tps=1000.0 * n / t, ms=t, rounds=rounds, verify=drafted,
                accepted=accepted, plain=plain, widths=widths)


def oracle(pr, cost, wmax=5, allow_plain=True, ntok=None):
    """Exact DP: minimum modelled time to commit the n tokens, knowing A[p]."""
    n = pr["n"] if ntok is None else ntok
    T = [0.0] * (n + 1)
    choice = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        best, bw = (cost.P + T[p + 1]) if allow_plain else float("inf"), 0
        if pr["draftable"][p]:
            for w in range(1, min(wmax, n - p - 1) + 1):
                v = cost.D + cost.V(w + 1) + T[p + 1 + min(w, pr["A"][p])]
                if v < best:
                    best, bw = v, w
            if not allow_plain and bw == 0 and n - p - 1 >= 1:
                pass
        if best == float("inf"):
            best, bw = cost.D + cost.P + T[p + 1], 0
        T[p], choice[p] = best, bw
    # walk
    p, rounds, acc, widths = 0, 0, 0, {}
    while p < n:
        w = choice[p]
        widths[w] = widths.get(w, 0) + 1
        m = min(w, pr["A"][p]) if w else 0
        acc += m
        rounds += 1
        p += 1 + m
    return dict(tps=1000.0 * n / T[0], ms=T[0], rounds=rounds, accepted=acc, widths=widths)
