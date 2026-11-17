"""D5 stage 1: parse a DS4_ROCM_D5_TRACE file (+ the run's run.log) and replay
the resident expert cache.  Stdlib only.

Trace lines (see rocm/ds4_rocm_runtime.cuh, ds4.c d5_probe):
  C <budget> <slot_count> <n_resident>      one-time snapshot header
  R <layer> <expert>                        snapshot entries, LRU order (oldest first)
  S <layer> <id0..id5> <miss_mask>          one single-token selected load
  P0|P1|P2 <target_layer> <us> <id0..id15>  early-router probe (DS4_ROCM_D5_PROBE)

A token is a run of S lines for layers 0..N-1.  Probe lines are attached to the
next S line of their target layer.
"""
import collections
import re

N_LAYER = 40


class Step:
    __slots__ = ("layer", "ids", "miss", "probes")

    def __init__(self, layer, ids, miss):
        self.layer = layer
        self.ids = ids
        self.miss = miss          # set of expert ids that missed (recorded)
        self.probes = {}          # tag -> (us, [16 ids])


def parse_trace(path):
    snap_budget = None
    snap = []
    steps = []
    pending = {}                  # (tag, layer) -> (us, ids)
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            k = p[0]
            if k == "S":
                layer = int(p[1])
                ids = [int(x) for x in p[2:8]]
                mask = int(p[8])
                miss = {ids[i] for i in range(6) if mask >> i & 1}
                st = Step(layer, ids, miss)
                for tag in ("P0", "P1", "P2"):
                    v = pending.pop((tag, layer), None)
                    if v is not None:
                        st.probes[tag] = v
                steps.append(st)
            elif k in ("P0", "P1", "P2"):
                pending[(k, int(p[1]))] = (float(p[2]), [int(x) for x in p[3:]])
            elif k == "C":
                snap_budget = (int(p[1]), int(p[2]), int(p[3]))
            elif k == "R":
                snap.append((int(p[1]), int(p[2])))
    return snap_budget, snap, steps


PERF_RE = re.compile(r"perf token (\d+) pos (\d+) ms ([0-9.]+) slots (\d+) misses (\d+) "
                     r"wait_ms ([0-9.]+) jobs (\d+) id (\d+)")


def parse_perf(run_log):
    out = []
    with open(run_log, errors="replace") as f:
        for line in f:
            m = PERF_RE.search(line)
            if m:
                out.append(dict(tok=int(m.group(1)), pos=int(m.group(2)), ms=float(m.group(3)),
                                misses=int(m.group(5)), wait_ms=float(m.group(6)),
                                id=int(m.group(8))))
    return out


def tokens_from_steps(steps):
    toks = []
    cur = None
    for st in steps:
        if st.layer == 0:
            cur = []
            toks.append(cur)
        if cur is None:
            raise ValueError("trace does not start at layer 0")
        cur.append(st)
    for t in toks:
        if [s.layer for s in t] != list(range(len(t))):
            raise ValueError("token with layers out of order")
    return toks


class LRU:
    """Replay of cuda_stream_resident_* for single-token loads: per selected
    slot, hit -> touch; miss -> evict the least recently used entry that is
    not one of the current layer's selected experts (when full), insert.
    Then cuda_stream_selected_prepare_ptrs touches all six in slot order."""

    def __init__(self, capacity, snapshot=()):
        self.cap = capacity
        self.od = collections.OrderedDict()
        for key in snapshot:
            self.od[key] = None

    def __contains__(self, key):
        return key in self.od

    def access(self, layer, ids):
        cur = {(layer, e) for e in ids}
        missed = set()
        for e in ids:
            key = (layer, e)
            if key in self.od:
                self.od.move_to_end(key)
                continue
            missed.add(e)
            while len(self.od) >= self.cap:
                victim = None
                for kk in self.od:
                    if kk not in cur:
                        victim = kk
                        break
                if victim is None:
                    break
                del self.od[victim]
            self.od[key] = None
        for e in ids:
            self.od.move_to_end((layer, e))
        return missed


def load_run(trace_path, run_log):
    budget, snap, steps = parse_trace(trace_path)
    toks = tokens_from_steps(steps)
    perf = parse_perf(run_log)
    n_dec = len(perf)
    if n_dec > len(toks):
        raise ValueError("more perf tokens than traced tokens")
    first_dec = len(toks) - n_dec
    # validation 1: per-token misses in the trace equal the perf line's misses
    bad = 0
    for k, pr in enumerate(perf):
        tm = sum(len(s.miss) for s in toks[first_dec + k])
        if tm != pr["misses"]:
            bad += 1
    # validation 2: LRU replay from the snapshot reproduces every miss set
    lru = LRU(budget[0] if budget else 10 ** 9, snap)
    replay_bad = 0
    for t in toks:
        for s in t:
            if lru.access(s.layer, s.ids) != s.miss:
                replay_bad += 1
    return dict(budget=budget, snap=snap, toks=toks, perf=perf, first_dec=first_dec,
                perf_mismatch=bad, replay_mismatch=replay_bad,
                n_steps=sum(len(t) for t in toks))
