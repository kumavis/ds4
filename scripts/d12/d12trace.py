"""D12: parse a DS4_D12_CACHE_TRACE file (optionally .gz) into a call stream.

Trace format: see the D12 comment in rocm/ds4_rocm_runtime.cuh.

A call is one resident-cache call (one layer of one step / verify round /
seed).  Fields:
  src     's' single-token load, 'b' batched (verify) load,
          'd' seed from layer selection, 'e' seed of a priority list
  layer, ids (protected set, list), acc (list of (key, was_hit) in lookup
  order), ev (keys evicted while serving the call), touches (keys bumped
  without lookup, in order, interleaved info kept in `seq`)
  mark    index of the phase mark (M line) this call belongs to
Keys are layer * 384 + expert.
"""
import gzip

NE = 384


def key(layer, expert):
    return layer * NE + expert


class Mark:
    __slots__ = ("kind", "pos", "rows", "budget", "slots", "resident", "free", "idx")

    def __init__(self, f, idx):
        self.kind = f[1]
        self.pos, self.rows, self.budget, self.slots, self.resident, self.free = map(int, f[2:8])
        self.idx = idx


class Call:
    __slots__ = ("src", "map", "layer", "ntok", "nsel", "ids", "seq", "mark")

    def __init__(self, f, mark):
        self.src = f[1]
        self.map = int(f[2])
        self.layer = int(f[3])
        self.ntok = int(f[4])
        self.nsel = int(f[5])
        self.ids = [int(x) for x in f[6:]]
        self.seq = []   # list of (op, key): op in 'h','m','t','x'
        self.mark = mark

    def protected(self):
        return {key(self.layer, e) for e in self.ids}

    def demand(self):
        """(key, was_hit) lookups in order."""
        return [(k, op == "h") for op, k in self.seq if op in "hm"]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    marks, calls, flushes = [], [], []
    cur_mark = -1
    cur = None
    loose = []   # events outside a call (should be none except x/F)
    with op(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if not f:
                continue
            t = f[0]
            if t in ("h", "m", "t", "x"):
                k = key(int(f[1]), int(f[2]))
                if cur is None:
                    loose.append((t, k))
                else:
                    cur.seq.append((t, k))
            elif t == "A":
                cur = Call(f, cur_mark)
                calls.append(cur)
            elif t == "M":
                marks.append(Mark(f, len(marks)))
                cur_mark = len(marks) - 1
                cur = None
            elif t == "F":
                flushes.append(len(calls))
                cur = None
            else:
                raise ValueError("bad line: " + line)
    return marks, calls, flushes, loose
