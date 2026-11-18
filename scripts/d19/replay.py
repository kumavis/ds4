"""Replay D19 MXFP4 traces, one file per process, with D13 on and D5 off."""
import argparse
from collections import OrderedDict, deque
import gzip

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "d12"))
from sim import S3FIFO


class ResidentCounts(dict):
    """Observe policy evictions without consulting the trace."""

    def __init__(self, on_remove):
        super().__init__()
        self.on_remove = on_remove

    def __delitem__(self, key):
        self.on_remove(key)
        super().__delitem__(key)



class Slots:
    def __init__(self, policy):
        self.policy = policy
        self.expected = deque()
        self.entries = []
        self.index = {}
        self.lent = {}
        self.free = []
        self.active = False
        self.policy_victims = self.physical_victims = 0
        policy.f = ResidentCounts(self.evict)

    def lend(self):
        if not self.active:
            self.free.extend([True] * 768)
            self.active = True

    def insert(self, key):
        self.lent[key] = self.free.pop() if self.free else False
        self.index[key] = len(self.entries)
        self.entries.append(key)

    def evict(self, key):
        self.expected.append(key)
        idx = self.index.pop(key)
        last = self.entries.pop()
        if idx < len(self.entries):
            self.entries[idx] = last
            self.index[last] = idx
        self.free.append(self.lent.pop(key))
        self.policy_victims += 1

    def reclaim(self):
        for idx in reversed(range(len(self.entries))):
            key = self.entries[idx]
            if self.lent[key]:
                self.policy.S.pop(key, None)
                self.policy.M.pop(key, None)
                dict.__delitem__(self.policy.f, key)
                self.evict(key)
                self.physical_victims += 1
                self.policy_victims -= 1
        self.free = [slot for slot in self.free if not slot]
        self.active = False


class RuntimeLRU:
    """Exact runtime recency on an archived trace, including pointer-table touches."""

    def __init__(self, cap):
        self.cap = cap
        self.S, self.M = OrderedDict(), {}
        self.f = {}

    def access(self, key, ci, t, protected, nu):
        if key in self.S:
            self.S.move_to_end(key)
            return True
        if len(self.S) >= self.cap:
            victim = next(k for k in self.S if k not in protected)
            del self.S[victim]
            del self.f[victim]
        self.S[key] = None
        self.f[key] = 0
        return False


def replay(path, policy_class=S3FIFO):
    with gzip.open(path, "rt") as stream:
        last_access = 0
        flushed = False
        for i, line in enumerate(stream, 1):
            if last_access and line.startswith("F"):
                flushed = True
            elif flushed:
                raise ValueError(f"{path}:{i}: events after final flush; replay processes separately")
            elif line.startswith(("m ", "h ", "t ")):
                last_access = i
        if not last_access or not flushed:
            raise ValueError(f"{path}: empty or incomplete trace")
    with gzip.open(path, "rt") as stream:
        lines = iter(enumerate(stream, 1))
        slots = None
        lookup = hits = checked = 0
        body = ended = False
        marks = 0
        previous_phase = None
        protected = set()
        for lineno, line in lines:
            f = line.split()
            op = f[0]
            if op == "F":
                if body:
                    ended = True
                continue
            if ended:
                raise ValueError(f"{path}:{lineno}: events after final flush; replay processes separately")
            if op == "M":
                body = True
                if slots is not None and slots.expected:
                    raise ValueError(f"{path}:{lineno}: pending victims across phase mark")
                marks += 1
                if slots is None:
                    budget = int(f[4])
                    policy = policy_class(budget)
                    slots = Slots(policy)
                    slots.lend()
                if int(f[4]) != budget or len(slots.entries) != int(f[6]):
                    raise ValueError((path, lineno, "mark budget or resident count", f, len(slots.entries)))
                # Every sweep gives the staging slots back before the next mark,
                # even when the following mark is another sweep rather than q.
                if previous_phase == "P":
                    slots.lend()
                    policy.cap = budget
                    policy.ks = max(1, policy.cap // 10)
                previous_phase = f[1]
                if f[1] == "P":
                    slots.reclaim()
                    policy.cap = budget - 768
                    policy.ks = max(1, policy.cap // 10)
                if f[1] == "q":
                    policy.cap = budget
                    policy.ks = max(1, budget // 10)
                    slots.lend()
            elif op == "A":
                if slots.expected:
                    raise ValueError(f"{path}:{lineno}: pending victim across call")
                protected = {int(f[3])*384 + int(e) for e in f[6:]}
            elif op == "t" and policy_class is RuntimeLRU:
                policy.S.move_to_end(int(f[1])*384 + int(f[2]))
            elif op in ("h", "m", "x"):
                key = int(f[1])*384 + int(f[2])
                if op == "x":
                    predicted = slots.expected.popleft() if slots.expected else None
                    if predicted != key:
                        raise ValueError((path, lineno, "victim", key, predicted, lookup, checked))
                    checked += 1
                else:
                    if slots.expected:
                        raise ValueError(f"{path}:{lineno}: expected victim before next lookup")
                    hit = policy.access(key, None, None, protected, None)
                    if hit != (op == "h"):
                        raise ValueError((path, lineno, "lookup", key, hit, lookup, checked))
                    if not hit:
                        slots.insert(key)
                    hits += hit
                    lookup += 1
            elif op != "t":
                raise ValueError(f"{path}:{lineno}: unknown opcode {op}")
            if lineno == last_access:
                slots.reclaim()
        if not ended or slots.expected:
            raise ValueError(f"{path}: incomplete trace")
        if slots.policy_victims + slots.physical_victims != checked:
            raise ValueError(f"{path}: ordered victim count mismatch")
        print("phase_marks", marks, "policy", policy_class.__name__)
        print(path, lookup, hits, lookup-hits, slots.policy_victims, slots.physical_victims, checked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lru", action="store_true", help="validate archived D19 LRU control traces")
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    for trace in args.traces:
        replay(trace, RuntimeLRU if args.lru else S3FIFO)
