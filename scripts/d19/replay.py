"""Replay D19 plain-code, DSpark-JSON and sequence traces (MXFP4, D13 on, D5 off)."""
import argparse
from collections import deque
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


def replay(path):
    with gzip.open(path, "rt") as stream:
        last_lookup = max(i for i, line in enumerate(stream, 1) if line.startswith(("m ", "h ")))
    with gzip.open(path, "rt") as stream:
        lines = iter(enumerate(stream, 1))
        slots = None
        lookup = hits = checked = 0
        protected = set()
        for lineno, line in lines:
            f = line.split()
            op = f[0]
            if op == "M":
                if slots is None:
                    budget = int(f[4])
                    policy = S3FIFO(budget)
                    slots = Slots(policy)
                    slots.lend()
                if f[1] == "P":
                    slots.reclaim()
                    policy.cap = budget - 768
                    policy.ks = max(1, policy.cap // 10)
                if f[1] == "q":
                    policy.cap = budget
                    policy.ks = max(1, budget // 10)
                    slots.lend()
            elif op == "A":
                protected = {int(f[3])*384 + int(e) for e in f[6:]}
            elif op in ("h", "m", "x"):
                key = int(f[1])*384 + int(f[2])
                if op == "x":
                    predicted = slots.expected.popleft() if slots.expected else None
                    if predicted != key:
                        raise ValueError((path, lineno, "victim", key, predicted, lookup, checked))
                    checked += 1
                else:
                    hit = policy.access(key, None, None, protected, None)
                    if hit != (op == "h"):
                        raise ValueError((path, lineno, "lookup", key, hit, lookup, checked))
                    if not hit:
                        slots.insert(key)
                    hits += hit
                    lookup += 1
                    if lineno == last_lookup:
                        slots.reclaim()
        assert not slots.expected
        print(path, lookup, hits, lookup-hits, slots.policy_victims, slots.physical_victims, checked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    for trace in parser.parse_args().traces:
        replay(trace)
