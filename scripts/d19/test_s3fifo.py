"""CPU regression cases for the D19 runtime eviction order."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "d12"))
from sim import S3FIFO


class EvictionTests(unittest.TestCase):
    def make_policy(self, small, main, capacity=4):
        policy = S3FIFO(capacity, small=0.5)
        for queue, entries in ((policy.S, small), (policy.M, main)):
            for key, frequency in entries:
                queue[key] = None
                policy.f[key] = frequency
        return policy

    def test_promotion_uses_only_one_victim(self):
        policy = self.make_policy([(1, 1), (2, 0)], [(3, 0), (4, 0)])
        self.assertFalse(policy.access(5, None, None, set(), None))
        self.assertEqual(list(policy.M), [3, 4, 1])
        self.assertEqual(list(policy.S), [5])
        self.assertEqual(list(policy.G), [2])

    def test_protected_main_falls_back_to_small(self):
        policy = self.make_policy([(1, 0)], [(2, 0), (3, 0), (4, 0)])
        self.assertFalse(policy.access(5, None, None, {2, 3, 4}, None))
        self.assertEqual(set(policy.M), {2, 3, 4})
        self.assertEqual(list(policy.S), [5])
        self.assertEqual(list(policy.G), [1])

    def test_no_victim_when_all_protected(self):
        policy = self.make_policy([(1, 1), (2, 0)], [(3, 0), (4, 2)])
        with self.assertRaisesRegex(RuntimeError, "no evictable resident"):
            policy.access(5, None, None, {1, 2, 3, 4}, None)
        self.assertEqual(policy.f, {1: 1, 2: 0, 3: 0, 4: 2})


if __name__ == "__main__":
    unittest.main()
