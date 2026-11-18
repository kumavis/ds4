"""Chronology gates for a disk-restored D19 long-session pair."""
from copy import deepcopy
from pathlib import Path
import unittest

from q4_long import ORDER, chronology, tag


class LongPairTests(unittest.TestCase):
    def setUp(self):
        path = Path("/example/a")
        initial = [{"turn": 25, "reply_sha": "abc"}]
        self.a = ({"max_context": 66002, "ts_end": "2026-11-18T13:00:00-10:00"},
                  initial, {"turns": initial}, path)
        self.b = ({"ts": "2026-11-18T14:00:00-10:00"},
                  [{"turn": 26, "norm": True, "kv_hit_tokens": 66002, "reused_tokens": 66002,
                    "prompt_tokens": 68100, "reply_tokens": 256},
                   {"turn": 27, "prompt_tokens": 101340, "reply_tokens": 256}],
                  {"continued_from": str(path / "transcript-grow.json"),
                   "turns": deepcopy(initial) + [{"turn": 26}, {"turn": 27}]}, Path("/example/b"))

    def test_full_prefix_and_actual_context(self):
        chronology(self.a, self.b)
        self.assertEqual(len({tag(*arm) for arm in ORDER}), 4)
        for arm in ("lru", "s3fifo"):
            self.assertLess(ORDER.index((arm, "a")), ORDER.index((arm, "b")))

    def test_nominal_target_does_not_prove_actual(self):
        self.b[1][-1]["prompt_tokens"] = 98816
        with self.assertRaisesRegex(ValueError, "100000"):
            chronology(self.a,self.b)

    def test_wrong_prefix_or_earlier_restore_fails(self):
        for change in ("hit", "turn", "transcript", "time"):
            with self.subTest(change=change):
                b=deepcopy(self.b)
                if change=="hit": b[1][0]["kv_hit_tokens"]-=1
                if change=="turn": b[1][0]["turn"]+=1
                if change=="transcript": b[2]["turns"][0]["reply_sha"]="different"
                if change=="time": b[0]["ts"]="2026-11-18T12:00:00-10:00"
                with self.assertRaises(ValueError): chronology(self.a,b)


if __name__=="__main__":
    unittest.main()
