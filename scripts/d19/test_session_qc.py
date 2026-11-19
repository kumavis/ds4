"""Negative controls for raw D19 per-turn evidence."""
from hashlib import sha256
from io import BytesIO
import unittest

import session_qc as qc


class PooledDecodeTests(unittest.TestCase):
    def test_exclude_first_token_of_every_reply(self):
        only = {"reply_tokens": 1, "raw_decode_s": 0.25}
        self.assertIsNone(qc.pooled_decode([only])["decode_tps"])
        self.assertIsNone(qc.pooled_decode([dict(only, raw_decode_s=0)])["decode_tps"])
        metrics = qc.pooled_decode([only, {"reply_tokens": 4, "raw_decode_s": 1.75}])
        self.assertEqual(metrics["post_first_tokens"], 3)
        self.assertEqual(metrics["decode_tps"], 1.5)
        self.assertEqual(metrics["all_reply_tokens_per_decode_s"], 2.5)


class RawTurnTests(unittest.TestCase):
    def setUp(self):
        self.log = b"D8_STEP pos=10 tok=23\nD8_STEP pos=11 tok=45\n"
        text_hash = sha256(b"test").hexdigest()[:16]
        self.record = dict(turn=0, prompt_tokens=10, reply_tokens=2, context_after=12, http_status=200, error=None,
                           finish="length", d8_steps=2, reply_sha=text_hash, reply_tok_sha=qc.digest(["23", "45"]))
        self.raw = dict(turn=0, http_status=200, error=None, finish="length", reply_sha=text_hash,
                        usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12), log_off=[0, len(self.log)])
        self.transcript = dict(turn=0, prompt_tokens=10, completion_tokens=2, reply="test", reply_sha=text_hash)

    def check(self, previous=0):
        return qc.validate_turn(self.record, self.raw, self.transcript, BytesIO(self.log), previous, len(self.log), False)

    def test_complete_and_contiguous(self):
        self.assertEqual(self.check()[1], [(10, 23), (11, 45)])
        self.log = self.log.replace(b"pos=11", b"pos=13")
        with self.assertRaisesRegex(ValueError, "positions"):
            self.check()

    def test_empty_and_overlapping_log_slices_fail(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.check(previous=1)
        self.raw["log_off"][1] = len(self.log) + 1
        with self.assertRaisesRegex(ValueError, "offsets"):
            self.check()
        self.log = b"no steps"
        self.raw["log_off"] = [0, len(self.log)]
        self.record.update(d8_steps=0, reply_tok_sha=None)
        with self.assertRaisesRegex(ValueError, "positions"):
            self.check()

    def test_text_and_usage_must_match_raw(self):
        self.transcript["reply"] = "changed"
        with self.assertRaisesRegex(ValueError, "visible reply"):
            self.check()
        self.raw["usage"]["completion_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "usage"):
            self.check()


if __name__ == "__main__":
    unittest.main()
