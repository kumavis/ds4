"""Small traces exercise D13 sweep boundaries, shutdown and process-local state."""
from contextlib import redirect_stdout
import gzip
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from replay import RuntimeLRU, S3FIFO, replay


TRACE = """F
F
M P 0 256 770 768 0 768
A e 0 0 1 1 0
m 0 0
M P 256 256 770 769 1 768
M S 512 1 770 769 1 768
A s 0 0 1 3 1 2 3
m 0 1
m 0 2
m 0 3
t 0 1
t 0 2
t 0 3
x 0 3
x 0 2
x 0 1
F
F
"""


class SessionReplayTests(unittest.TestCase):
    def replay_text(self, trace, policy):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(trace)
            output = StringIO()
            with redirect_stdout(output):
                replay(path, policy)
            return output.getvalue().splitlines()[-1].split()[-6:]

    def test_adjacent_sweeps_and_fresh_process(self):
        for policy in (S3FIFO, RuntimeLRU):
            for offset in (0, 66002):
                # The restored server starts at a nonzero position with an empty expert cache.
                trace = TRACE.replace("M P 0 ", f"M P {offset} ").replace(
                    "M P 256 ", f"M P {offset + 256} ").replace("M S 512 ", f"M S {offset + 512} ")
                with self.subTest(policy=policy, offset=offset):
                    self.assertEqual(self.replay_text(trace, policy), ["4", "0", "4", "0", "3", "3"])

    def test_misordered_physical_victim(self):
        with self.assertRaisesRegex(ValueError, "victim"):
            self.replay_text(TRACE.replace("x 0 3", "x 0 2"), S3FIFO)

    def test_concatenated_processes_rejected(self):
        with self.assertRaisesRegex(ValueError, "events after final flush"):
            self.replay_text(TRACE + TRACE, S3FIFO)


if __name__ == "__main__":
    unittest.main()
