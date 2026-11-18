"""CPU-only checks for D22's attended session pressure helper."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import driver


class FakeChunk:
    """Represents the requested size without allocating resident test pages."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __setitem__(self, key, value):
        pass


class PressureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rec = SimpleNamespace(run_dir=temp.name, stop_flag=False)
        allocator = patch.object(driver, "bytearray", FakeChunk, create=True)
        allocator.start()
        self.addCleanup(allocator.stop)

    def check_ramp_abort(self, cause):
        hold = driver.PressureHold(self.rec, 12688)
        with patch.object(driver, "mem_available_mb", return_value=14000) as sample:
            ramp = threading.Thread(target=hold.ramp, args=(512, 12950))
            ramp.start()
            while not hold.chunks:
                self.assertTrue(ramp.is_alive())
                threading.Event().wait(0.005)

            if cause == "flag":
                (self.root / "RELEASE_PRESSURE").touch()
            elif isinstance(cause, BaseException):
                sample.side_effect = cause
            else:
                sample.return_value = cause

            ramp.join(2)
            hold.watcher.join(1)
            self.assertFalse(ramp.is_alive())
            self.assertFalse(hold.watcher.is_alive())
            self.assertTrue(self.rec.stop_flag)
            self.assertEqual(hold.chunks, [])

        self.assertIn("released 256 MiB", (self.root / "hold.log").read_text())

    def test_hard_floor_during_ramp(self):
        self.check_ramp_abort(12687)

    def test_missing_telemetry_during_ramp(self):
        self.check_ramp_abort(None)

    def test_failed_read_during_ramp(self):
        self.check_ramp_abort(OSError("mock read failed"))

    def test_operator_release_during_ramp(self):
        self.check_ramp_abort("flag")

    def test_no_additional_pressure_near_target(self):
        hold = driver.PressureHold(self.rec, 12688)
        with patch.object(driver, "mem_available_mb", return_value=12966):
            self.assertEqual(hold.ramp(1536, 12950), "held")
            self.assertEqual(hold.chunks, [])
            hold.release("step")
            hold.watcher.join(1)
            self.assertFalse(self.rec.stop_flag)

    def test_normal_release_of_small_chunk(self):
        hold = driver.PressureHold(self.rec, 12688)
        with patch.object(driver, "mem_available_mb", return_value=14000):
            self.assertEqual(hold.ramp(1, 12950), "held")
            self.assertEqual([len(chunk) for chunk in hold.chunks], [1 << 20])
            hold.release("step")
            hold.watcher.join(1)
        self.assertFalse(self.rec.stop_flag)
        self.assertFalse(hold.watcher.is_alive())
        self.assertEqual(hold.chunks, [])
        self.assertIn("released 1 MiB (step)", (self.root / "hold.log").read_text())

    def test_held_bytes_released_between_turns(self):
        hold = driver.PressureHold(self.rec, 12688)
        with patch.object(driver, "mem_available_mb", return_value=14000) as sample:
            self.assertEqual(hold.ramp(1, 12950), "held")
            sample.return_value = None
            hold.watcher.join(1)
            self.assertFalse(hold.watcher.is_alive())
            self.assertEqual(hold.chunks, [])
            self.assertTrue(self.rec.stop_flag)

    def check_early_exit(self, grow_result):
        args = SimpleNamespace(
            payload="unused",
            run_dir=str(self.root),
            port=1,
            budget_min=45,
            ctx=32768,
            step=["hold:max_mb=1", "grow:target=16000", "release"],
        )

        def verify_release_before_stop(run_dir, port):
            log = (self.root / "hold.log").read_text()
            self.assertIn("released 1 MiB (scenario finished)", log)
            return {"result": "stopped"}

        with (
            patch.object(driver, "load_payload", return_value={}),
            patch.object(driver, "wait_ready", return_value=0),
            patch.object(driver, "mem_available_mb", return_value=14000),
            patch.object(driver, "run_grow", side_effect=[grow_result]),
            patch.object(driver, "clean_stop", side_effect=verify_release_before_stop) as stop,
        ):
            result = driver.cmd_run(args)
            stop.assert_called_once_with(str(self.root), 1)
        return result, json.loads((self.root / "driver.json").read_text())

    def test_early_mem_guard_releases_before_shutdown(self):
        _, status = self.check_early_exit("mem-guard")
        self.assertEqual(status["steps"][-1]["end"], "mem-guard")

    def test_exception_releases_before_shutdown(self):
        result, status = self.check_early_exit(RuntimeError("mocked error"))
        self.assertEqual(result, 1)
        self.assertEqual(status["stop"]["result"], "stopped")


if __name__ == "__main__":
    unittest.main()
