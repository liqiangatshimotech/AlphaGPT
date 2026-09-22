import tempfile
import time
import unittest
from pathlib import Path

from execution.jupiter import JupiterAggregator
from execution.rate_limit import _reserve_slot


class JupiterRateLimitTests(unittest.TestCase):
    def test_retry_after_header_is_respected_and_capped(self):
        self.assertEqual(
            JupiterAggregator._retry_after_seconds({"Retry-After": "3"}, now=100),
            3.0,
        )
        self.assertEqual(
            JupiterAggregator._retry_after_seconds({"x-ratelimit-reset": "130"}, now=100),
            30.0,
        )

    def test_shared_slot_serializes_local_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "jupiter-rate.lock")
            _reserve_slot(path, 0.03)
            started = time.monotonic()
            _reserve_slot(path, 0.03)
            self.assertGreaterEqual(time.monotonic() - started, 0.025)


if __name__ == "__main__":
    unittest.main()
