"""SQLite-backed checks for live candle freshness and historical loading."""

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlalchemy

from model_core.config import ModelConfig
from model_core.data_loader import CryptoDataLoader


class DataLoaderFreshnessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_url = f"sqlite:///{Path(directory.name) / 'candles.sqlite'}"
        model_url = patch.object(ModelConfig, "DB_URL", db_url)
        model_url.start()
        self.addCleanup(model_url.stop)

        self.as_of = datetime(2026, 9, 25, 8, 0, tzinfo=timezone.utc)
        engine = sqlalchemy.create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text("CREATE TABLE tokens (address TEXT PRIMARY KEY)"))
            conn.execute(sqlalchemy.text("""
                CREATE TABLE ohlcv (
                    time TIMESTAMP NOT NULL, address TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, liquidity REAL, fdv REAL
                )
            """))
            for address in ("stale", "fresh", "boundary", "future"):
                conn.execute(sqlalchemy.text("INSERT INTO tokens (address) VALUES (:address)"),
                             {"address": address})
            self._insert(conn, "stale", range(120, 113, -1), liquidity=9_000_000)
            self._insert(conn, "fresh", (12, 11, 10), liquidity=300_000)
            self._insert(conn, "boundary", (31, 30), liquidity=400_000)
            self._insert(conn, "future", (0, -1, -2, -3), liquidity=8_000_000)
        engine.dispose()

    def _insert(self, conn, address, minutes_ago, liquidity):
        for minutes in minutes_ago:
            time = self.as_of - timedelta(minutes=minutes)
            conn.execute(sqlalchemy.text("""
                INSERT INTO ohlcv (
                    time, address, open, high, low, close, volume, liquidity, fdv
                ) VALUES (
                    :time, :address, 1, 1.1, 0.9, 1, 100, :liquidity, 2000000
                )
            """), {"time": time.replace(tzinfo=None).isoformat(sep=" "),
                   "address": address, "liquidity": liquidity})

    def test_live_filter_runs_before_top_n_and_excludes_future_and_stale_rows(self):
        loader = CryptoDataLoader()
        loader.load_data(limit_tokens=2, max_candle_age_seconds=1800,
                         as_of=datetime(2026, 9, 25, 16, 0,
                                        tzinfo=timezone(timedelta(hours=8))))

        self.assertEqual(loader.addresses, ["fresh", "boundary"])
        self.assertEqual(loader.latest_candle_times, {
            "fresh": self.as_of - timedelta(minutes=10),
            "boundary": self.as_of - timedelta(minutes=30),
        })
        self.assertEqual(tuple(loader.feat_tensor.shape[:2]), (2, 6))
        self.assertEqual(loader.raw_data_cache["liquidity"][:, -1].tolist(),
                         [300_000, 400_000])

    def test_historical_default_preserves_count_based_selection(self):
        loader = CryptoDataLoader()
        loader.load_data(limit_tokens=1)

        self.assertEqual(loader.addresses, ["stale"])
        self.assertEqual(loader.latest_candle_times["stale"],
                         self.as_of - timedelta(minutes=114))

    def test_no_fresh_tokens_clears_previous_tensors(self):
        loader = CryptoDataLoader()
        loader.load_data(limit_tokens=1)
        with self.assertRaisesRegex(ValueError, "No fresh tokens found"):
            loader.load_data(limit_tokens=2, max_candle_age_seconds=1800,
                             as_of=self.as_of + timedelta(days=1))

        self.assertEqual(loader.addresses, [])
        self.assertEqual(loader.latest_candle_times, {})
        self.assertIsNone(loader.feat_tensor)
        self.assertIsNone(loader.raw_data_cache)


if __name__ == "__main__":
    unittest.main()
