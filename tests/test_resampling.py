import unittest

import numpy as np
import pandas as pd

from model_core.resampling import aggregate_ohlcv, resample_ohlcv


def _rows(address="A"):
    times = pd.date_range("2026-01-01 00:00:00", periods=6, freq="1min")
    return pd.DataFrame(
        {
            "time": times,
            "address": address,
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [1, 2, 3, 4, 5, 6],
            "liquidity": [100, 101, 102, 103, 104, 105],
            "fdv": [1000, 1001, 1002, 1003, 1004, 1005],
        }
    )


class ResamplingTests(unittest.TestCase):
    def test_ohlcv_and_last_state_aggregation(self):
        result = resample_ohlcv(_rows(), "3min")
        self.assertEqual(len(result), 2)
        first = result.iloc[0]
        self.assertEqual(first["time"], pd.Timestamp("2026-01-01 00:00:00"))
        self.assertEqual(first["open"], 10)
        self.assertEqual(first["high"], 13)
        self.assertEqual(first["low"], 9)
        self.assertEqual(first["close"], 12.5)
        self.assertEqual(first["volume"], 6)
        self.assertEqual(first["liquidity"], 102)
        self.assertEqual(first["fdv"], 1002)
        self.assertTrue(bool(first["observed"]))
        self.assertTrue(bool(first["complete"]))
        self.assertEqual(first["source_count"], 3)
        self.assertEqual(first["valid_count"], 3)

    def test_partial_and_empty_buckets_are_marked_unobserved(self):
        frame = _rows().drop(index=[1, 4]).reset_index(drop=True)
        result = resample_ohlcv(frame, "3min")
        # Both buckets retain partial aggregates for diagnostics, but neither
        # can be used as a complete 3-minute candle.
        self.assertEqual(len(result), 2)
        self.assertFalse(result["observed"].any())
        self.assertEqual(result.loc[0, "valid_count"], 2)
        self.assertEqual(result.loc[1, "valid_count"], 2)
        self.assertEqual(result.loc[0, "volume"], 4)

        # An entirely missing middle bucket is represented explicitly when
        # include_empty=True and has no fabricated OHLC values.
        sparse = _rows().drop(index=[2, 3]).reset_index(drop=True)
        result = resample_ohlcv(sparse, "2min")
        self.assertEqual(len(result), 3)
        empty = result.iloc[1]
        self.assertEqual(empty["source_count"], 0)
        self.assertFalse(bool(empty["observed"]))
        self.assertTrue(pd.isna(empty["close"]))

    def test_explicit_source_mask_and_optional_empty_rows(self):
        frame = _rows().iloc[:3].copy()
        frame["observed"] = True
        frame.loc[1, "observed"] = False
        result = resample_ohlcv(frame, "3min")
        self.assertFalse(bool(result.loc[0, "observed"]))
        self.assertEqual(result.loc[0, "valid_count"], 2)
        compact = resample_ohlcv(frame, "3min", include_empty=False)
        self.assertEqual(len(compact), 1)
        self.assertIs(aggregate_ohlcv, resample_ohlcv)

    def test_multiple_addresses_are_independent(self):
        frame = pd.concat([_rows("A"), _rows("B")], ignore_index=True)
        frame.loc[frame["address"] == "B", "close"] += 100
        result = resample_ohlcv(frame, "5min")
        self.assertEqual(result["address"].tolist(), ["A", "A", "B", "B"])
        self.assertEqual(result.loc[result["address"] == "A", "close"].iloc[0], 14.5)
        self.assertEqual(result.loc[result["address"] == "B", "close"].iloc[0], 114.5)

    def test_bad_intervals_and_missing_ohlc_fail(self):
        with self.assertRaises(ValueError):
            resample_ohlcv(_rows(), "90sec")
        with self.assertRaises(ValueError):
            resample_ohlcv(_rows().drop(columns=["high"]), "3min")
        with self.assertRaises(ValueError):
            resample_ohlcv(_rows(), "0min")


if __name__ == "__main__":
    unittest.main()
