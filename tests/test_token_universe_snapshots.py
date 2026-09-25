"""Offline checks for append-only, point-in-time token discovery records."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from solders.keypair import Keypair

from data_pipeline.data_manager import DataManager
from data_pipeline.providers.birdeye import BirdeyeProvider


class TokenUniverseSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.snapshot_path = Path(directory.name) / "nested" / "universe.jsonl"
        self.path_patch = patch.dict(os.environ, {
            "TOKEN_UNIVERSE_SNAPSHOT_PATH": str(self.snapshot_path),
        })
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

        self.manager = DataManager.__new__(DataManager)
        self.manager.db = SimpleNamespace(
            get_latest_ohlcv_times=AsyncMock(return_value={}),
            upsert_tokens=AsyncMock(),
            batch_insert_ohlcv=AsyncMock(return_value=0),
        )
        self.manager.birdeye = SimpleNamespace(
            get_trending_tokens=AsyncMock(),
            get_token_history=AsyncMock(return_value=[]),
            headers={},
            _interval_seconds=lambda: 60,
        )

    async def test_appends_sanitized_candidates_and_empty_selection(self):
        eligible = {
            "address": "GoodToken", "symbol": "GOOD", "name": "Good Token",
            "decimals": 6, "liquidity": 750_000, "fdv": 12_000_000,
            "api_key": "must-never-appear",
        }
        low_liquidity = {
            "address": "LowToken", "liquidity": 499_999, "fdv": 12_000_000,
        }
        malformed = {
            "address": "Bad\nToken", "liquidity": float("nan"),
            "fdv": float("inf"), "secret": "must-never-appear",
        }
        self.manager.birdeye.get_trending_tokens.side_effect = [
            [eligible, low_liquidity, malformed], [],
        ]
        self.snapshot_path.parent.mkdir(parents=True)
        self.snapshot_path.write_text("existing-line\n", encoding="utf-8")

        result = await self.manager.pipeline_sync_daily()
        empty_result = await self.manager.pipeline_sync_daily()

        self.assertEqual(result["tokens"], 1)
        self.assertIsNone(empty_result)
        self.manager.db.upsert_tokens.assert_awaited_once()
        lines = self.snapshot_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "existing-line")
        self.assertEqual(len(lines), 3)
        first, second = (json.loads(line) for line in lines[1:])
        self.assertEqual(first["source"], "birdeye_trending")
        self.assertEqual(first["fetch_status"], "success")
        self.assertEqual(second["fetch_status"], "success")
        observed_at = datetime.fromisoformat(first["observed_at"].replace("Z", "+00:00"))
        self.assertEqual(observed_at.tzinfo, timezone.utc)
        self.assertEqual(first["candidates"], [
            {"address": "GoodToken", "liquidity": 750_000.0,
             "fdv": 12_000_000.0, "selected": True},
            {"address": "LowToken", "liquidity": 499_999.0,
             "fdv": 12_000_000.0, "selected": False},
            {"address": None, "liquidity": None,
             "fdv": None, "selected": False},
        ])
        self.assertEqual(second["candidates"], [])
        self.assertNotIn("must-never-appear", "\n".join(lines))

    async def test_snapshot_write_failure_does_not_interrupt_market_sync(self):
        self.manager.birdeye.get_trending_tokens.return_value = [{
            "address": "GoodToken", "symbol": "GOOD", "name": "Good Token",
            "decimals": 6, "liquidity": 750_000, "fdv": 12_000_000,
        }]
        self.snapshot_path.mkdir(parents=True)

        result = await self.manager.pipeline_sync_daily()

        self.assertEqual(result["tokens"], 1)
        self.manager.db.upsert_tokens.assert_awaited_once()
        self.manager.birdeye.get_token_history.assert_awaited_once()

    async def test_trending_error_is_not_recorded_as_successful_empty_universe(self):
        self.manager.birdeye.last_trending_status = "error"
        self.manager.birdeye.get_trending_tokens.return_value = []

        self.assertIsNone(await self.manager.pipeline_sync_daily())

        row = json.loads(self.snapshot_path.read_text().splitlines()[0])
        self.assertEqual(row["fetch_status"], "error")
        self.assertEqual(row["candidates"], [])

    async def test_provider_marks_request_failure_separately_from_empty_success(self):
        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        provider = BirdeyeProvider.__new__(BirdeyeProvider)
        provider.headers = {}
        provider._get_json = AsyncMock(side_effect=TimeoutError("offline fake timeout"))
        with patch("data_pipeline.providers.birdeye.aiohttp.ClientSession", return_value=FakeSession()):
            self.assertEqual(await provider.get_trending_tokens(), [])
            self.assertEqual(provider.last_trending_status, "error")
            provider._get_json.side_effect = None
            provider._get_json.return_value = {"data": {"tokens": []}}
            self.assertEqual(await provider.get_trending_tokens(), [])
            self.assertEqual(provider.last_trending_status, "success")
            provider._get_json.return_value = {"data": {}}
            self.assertEqual(await provider.get_trending_tokens(), [])
            self.assertEqual(provider.last_trending_status, "error")

    async def test_quote_sample_rotates_and_is_bounded_to_five_mints(self):
        selected = [{"address": str(Keypair.from_seed(bytes([index]) * 32).pubkey())}
                    for index in range(12)]

        class FakeQuoteClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        with patch.dict(os.environ, {
            "TOKEN_QUOTE_SAMPLE_SIZE": "5", "JUPITER_API_KEY": "fixture-key",
        }), patch("data_pipeline.data_manager.JupiterQuoteOnlyClient", return_value=FakeQuoteClient()), \
                patch("data_pipeline.data_manager.collect_many", new=AsyncMock(
                    return_value=[{"status": "available"}] * 5
                )) as collect:
            await self.manager._collect_quote_snapshots(selected, "2026-09-25T08:00:00Z")
            first = collect.await_args.args[0]
            await self.manager._collect_quote_snapshots(selected, "2026-09-25T08:15:00Z")
            second = collect.await_args.args[0]

        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertEqual(len(second), 5)
        self.assertNotEqual(first, second)
        self.assertEqual(collect.await_count, 2)

    async def test_quote_collection_failure_does_not_abort_pipeline(self):
        selected = [{"address": str(Keypair.from_seed(bytes(32)).pubkey())}]

        class FakeQuoteClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        with patch.dict(os.environ, {
            "TOKEN_QUOTE_SAMPLE_SIZE": "1", "JUPITER_API_KEY": "fixture-key",
        }), patch("data_pipeline.data_manager.JupiterQuoteOnlyClient", return_value=FakeQuoteClient()), \
                patch("data_pipeline.data_manager.collect_many", new=AsyncMock(
                    side_effect=TimeoutError("offline fake timeout")
                )):
            await self.manager._collect_quote_snapshots(selected, "2026-09-25T08:00:00Z")

    async def test_bad_or_duplicate_mint_does_not_drop_other_quote_samples(self):
        valid = str(Keypair.from_seed(bytes(32)).pubkey())
        other = str(Keypair.from_seed(bytes([1]) * 32).pubkey())
        selected = [
            {"address": "invalid"}, {"address": valid}, {"address": valid},
            {"address": other},
        ]

        class FakeQuoteClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

        with patch.dict(os.environ, {
            "TOKEN_QUOTE_SAMPLE_SIZE": "4", "JUPITER_API_KEY": "fixture-key",
        }), patch("data_pipeline.data_manager.JupiterQuoteOnlyClient", return_value=FakeQuoteClient()), \
                patch("data_pipeline.data_manager.collect_many", new=AsyncMock(
                    return_value=[{"status": "available"}] * 2
                )) as collect:
            await self.manager._collect_quote_snapshots(selected, "2026-09-25T08:00:00Z")

        self.assertEqual(set(collect.await_args.args[0]), {valid, other})


if __name__ == "__main__":
    unittest.main()
