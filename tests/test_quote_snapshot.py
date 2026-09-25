"""Offline checks for the read-only, append-only quote collector."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from data_pipeline.quote_snapshot import (
    JupiterQuoteOnlyClient,
    ONE_SOL_LAMPORTS,
    QuoteError,
    SOL_MINT,
    append_jsonl,
    collect_many,
    collect_mint,
)


TOKEN = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
OTHER_TOKEN = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
TOKEN_AMOUNT = 123_456_789


def quote(input_mint, output_mint, in_amount, out_amount, *, slot=123, label="Orca V2"):
    return {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "inAmount": str(in_amount),
        "outAmount": str(out_amount),
        "swapMode": "ExactIn",
        "contextSlot": slot,
        "timeTaken": 0.12,
        "routePlan": [{"swapInfo": {"label": label}}],
    }


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get_quote(self, input_mint, output_mint, amount):
        self.calls.append((input_mint, output_mint, amount))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, "2026-09-25T00:00:00+00:00", "2026-09-25T00:00:01+00:00"


class FakeResponse:
    def __init__(self, status, data):
        self.status = status
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def json(self):
        return self.data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, headers, timeout, allow_redirects):
        self.calls.append((url, params, headers, timeout, allow_redirects))
        return self.responses.pop(0)


class QuoteSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_full_position_round_trip_and_metadata(self):
        provider = FakeProvider([
            quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT, slot=111),
            quote(TOKEN, SOL_MINT, TOKEN_AMOUNT, 970_000_000, slot=112, label="Raydium"),
        ])
        row = await collect_mint(TOKEN, provider)
        self.assertEqual(provider.calls, [
            (SOL_MINT, TOKEN, ONE_SOL_LAMPORTS),
            (TOKEN, SOL_MINT, TOKEN_AMOUNT),
        ])
        self.assertEqual(row["status"], "available")
        self.assertEqual(row["round_trip_cost_bps"], 300.0)
        self.assertEqual((row["buy_context_slot"], row["sell_context_slot"]), (111, 112))
        self.assertEqual(row["buy_route_labels"], ["Orca V2"])
        self.assertEqual(row["sell_route_labels"], ["Raydium"])
        self.assertEqual(row["buy_requested_at"], "2026-09-25T00:00:00+00:00")
        self.assertEqual(row["sell_received_at"], "2026-09-25T00:00:01+00:00")
        self.assertIsNone(row["error_category"])

    async def test_bad_buy_quote_stops_before_sell_and_has_no_cost(self):
        provider = FakeProvider([quote(TOKEN, TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT)])
        row = await collect_mint(TOKEN, provider)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(row["status"], "unavailable")
        self.assertEqual(row["error_category"], "buy_invalid_quote")
        self.assertIsNone(row["round_trip_cost_bps"])

    async def test_bad_sell_size_and_http_failure_are_categorized(self):
        buy = quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT)
        provider = FakeProvider([
            buy, quote(TOKEN, SOL_MINT, TOKEN_AMOUNT - 1, 980_000_000)
        ])
        row = await collect_mint(TOKEN, provider)
        self.assertEqual(row["error_category"], "sell_invalid_quote")
        self.assertIsNone(row["round_trip_cost_bps"])

        provider = FakeProvider([buy, QuoteError("rate_limited", 429)])
        row = await collect_mint(TOKEN, provider)
        self.assertEqual(row["error_category"], "sell_rate_limited")
        self.assertEqual(row["http_status"], 429)

    async def test_token_timeout_fails_closed(self):
        class HangingProvider:
            async def get_quote(self, *_):
                await asyncio.sleep(1)

        row = await collect_mint(TOKEN, HangingProvider(), token_timeout=0.01)
        self.assertEqual(row["error_category"], "buy_token_timeout")
        self.assertIsNone(row["round_trip_cost_bps"])

    async def test_sequential_collection_appends_without_replacing(self):
        provider = FakeProvider([
            quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT),
            quote(TOKEN, SOL_MINT, TOKEN_AMOUNT, 980_000_000),
            quote(SOL_MINT, OTHER_TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT),
            quote(OTHER_TOKEN, SOL_MINT, TOKEN_AMOUNT, 960_000_000),
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.jsonl"
            append_jsonl(output, {"existing": True})
            rows = await collect_many([TOKEN, OTHER_TOKEN], output, provider)
            on_disk = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(on_disk[0], {"existing": True})
            self.assertEqual(on_disk[1:], rows)
            self.assertEqual(len(provider.calls), 4)
            self.assertEqual([row["round_trip_cost_bps"] for row in rows], [200, 400])

    async def test_only_official_https_quote_get_is_allowed(self):
        for url in (
            "http://api.jup.ag/swap/v1",
            "https://example.com/swap/v1",
            "https://api.jup.ag.evil.test/swap/v1",
            "https://api.jup.ag/swap/v1?redirect=evil",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                JupiterQuoteOnlyClient("fixture-key", base_url=url)

        session = FakeSession([FakeResponse(200, quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS, TOKEN_AMOUNT))])
        calls = []

        async def gate(interval):
            calls.append(interval)

        client = JupiterQuoteOnlyClient("fixture-key", session=session, rate_gate=gate)
        async with client:
            data, _, _ = await client.get_quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS)
        self.assertEqual(data["outAmount"], str(TOKEN_AMOUNT))
        self.assertEqual(calls, [0.25])
        self.assertEqual(len(session.calls), 1)
        url, params, headers, timeout, allow_redirects = session.calls[0]
        self.assertEqual(url, "https://api.jup.ag/swap/v1/quote")
        self.assertEqual(params["swapMode"], "ExactIn")
        self.assertEqual(headers["x-api-key"], "fixture-key")
        self.assertLessEqual(timeout.total, 6.0)
        self.assertFalse(allow_redirects)

    async def test_redirect_cannot_forward_api_key(self):
        session = FakeSession([FakeResponse(302, {})])

        async def gate(_interval):
            pass

        client = JupiterQuoteOnlyClient("fixture-key", session=session, rate_gate=gate)
        async with client:
            with self.assertRaises(QuoteError) as raised:
                await client.get_quote(SOL_MINT, TOKEN, ONE_SOL_LAMPORTS)
        self.assertEqual(raised.exception.category, "redirect_blocked")
        self.assertFalse(session.calls[0][-1])

    async def test_invalid_batch_is_rejected_before_network(self):
        provider = FakeProvider([])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quotes.jsonl"
            for mints in ([TOKEN, TOKEN], ["not-a-mint"], [TOKEN] * 6):
                with self.subTest(mints=mints), self.assertRaises(ValueError):
                    await collect_many(mints, output, provider)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
