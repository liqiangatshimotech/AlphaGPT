"""Offline checks for the full-position entry/exit quote guard."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from execution.config import ExecutionConfig
from strategy_manager.risk import RiskEngine


TOKEN = "candidate-token"
BUY_LAMPORTS = 1_000_000_000
EXPECTED_TOKENS = 123_456_789


def buy_quote(**changes):
    quote = {
        "inputMint": ExecutionConfig.SOL_MINT,
        "outputMint": TOKEN,
        "inAmount": str(BUY_LAMPORTS),
        "outAmount": str(EXPECTED_TOKENS),
        "swapMode": "ExactIn",
    }
    quote.update(changes)
    return quote


def sell_quote(out_lamports, **changes):
    quote = {
        "inputMint": TOKEN,
        "outputMint": ExecutionConfig.SOL_MINT,
        "inAmount": str(EXPECTED_TOKENS),
        "outAmount": str(out_lamports),
        "swapMode": "ExactIn",
    }
    quote.update(changes)
    return quote


class EntryRoundTripRiskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.limit_environment = patch.dict(
            os.environ, {"MAX_ENTRY_ROUND_TRIP_COST_BPS": "300"}
        )
        self.limit_environment.start()
        self.addCleanup(self.limit_environment.stop)
        self.risk = RiskEngine()
        self.provider = SimpleNamespace(
            get_quote=AsyncMock(return_value=sell_quote(975_000_000))
        )

    async def check(self, buy=None):
        return await self.risk.check_entry_round_trip(
            TOKEN, buy if buy is not None else buy_quote(),
            input_lamports=BUY_LAMPORTS, quote_provider=self.provider,
        )

    async def test_checks_real_expected_position_size_and_accepts_2_5_percent_cost(self):
        self.assertTrue(await self.check())
        self.provider.get_quote.assert_awaited_once_with(
            input_mint=TOKEN,
            output_mint=ExecutionConfig.SOL_MINT,
            amount_integer=EXPECTED_TOKENS,
        )

    async def test_rejects_loss_beyond_configured_cap(self):
        self.provider.get_quote.return_value = sell_quote(940_000_000)
        self.assertFalse(await self.check())

    async def test_accepts_exact_boundary_but_rejects_one_lamport_beyond(self):
        self.provider.get_quote.return_value = sell_quote(970_000_000)
        self.assertTrue(await self.check())
        self.provider.get_quote.return_value = sell_quote(969_999_999)
        self.assertFalse(await self.check())

    async def test_environment_limit_changes_rejection_threshold(self):
        with patch.dict(os.environ, {"MAX_ENTRY_ROUND_TRIP_COST_BPS": "200"}):
            self.assertFalse(await self.check())

    async def test_bad_buy_quote_fails_without_sell_request(self):
        for bad in (
            buy_quote(inputMint=TOKEN),
            buy_quote(inAmount="500000000"),
            buy_quote(outAmount="0"),
            buy_quote(outAmount="1.23"),
            buy_quote(outAmount=str(2**64)),
            buy_quote(swapMode="ExactOut"),
            {key: value for key, value in buy_quote().items() if key != "swapMode"},
            {},
        ):
            with self.subTest(quote=bad):
                self.assertFalse(await self.check(bad))
        self.provider.get_quote.assert_not_awaited()

    async def test_bad_or_missing_sell_quote_fails_closed(self):
        for bad in (
            None,
            {},
            sell_quote(975_000_000, inputMint=ExecutionConfig.SOL_MINT),
            sell_quote(975_000_000, inAmount="1000000"),
            {key: value for key, value in sell_quote(975_000_000).items() if key != "swapMode"},
            sell_quote(0),
        ):
            with self.subTest(quote=bad):
                self.provider.get_quote.return_value = bad
                self.assertFalse(await self.check())

        self.provider.get_quote.side_effect = TimeoutError("quote timeout")
        self.assertFalse(await self.check())

    async def test_invalid_cap_fails_closed_without_sell_request(self):
        with patch.dict(os.environ, {"MAX_ENTRY_ROUND_TRIP_COST_BPS": "invalid"}):
            self.assertFalse(await self.check())
        self.provider.get_quote.assert_not_awaited()

    async def test_liquidity_gate_does_not_make_dummy_sell_request(self):
        self.assertTrue(await self.risk.check_safety(TOKEN, 500000))
        for value in (None, float("nan"), float("inf"), 499999):
            with self.subTest(value=value):
                self.assertFalse(await self.risk.check_safety(TOKEN, value))
        self.provider.get_quote.assert_not_awaited()
