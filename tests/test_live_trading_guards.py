"""Offline checks for entry and exit safeguards in the live runner."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
import aiohttp

from execution.rpc_handler import TransactionPreflightRejected
from strategy_manager.config import StrategyConfig
from strategy_manager.portfolio import PortfolioManager
from strategy_manager.runner import StrategyRunner


TOKEN = "guard-token"
RAW_BALANCE = 100_000_000


class LiveTradingGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.runner = StrategyRunner.__new__(StrategyRunner)
        self.runner.portfolio = PortfolioManager(str(root / "portfolio.json"))
        self.runner.pending_orders_path = str(root / "pending.json")
        self.runner.order_history_path = str(root / "history.jsonl")
        self.runner.stop_signal_path = str(root / "STOP_SIGNAL")
        self.runner.pending_orders = {}
        self.runner.entry_cooldowns = {}
        self.runner.entries_paused = False
        self.runner.token_map = {}
        self.runner.formula = [0]
        rpc = SimpleNamespace(
            client=object(),
            get_token_balance=AsyncMock(return_value=RAW_BALANCE),
            get_signature_status=AsyncMock(return_value=("pending", None)),
        )
        jup = SimpleNamespace(
            get_quote=AsyncMock(return_value={"outAmount": "900000000"}),
            get_program_id_label=AsyncMock(return_value="GoonFi"),
        )
        self.runner.trader = SimpleNamespace(rpc=rpc, jup=jup, config=SimpleNamespace(SOL_MINT="SOL"))
        self.runner._run_inference = AsyncMock(return_value=-1)

    def _position(self):
        self.runner.portfolio.add_position(TOKEN, "GUARD", 1.0, 100.0, 100.0)

    async def test_all_saturated_scores_block_entries_before_risk_or_orders(self):
        self.runner.loader = SimpleNamespace(
            feat_tensor=torch.zeros(3, 1, 1),
            raw_data_cache={"liquidity": torch.full((3, 1), 1_000_000.0)},
        )
        self.runner.vm = SimpleNamespace(execute=Mock(return_value=torch.full((3, 1), 1_000_000.0)))
        self.runner.risk = SimpleNamespace(check_safety=AsyncMock())
        self.runner.trader.buy = AsyncMock()

        await self.runner.scan_for_entries()

        self.runner.risk.check_safety.assert_not_awaited()
        self.runner.trader.buy.assert_not_awaited()

    async def test_varied_but_all_buy_scores_are_blocked(self):
        self.runner.loader = SimpleNamespace(
            feat_tensor=torch.zeros(3, 1, 1),
            raw_data_cache={"liquidity": torch.full((3, 1), 1_000_000.0)},
        )
        self.runner.vm = SimpleNamespace(
            execute=Mock(return_value=torch.tensor([[2.0], [2.5], [3.0]]))
        )
        self.runner.risk = SimpleNamespace(check_safety=AsyncMock())
        self.runner.trader.buy = AsyncMock()

        await self.runner.scan_for_entries()

        self.runner.risk.check_safety.assert_not_awaited()
        self.runner.trader.buy.assert_not_awaited()

    async def test_full_exit_quote_uses_exact_raw_balance(self):
        with patch("strategy_manager.runner.get_mint_decimals", new=AsyncMock(return_value=6)):
            price = await self.runner._fetch_live_price_sol(TOKEN, raw_balance=RAW_BALANCE)

        self.assertAlmostEqual(price, 0.009)
        self.runner.trader.jup.get_quote.assert_awaited_with(
            input_mint=TOKEN, output_mint="SOL", amount_integer=RAW_BALANCE
        )

    async def test_certificate_mismatch_is_not_retried_by_price_helper(self):
        class CertificateMismatch(aiohttp.ClientSSLError):
            def __init__(self):
                Exception.__init__(self, "certificate mismatch")

        self.runner.trader.jup.get_quote.side_effect = CertificateMismatch()
        with patch("strategy_manager.runner.get_mint_decimals", new=AsyncMock(return_value=6)):
            with self.assertRaises(CertificateMismatch):
                await self.runner._fetch_live_price_sol(TOKEN, raw_balance=RAW_BALANCE)
        self.assertEqual(self.runner.trader.jup.get_quote.await_count, 1)

    async def test_full_exit_loss_triggers_stop_even_when_spot_does_not(self):
        self._position()
        self.runner._fetch_live_price_sol = AsyncMock(
            side_effect=lambda _token, **kwargs: 0.94 if "raw_balance" in kwargs else 1.01
        )
        self.runner._execute_sell = AsyncMock(return_value=True)

        await self.runner.monitor_positions()

        self.runner._execute_sell.assert_awaited_once_with(TOKEN, 1.0, "StopLoss")
        self.assertEqual(self.runner._fetch_live_price_sol.await_count, 1)

    async def test_missing_chain_balance_uses_persisted_size_for_stop_only(self):
        self._position()
        self.runner.trader.rpc.get_token_balance.return_value = None
        self.runner._fetch_live_price_sol = AsyncMock(return_value=0.9)
        self.runner._execute_sell = AsyncMock(return_value=False)

        with patch("strategy_manager.runner.get_mint_decimals", new=AsyncMock(return_value=6)):
            await self.runner.monitor_positions()

        self.runner._fetch_live_price_sol.assert_awaited_once_with(
            TOKEN, raw_balance=RAW_BALANCE, decimals=6
        )
        self.runner._execute_sell.assert_awaited_once_with(
            TOKEN, 1.0, "StopLoss", expected_raw_balance=RAW_BALANCE
        )
        self.assertNotIn(TOKEN, self.runner.entry_cooldowns)

    async def test_balance_timeout_still_checks_stop_with_local_size(self):
        self._position()
        never = asyncio.Event()

        async def stalled(*_, **__):
            await never.wait()

        self.runner.trader.rpc.get_token_balance.side_effect = stalled
        self.runner._fetch_live_price_sol = AsyncMock(return_value=0.9)
        self.runner._execute_sell = AsyncMock(return_value=False)
        with (
            patch.object(StrategyConfig, "POSITION_BALANCE_TIMEOUT_SECONDS", 0.01),
            patch("strategy_manager.runner.get_mint_decimals", new=AsyncMock(return_value=6)),
        ):
            await self.runner.monitor_positions()

        self.runner._execute_sell.assert_awaited_once_with(
            TOKEN, 1.0, "StopLoss", expected_raw_balance=RAW_BALANCE
        )

    async def test_new_position_trails_full_exit_quotes(self):
        self._position()
        pos = self.runner.portfolio.positions[TOKEN]
        self.assertEqual(pos.highest_price_basis, "full_exit")
        self.assertFalse(pos.highest_price_initialized)
        self.runner._fetch_live_price_sol = AsyncMock(
            side_effect=lambda _token, **kwargs: 0.97 if "raw_balance" in kwargs else 1.2
        )
        self.runner._execute_sell = AsyncMock(return_value=True)

        await self.runner.monitor_positions()

        self.assertEqual(pos.highest_price, 0.97)
        self.assertTrue(pos.highest_price_initialized)
        self.assertEqual(self.runner._fetch_live_price_sol.await_count, 1)
        self.runner._execute_sell.assert_not_awaited()

        self.runner._fetch_live_price_sol.return_value = 1.07
        self.runner._fetch_live_price_sol.side_effect = None
        await self.runner.monitor_positions()
        self.assertEqual(pos.highest_price, 1.07)
        self.runner._fetch_live_price_sol.return_value = 1.03
        await self.runner.monitor_positions()
        self.runner._execute_sell.assert_awaited_once_with(TOKEN, 1.0, "TrailingStop")

    async def test_legacy_position_defaults_to_spot_high_water_basis(self):
        self._position()
        state = json.loads(Path(self.runner.portfolio.state_file).read_text())
        state[TOKEN].pop("highest_price_basis")
        state[TOKEN].pop("highest_price_initialized")
        Path(self.runner.portfolio.state_file).write_text(json.dumps(state))

        restored = PortfolioManager(self.runner.portfolio.state_file)

        self.assertEqual(restored.positions[TOKEN].highest_price_basis, "spot")
        self.assertTrue(restored.positions[TOKEN].highest_price_initialized)

    async def test_zero_balance_is_quarantined_without_deleting_position(self):
        self._position()
        self.runner.trader.rpc.get_token_balance.side_effect = [0, 0]
        self.runner._fetch_live_price_sol = AsyncMock()

        await self.runner.monitor_positions()

        self.assertIn(TOKEN, self.runner.portfolio.positions)
        self.assertIn(TOKEN, self.runner.zero_balance_quarantine)
        self.runner._fetch_live_price_sol.assert_not_awaited()

    async def test_zero_balance_quarantine_alert_repeats_hourly(self):
        self._position()
        self.runner.trader.rpc.get_token_balance.return_value = 0
        with patch("strategy_manager.runner.logger.error") as alert:
            with patch("strategy_manager.runner.time.time", return_value=1000.0):
                await self.runner.monitor_positions()
                await self.runner.monitor_positions()
            with patch("strategy_manager.runner.time.time", return_value=4601.0):
                await self.runner.monitor_positions()
        self.assertEqual(alert.call_count, 2)

    async def test_zero_balance_with_unavailable_finalized_read_keeps_position(self):
        self._position()
        self.runner.trader.rpc.get_token_balance.side_effect = [0, None]
        self.runner._fetch_live_price_sol = AsyncMock()

        await self.runner.monitor_positions()

        self.assertIn(TOKEN, self.runner.portfolio.positions)
        self.runner._fetch_live_price_sol.assert_not_awaited()

    async def test_error_in_one_position_does_not_skip_later_stop_loss(self):
        self._position()
        self.runner.portfolio.add_position("second-token", "SECOND", 1.0, 100.0, 100.0)
        self.runner._fetch_live_price_sol = AsyncMock(return_value=0.9)
        self.runner._execute_sell = AsyncMock(side_effect=[RuntimeError("quote TLS error"), True])

        await self.runner.monitor_positions()

        self.assertEqual(self.runner._execute_sell.await_count, 2)
        self.assertEqual(self.runner._execute_sell.await_args.args[0], "second-token")

    async def test_hung_quote_is_bounded_and_later_stop_loss_runs(self):
        self._position()
        self.runner.portfolio.add_position("second-token", "SECOND", 1.0, 100.0, 100.0)
        never = asyncio.Event()

        async def price(token, **kwargs):
            if token == TOKEN:
                await never.wait()
            return 0.9

        self.runner._fetch_live_price_sol = AsyncMock(side_effect=price)
        self.runner._execute_sell = AsyncMock(return_value=True)
        with patch.object(StrategyConfig, "POSITION_QUOTE_TIMEOUT_SECONDS", 0.01):
            await self.runner.monitor_positions()

        self.runner._execute_sell.assert_awaited_once_with("second-token", 1.0, "StopLoss")

    async def test_slow_pipeline_sync_does_not_block_position_monitoring(self):
        Path(self.runner.stop_signal_path).write_text("STOP\n")
        sync_started = asyncio.Event()
        monitor_seen = asyncio.Event()
        never = asyncio.Event()

        async def sync():
            sync_started.set()
            await never.wait()

        async def monitor():
            monitor_seen.set()

        self.runner.last_scan_time = 0
        self.runner._pipeline_task = None
        self.runner.data_mgr = SimpleNamespace(pipeline_sync_daily=AsyncMock(side_effect=sync))
        self.runner.loader = SimpleNamespace(load_data=Mock())
        self.runner._build_token_mapping = AsyncMock()
        self.runner._reconcile_pending_orders = AsyncMock()
        self.runner.monitor_positions = AsyncMock(side_effect=monitor)

        loop_task = asyncio.create_task(self.runner.run_loop())
        try:
            await asyncio.wait_for(sync_started.wait(), timeout=1)
            await asyncio.wait_for(monitor_seen.wait(), timeout=1)
        finally:
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
            self.runner._pipeline_task.cancel()
            await asyncio.gather(self.runner._pipeline_task, return_exceptions=True)

    async def test_hung_entry_scan_is_cancelled_after_bounded_time(self):
        cancelled = asyncio.Event()
        never = asyncio.Event()

        async def scan():
            try:
                await never.wait()
            finally:
                cancelled.set()

        self.runner.last_scan_time = time.time()
        self.runner._pipeline_task = None
        self.runner.loader = SimpleNamespace(load_data=Mock())
        self.runner._build_token_mapping = AsyncMock()
        self.runner._reconcile_pending_orders = AsyncMock()
        self.runner.monitor_positions = AsyncMock()
        self.runner.scan_for_entries = AsyncMock(side_effect=scan)
        with patch.object(StrategyConfig, "ENTRY_SCAN_TIMEOUT_SECONDS", 0.01):
            loop_task = asyncio.create_task(self.runner.run_loop())
            try:
                await asyncio.wait_for(cancelled.wait(), timeout=1)
            finally:
                loop_task.cancel()
                await asyncio.gather(loop_task, return_exceptions=True)
        self.runner.monitor_positions.assert_awaited()

    async def test_confirmed_stop_loss_cooldown_restores_from_journal(self):
        now = time.time()
        Path(self.runner.order_history_path).write_text("\n".join([
            json.dumps({
                "side": "sell", "reason": "StopLoss", "outcome": "confirmed",
                "token_address": TOKEN, "resolved_at": now - 3600,
            }),
            json.dumps({
                "side": "sell", "reason": "StopLoss", "outcome": "expired",
                "token_address": "expired-token", "resolved_at": now,
            }),
        ]))

        restored = self.runner._load_entry_cooldowns()

        self.assertGreater(restored[TOKEN], now + 22 * 3600)
        self.assertNotIn("expired-token", restored)

    def _journaled_sell(self, signature="rejected-signature"):
        self.runner._record_pending_sell(
            TOKEN, signature, 1.0, "StopLoss",
            {"pre_raw_balance": RAW_BALANCE, "amount_raw": RAW_BALANCE, "decimals": 6},
        )

    async def test_proven_preflight_rejection_releases_matching_unchanged_order(self):
        self._journaled_sell()
        error = TransactionPreflightRejected(
            "rejected-signature", "simulation failed", simulation_error="0x24",
            single_attempt_proven=True,
            logs=["Program goonuddtQRrWqqn5nFyczVKaie28f3kDkHWkHtURSLE failed: custom program error: 0x24"],
        )

        released = await self.runner._release_preflight_rejection(TOKEN, error)

        self.assertTrue(released)
        self.assertNotIn(TOKEN, self.runner.pending_orders)
        archive = json.loads(Path(self.runner.order_history_path).read_text())
        self.assertEqual(archive["outcome"], "preflight_rejected")
        self.assertEqual(archive["evidence"]["raw_balance"], RAW_BALANCE)

    async def test_unproven_preflight_rejection_stays_pending(self):
        self._journaled_sell()
        error = TransactionPreflightRejected("rejected-signature", "simulation failed")

        released = await self.runner._release_preflight_rejection(TOKEN, error)

        self.assertFalse(released)
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.runner.trader.rpc.get_signature_status.assert_not_awaited()

    async def test_rejected_sell_requotes_once_excluding_failed_route(self):
        self._position()
        error = TransactionPreflightRejected(
            "rejected-signature", "simulation failed", simulation_error="0x24",
            single_attempt_proven=True,
            logs=["Program goonuddtQRrWqqn5nFyczVKaie28f3kDkHWkHtURSLE failed: custom program error: 0x24"],
        )
        error.route_labels = ["GoonFi"]
        calls = []

        async def sell(_token, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                kwargs["on_submitting"]({
                    "signature": "rejected-signature", "pre_raw_balance": RAW_BALANCE,
                    "amount_raw": RAW_BALANCE, "decimals": 6,
                })
                raise error
            return False

        self.runner.trader.sell = AsyncMock(side_effect=sell)
        self.runner.trader.rpc.get_expiry_evidence = AsyncMock(return_value=("pending", {}))

        result = await self.runner._execute_sell(TOKEN, 1.0, "StopLoss")

        self.assertFalse(result)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["exclude_dexes"])
        self.assertEqual(calls[1]["exclude_dexes"], ["GoonFi"])
        self.assertNotIn(TOKEN, self.runner.pending_orders)
        self.runner.trader.sell = AsyncMock(return_value=False)
        await self.runner._execute_sell(TOKEN, 1.0, "StopLoss")
        self.assertEqual(
            self.runner.trader.sell.await_args.kwargs["exclude_dexes"], ["GoonFi"]
        )

    async def test_unmapped_preflight_failure_does_not_retry_on_other_dex(self):
        self._position()
        error = TransactionPreflightRejected(
            "rejected-signature", "insufficient funds", simulation_error="funds",
            single_attempt_proven=True,
        )
        error.route_labels = ["GoonFi"]

        async def sell(_token, **kwargs):
            kwargs["on_submitting"]({
                "signature": "rejected-signature", "pre_raw_balance": RAW_BALANCE,
                "amount_raw": RAW_BALANCE, "decimals": 6,
            })
            raise error

        self.runner.trader.sell = AsyncMock(side_effect=sell)

        result = await self.runner._execute_sell(TOKEN, 1.0, "StopLoss")

        self.assertFalse(result)
        self.assertEqual(self.runner.trader.sell.await_count, 1)
        self.runner.trader.jup.get_program_id_label.assert_not_awaited()
        self.assertNotIn(TOKEN, self.runner.pending_orders)


if __name__ == "__main__":
    unittest.main()
