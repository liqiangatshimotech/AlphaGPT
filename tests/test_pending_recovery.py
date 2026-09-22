"""Offline regression tests for pending exits and strategy resumption.

No runner/trader constructors run: state lives in a TemporaryDirectory and every
RPC, quote, inference and order-submission boundary is a fake.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from strategy_manager.portfolio import PortfolioManager
from strategy_manager.runner import StrategyRunner


TOKEN = "pending-token"
OTHER_TOKEN = "other-token"
DECIMALS = 6
PRE_RAW_BALANCE = 100_000_000


class PendingRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state_path = Path(self.directory.name) / "portfolio.json"
        self.pending_path = Path(self.directory.name) / "pending.json"
        self.recovery_environment = patch.dict(
            os.environ,
            {"LEGACY_PENDING_RECOVERY_PATH": str(Path(self.directory.name) / "legacy-recovery.json")},
        )
        self.recovery_environment.start()
        self.addCleanup(self.recovery_environment.stop)
        self.runner = self._make_runner()
        self.runner.portfolio.add_position(TOKEN, "TOKEN", 1.0, 100.0, 100.0)
        self.decimals_patch = patch(
            "strategy_manager.runner.get_mint_decimals",
            new=AsyncMock(return_value=DECIMALS),
        )
        self.decimals_patch.start()
        self.addCleanup(self.decimals_patch.stop)

    def _make_runner(self):
        runner = StrategyRunner.__new__(StrategyRunner)
        runner.portfolio = PortfolioManager(str(self.state_path))
        runner.pending_orders_path = str(self.pending_path)
        runner.order_history_path = str(Path(self.directory.name) / "order-history.jsonl")
        runner.pending_orders = runner._load_pending_orders()
        runner.entry_cooldowns = {}
        runner.token_map = {}
        runner.entries_paused = True
        runner.stop_signal_path = str(Path(self.directory.name) / "STOP_SIGNAL")
        rpc = SimpleNamespace(
            client=object(),
            get_expiry_evidence=AsyncMock(return_value=("pending", {})),
            get_token_balance=AsyncMock(return_value=PRE_RAW_BALANCE),
        )
        runner.trader = SimpleNamespace(rpc=rpc, sell=AsyncMock(return_value=False))
        runner._fetch_live_price_sol = AsyncMock(return_value=1.0)
        runner._run_inference = AsyncMock(return_value=-1)
        return runner

    def _pending_sell(self, **changes):
        order = {
            "side": "sell",
            "token_address": TOKEN,
            "signature": "fake-signature-not-submitted",
            "ratio": 0.5,
            "reason": "Moonbag",
            "created_at": 1.0,
            "pre_raw_balance": PRE_RAW_BALANCE,
            "amount_raw": PRE_RAW_BALANCE // 2,
            "decimals": DECIMALS,
            "last_valid_block_height": 100,
            "recent_blockhash": "fake-blockhash",
        }
        order.update(changes)
        self.runner.pending_orders[TOKEN] = order
        self.runner._save_pending_orders()
        return order

    async def test_unknown_or_processed_never_releases_or_resubmits_sell(self):
        for state, detail in (("unknown", {"error": "timeout"}), ("pending", {"status": "processed"})):
            with self.subTest(state=state, detail=detail):
                self._pending_sell()
                self.runner.trader.rpc.get_expiry_evidence.return_value = (state, detail)
                self.runner._fetch_live_price_sol.return_value = 0.9
                await self.runner._reconcile_pending_orders()
                await self.runner.monitor_positions()
                self.assertIn(TOKEN, self.runner.pending_orders)
                self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
                self.runner.trader.sell.assert_not_awaited()

    async def test_unknown_full_sell_does_not_treat_zero_balance_as_confirmation(self):
        self._pending_sell(ratio=1.0, reason="StopLoss", amount_raw=PRE_RAW_BALANCE)
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("unknown", {"error": "timeout"})
        self.runner.trader.rpc.get_token_balance.return_value = 0
        await self.runner._reconcile_pending_orders()
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.assertIn(TOKEN, self.runner.portfolio.positions)
        self.runner.trader.sell.assert_not_awaited()

    async def test_confirmed_partial_sell_uses_actual_balance_and_marks_moonbag(self):
        self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "finalized"})
        # Deliberately not the arithmetical 50% balance: chain state wins.
        self.runner.trader.rpc.get_token_balance.return_value = 49_250_000
        await self.runner._reconcile_pending_orders()
        position = self.runner.portfolio.positions[TOKEN]
        self.assertEqual(position.amount_held, 49.25)
        self.assertTrue(position.is_moonbag)
        self.assertNotIn(TOKEN, self.runner.pending_orders)
        self.runner._fetch_live_price_sol.return_value = 1.11
        await self.runner.monitor_positions()
        self.runner.trader.sell.assert_not_awaited()
        persisted = json.loads(self.state_path.read_text())
        self.assertEqual(persisted[TOKEN]["amount_held"], 49.25)
        self.assertTrue(persisted[TOKEN]["is_moonbag"])

    async def test_confirmed_sell_with_unavailable_balance_remains_pending(self):
        self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "finalized"})
        self.runner.trader.rpc.get_token_balance.return_value = None
        await self.runner._reconcile_pending_orders()
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
        self.assertFalse(self.runner.portfolio.positions[TOKEN].is_moonbag)

    async def test_confirmed_sell_with_stale_balance_does_not_finalize_early(self):
        self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "confirmed"})
        await self.runner._reconcile_pending_orders()
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
        self.assertFalse(self.runner.portfolio.positions[TOKEN].is_moonbag)

    async def test_confirmed_full_sell_closes_position_only_at_zero_balance(self):
        self._pending_sell(ratio=1.0, reason="StopLoss", amount_raw=PRE_RAW_BALANCE)
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "finalized"})
        self.runner.trader.rpc.get_token_balance.return_value = 0
        await self.runner._reconcile_pending_orders()
        self.assertNotIn(TOKEN, self.runner.pending_orders)
        self.assertNotIn(TOKEN, self.runner.portfolio.positions)

    async def test_expired_unchanged_balance_releases_without_forcing_old_take_profit(self):
        self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("expired", {"expiry_slot": 110})
        await self.runner._reconcile_pending_orders()
        self.assertNotIn(TOKEN, self.runner.pending_orders)
        self.runner._fetch_live_price_sol.return_value = 1.02
        await self.runner.monitor_positions()
        self.runner.trader.sell.assert_not_awaited()
        self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
        self.assertFalse(self.runner.portfolio.positions[TOKEN].is_moonbag)
        self.assertEqual(json.loads(self.pending_path.read_text()), {})
        self.runner.trader.rpc.get_token_balance.assert_awaited_with(
            TOKEN, commitment="finalized", min_context_slot=110
        )
        archive = json.loads(Path(self.runner.order_history_path).read_text())
        self.assertEqual(archive["outcome"], "expired")
        self.assertEqual(archive["signature"], "fake-signature-not-submitted")
        self.assertEqual(archive["evidence"]["raw_balance"], PRE_RAW_BALANCE)
        self.assertEqual(archive["evidence"]["expiry_slot"], 110)

    async def test_expired_unchanged_balance_resumes_current_stop_loss(self):
        self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("expired", {"expiry_slot": 110})
        await self.runner._reconcile_pending_orders()
        self.runner._fetch_live_price_sol.return_value = 0.9
        await self.runner.monitor_positions()
        self.runner.trader.sell.assert_awaited_once()
        args, kwargs = self.runner.trader.sell.await_args
        self.assertEqual(args[0], TOKEN)
        self.assertEqual(kwargs["percentage"], 1.0)

    async def test_expired_changed_or_unavailable_balance_stays_quarantined(self):
        for balance in (49_250_000, 0, PRE_RAW_BALANCE + 1, None):
            with self.subTest(balance=balance):
                self._pending_sell()
                self.runner.trader.rpc.get_expiry_evidence.return_value = ("expired", {"expiry_slot": 110})
                self.runner.trader.rpc.get_token_balance.return_value = balance
                await self.runner._reconcile_pending_orders()
                self.assertIn(TOKEN, self.runner.pending_orders)
                self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
                self.runner.trader.sell.assert_not_awaited()

    async def test_legacy_order_without_original_balance_is_not_released_on_expiry(self):
        order = self._pending_sell()
        order.pop("pre_raw_balance")
        self.runner._save_pending_orders()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("expired", {"expiry_slot": 110})
        await self.runner._reconcile_pending_orders()
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.assertEqual(self.runner.portfolio.positions[TOKEN].amount_held, 100.0)
        self.runner.trader.sell.assert_not_awaited()

    async def test_one_pending_exit_does_not_block_another_positions_stop_loss(self):
        self._pending_sell()
        self.runner.portfolio.add_position(OTHER_TOKEN, "OTHER", 1.0, 100.0, 100.0)
        self.runner._fetch_live_price_sol.return_value = 0.9
        await self.runner.monitor_positions()
        self.runner.trader.sell.assert_awaited_once()
        args, kwargs = self.runner.trader.sell.await_args
        self.assertEqual(args[0], OTHER_TOKEN)
        self.assertEqual(kwargs["percentage"], 1.0)
        self.assertIn(TOKEN, self.runner.pending_orders)

    async def test_recovery_rpc_exception_does_not_abort_other_positions_stop_loss(self):
        self._pending_sell()
        self.runner.portfolio.add_position(OTHER_TOKEN, "OTHER", 1.0, 100.0, 100.0)
        self.runner.trader.rpc.get_expiry_evidence.side_effect = TimeoutError("offline fake timeout")
        self.runner._fetch_live_price_sol.return_value = 0.9
        await self.runner._reconcile_pending_orders()
        await self.runner.monitor_positions()
        self.assertIn(TOKEN, self.runner.pending_orders)
        self.runner.trader.sell.assert_awaited_once()
        args, kwargs = self.runner.trader.sell.await_args
        self.assertEqual(args[0], OTHER_TOKEN)
        self.assertEqual(kwargs["percentage"], 1.0)

    async def test_restart_after_portfolio_save_before_pending_cleanup_is_idempotent(self):
        order = self._pending_sell()
        self.runner.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "finalized"})
        self.runner.trader.rpc.get_token_balance.return_value = 50_000_000
        await self.runner._reconcile_pending_orders()
        # Simulate a crash after portfolio persistence and before pending cleanup.
        self.pending_path.write_text(json.dumps({TOKEN: order}))
        restarted = self._make_runner()
        restarted.trader.rpc.get_expiry_evidence.return_value = ("confirmed", {"status": "finalized"})
        restarted.trader.rpc.get_token_balance.return_value = 50_000_000
        await restarted._reconcile_pending_orders()
        self.assertEqual(restarted.portfolio.positions[TOKEN].amount_held, 50.0)
        self.assertTrue(restarted.portfolio.positions[TOKEN].is_moonbag)
        self.assertNotIn(TOKEN, restarted.pending_orders)
        restarted.trader.sell.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
