"""Offline checks for the portable monitoring entry point; no notifications."""

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import monitor_bot as monitor


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self, **kwargs):
        return self.body


class FakeSession:
    def __init__(self, response):
        self.post = Mock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.patches = [
            patch.object(monitor, "ROOT", self.root),
            patch.dict(os.environ, {"MONITOR_TIMEZONE": "Asia/Shanghai"}, clear=True),
            patch.object(monitor.aiohttp, "ClientSession", side_effect=AssertionError("unexpected HTTP")),
            patch.object(monitor.asyncpg, "connect", side_effect=AssertionError("unexpected database")),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def snapshot(self):
        return monitor.Snapshot(
            "2026-09-22T16:00:00+00:00", True, 1234, True, "public...wallet",
            2.5, 500, 3, None, [{"symbol": "TEST"}], [], [], [],
        )

    async def test_dry_run_uses_fallback_and_appends_report_once_without_sending(self):
        snapshot = self.snapshot()
        output = io.StringIO()
        with patch.object(monitor, "collect_snapshot", AsyncMock(return_value=snapshot)) as collect, \
                patch.object(monitor, "position_report", AsyncMock(return_value="POSITION REPORT")) as report, \
                patch.object(monitor, "deepseek_summary", AsyncMock(side_effect=RuntimeError("offline"))), \
                patch.object(monitor, "send_dingtalk", AsyncMock()) as send, redirect_stdout(output):
            await monitor.run_once(dry_run=True)
        collect.assert_awaited_once()
        report.assert_awaited_once_with(snapshot.local_positions)
        send.assert_not_awaited()
        self.assertIn("Runner=运行中", output.getvalue())
        self.assertEqual(output.getvalue().count("POSITION REPORT"), 1)
        self.assertIn("2026-09-23 00:00:00", output.getvalue())

    async def test_normal_iteration_sends_summary_and_report_once(self):
        with patch.object(monitor, "collect_snapshot", AsyncMock(return_value=self.snapshot())), \
                patch.object(monitor, "position_report", AsyncMock(return_value="POSITION REPORT")), \
                patch.object(monitor, "deepseek_summary", AsyncMock(return_value="AI SUMMARY")), \
                patch.object(monitor, "send_dingtalk", AsyncMock()) as send, redirect_stdout(io.StringIO()):
            await monitor.run_once()
        send.assert_awaited_once()
        self.assertEqual(send.await_args.args[0].count("POSITION REPORT"), 1)
        self.assertTrue(send.await_args.args[0].startswith("AI SUMMARY\n\nPOSITION REPORT"))

    async def test_deepseek_payload_preserves_configured_model_and_response(self):
        session = FakeSession(FakeResponse({"choices": [{"message": {"content": "  summary  "}}]}))
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "fixture-model", "DEEPSEEK_API_KEY": "fixture-key"}), \
                patch.object(monitor.aiohttp, "ClientSession", return_value=session):
            self.assertEqual(await monitor.deepseek_summary(self.snapshot()), "summary")
        _, kwargs = session.post.call_args
        self.assertEqual(kwargs["json"]["model"], "fixture-model")
        self.assertEqual(kwargs["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["json"]["max_tokens"], 500)
        self.assertNotIn("fixture-key", json.dumps(kwargs["json"]))

    async def test_dingtalk_signature_and_text_payload_are_preserved(self):
        session = FakeSession(FakeResponse({"errcode": 0}))
        with patch.dict(os.environ, {
            "DINGTALK_WEBHOOK_URL": "https://example.invalid/robot?access_token=fixture",
            "DINGTALK_SECRET": "fixture-secret",
        }), patch.object(monitor.time, "time", return_value=1700000000), \
                patch.object(monitor.aiohttp, "ClientSession", return_value=session):
            await monitor.send_dingtalk("fixture message")
        args, kwargs = session.post.call_args
        query = parse_qs(urlsplit(args[0]).query)
        self.assertEqual(query["access_token"], ["fixture"])
        self.assertEqual(query["timestamp"], ["1700000000000"])
        expected = hmac.new(b"fixture-secret", b"1700000000000\nfixture-secret", hashlib.sha256).digest()
        self.assertEqual(base64.b64decode(query["sign"][0]), expected)
        self.assertEqual(kwargs["json"], {"msgtype": "text", "text": {"content": "fixture message"}})

    async def test_rpc_error_is_not_mistaken_for_empty_success(self):
        session = FakeSession(FakeResponse({"error": {"message": "fixture failure"}}))
        with self.assertRaisesRegex(RuntimeError, "RPC getBalance failed"):
            await monitor.rpc_call(session, "getBalance", ["fixture"])

    async def test_snapshot_uses_mock_read_interfaces_and_closes_failed_db_connection(self):
        connection = SimpleNamespace(fetchval=AsyncMock(side_effect=RuntimeError("fixture query failure")), close=AsyncMock())
        keypair = SimpleNamespace(pubkey=lambda: "A" * 44)
        session = FakeSession(FakeResponse({}))
        (self.root / "STOP_SIGNAL").write_text("STOP")
        rpc = AsyncMock(return_value={"value": 2_500_000_000})
        with patch.object(monitor, "runner_process", return_value=(True, 1234)), \
                patch.object(monitor, "Keypair", SimpleNamespace(from_base58_string=lambda value: keypair)), \
                patch.object(monitor.asyncpg, "connect", AsyncMock(return_value=connection)), \
                patch.object(monitor.aiohttp, "ClientSession", return_value=session), \
                patch.object(monitor, "rpc_call", rpc):
            snapshot = await monitor.collect_snapshot()
        self.assertEqual(snapshot.wallet_balance_sol, 2.5)
        self.assertTrue(snapshot.entries_paused)
        self.assertIn("PostgreSQL unavailable: RuntimeError", snapshot.warnings)
        connection.close.assert_awaited_once()
        rpc.assert_awaited_once_with(session, "getBalance", ["A" * 44])

    async def test_cli_once_and_dry_run_exit_after_one_mock_iteration(self):
        for args, expected in ((["--once"], False), (["--dry-run"], True)):
            with self.subTest(args=args), patch("sys.argv", ["monitor_bot.py", *args]), \
                    patch.object(monitor, "run_once", AsyncMock()) as run:
                await monitor.main()
                run.assert_awaited_once_with(expected)

    async def test_cli_interval_keeps_minimum_sleep_and_continues_after_error(self):
        with patch("sys.argv", ["monitor_bot.py", "--interval", "1"]), \
                patch.object(monitor, "run_once", AsyncMock(side_effect=RuntimeError("fixture"))) as run, \
                patch.object(monitor.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)) as sleep, \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(asyncio.CancelledError):
                await monitor.main()
        run.assert_awaited_once_with(False)
        sleep.assert_awaited_once_with(30)

    def test_active_log_is_selected_and_configured_credentials_are_removed(self):
        (self.root / "logs").mkdir()
        (self.root / "logs/runner.log").write_text("Transaction Sent: " + "A" * 64)
        (self.root / "logs/runner.error.log").write_text(
            "Transaction Sent: " + "B" * 64 + "\nJupiter API Error: fixture-key\n"
        )
        with patch.dict(os.environ, {"JUPITER_API_KEY": "fixture-key"}):
            events = monitor.recent_log_events()
        self.assertEqual(monitor.signature_from_logs(events), ["B" * 64])
        self.assertNotIn("fixture-key", "\n".join(events))
        self.assertIn("<redacted>", "\n".join(events))

    def test_position_mapping_and_process_pid_are_readable_without_bytecode(self):
        position = {"symbol": "TEST", "token_address": "mint", "amount_held": 5, "initial_cost_sol": 1,
                    "entry_price": .2, "highest_price": .3, "is_moonbag": True}
        (self.root / "portfolio_state.json").write_text(json.dumps({"mint": position}))
        self.assertEqual(monitor.local_positions()[0], {
            "symbol": "TEST", "token": "mint", "amount": 5, "cost_sol": 1,
            "entry_price_sol": .2, "highest_price_sol": .3, "is_moonbag": True,
        })
        with patch.object(monitor.subprocess, "run", return_value=SimpleNamespace(
            stdout=" 1234 /fixture/python -m strategy_manager.runner\n",
        )):
            self.assertEqual(monitor.runner_process(), (True, 1234))


if __name__ == "__main__":
    unittest.main()
