"""Submission boundaries use fake network clients and a disposable signer."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from solders.pubkey import Pubkey

from execution.trader import SolanaTrader
from execution.rpc_handler import TransactionStatusUnknown
from tests.test_rpc_recovery import transaction, client, RecoveryRpc


class TraderJournalTests(unittest.IsolatedAsyncioTestCase):
    def make_trader(self):
        trader = SolanaTrader.__new__(SolanaTrader)
        wire = RecoveryRpc()
        wire.send_error = TimeoutError("response lost")
        trader.rpc = client(wire)
        self.tx = transaction()
        self.address = str(self.tx.message.account_keys[0])
        wire.get_token_accounts_by_owner_json_parsed = AsyncMock(return_value=SimpleNamespace(value=[
            SimpleNamespace(account=SimpleNamespace(data=SimpleNamespace(parsed={"info": {
                "tokenAmount": {"amount": "100000000", "decimals": 6}
            }})))
        ]))
        trader.TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        trader.jup = SimpleNamespace(
            get_quote=AsyncMock(return_value={"outAmount": "1000000000"}),
            get_swap_tx=AsyncMock(return_value={"swapTransaction": "unsigned-fake", "lastValidBlockHeight": 300}),
            deserialize_and_sign=lambda _: self.tx,
        )
        return trader, wire

    async def test_sell_persists_real_amount_and_expiry_before_send(self):
        trader, wire = self.make_trader()
        saved = []

        def persist(metadata):
            self.assertFalse(any(call[0] == "send" for call in wire.calls))
            saved.append(metadata)

        with patch("execution.trader.ExecutionConfig.get_wallet_address", return_value=self.address):
            with self.assertRaises(TransactionStatusUnknown):
                await trader.sell(self.address, percentage=0.5, on_submitting=persist)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["pre_raw_balance"], 100000000)
        self.assertEqual(saved[0]["amount_raw"], 50000000)
        self.assertEqual(saved[0]["last_valid_block_height"], 300)
        self.assertEqual(saved[0]["decimals"], 6)
        self.assertEqual(saved[0]["signature"], str(self.tx.signatures[0]))

    async def test_missing_expiry_never_submits(self):
        trader, wire = self.make_trader()
        trader.jup.get_swap_tx.return_value = {"swapTransaction": "missing-expiry"}
        with patch("execution.trader.ExecutionConfig.get_wallet_address", return_value=self.address):
            self.assertFalse(await trader.sell(self.address))
        self.assertFalse(any(call[0] == "send" for call in wire.calls))

    async def test_disk_error_prevents_sell_submission(self):
        trader, wire = self.make_trader()

        def fail(_):
            raise OSError("disk unavailable")

        with patch("execution.trader.ExecutionConfig.get_wallet_address", return_value=self.address):
            self.assertFalse(await trader.sell(self.address, on_submitting=fail))
        self.assertFalse(any(call[0] == "send" for call in wire.calls))


if __name__ == "__main__":
    unittest.main()
