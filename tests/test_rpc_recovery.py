import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import Message
from solders.transaction import VersionedTransaction

from execution.rpc_handler import QuickNodeClient, TransactionStatusUnknown


def transaction():
    payer = Keypair.from_seed(bytes(range(32)))
    return VersionedTransaction(
        Message.new_with_blockhash([], payer.pubkey(), Hash.default()), [payer]
    )


def response(value):
    return SimpleNamespace(value=value)


def status(confirmation="confirmed", error=None):
    return SimpleNamespace(confirmation_status=confirmation, err=error)


class RecoveryRpc:
    def __init__(self, statuses=None, height=301, found_transaction=None):
        self.statuses = list(statuses or [None, None])
        self.height = height
        self.found_transaction = found_transaction
        self.calls = []
        self.send_error = None

    async def get_signature_statuses(self, signatures, *, search_transaction_history):
        self.calls.append(("status", search_transaction_history))
        value = self.statuses.pop(0)
        if isinstance(value, Exception):
            raise value
        return response([value])

    async def get_block_height(self, *, commitment):
        self.calls.append(("height", commitment))
        return response(self.height)

    async def get_slot(self, *, commitment):
        self.calls.append(("slot", commitment))
        return response(400)

    async def get_transaction(self, signature, *, commitment, max_supported_transaction_version):
        self.calls.append(("transaction", commitment, max_supported_transaction_version))
        if isinstance(self.found_transaction, Exception):
            raise self.found_transaction
        return response(self.found_transaction)

    async def send_transaction(self, txn, opts):
        self.calls.append(("send",))
        if self.send_error is not None:
            raise self.send_error
        return response(txn.signatures[0])


def client(rpc):
    result = QuickNodeClient.__new__(QuickNodeClient)
    result.client = rpc
    result._status_attempts = 1
    result._confirmation_timeout_seconds = 0.05
    return result


class RpcRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_timeout_preserves_previously_persisted_local_identity(self):
        rpc = RecoveryRpc()
        rpc.send_error = TimeoutError("response lost after broadcasting")
        saved = []

        async def persist(metadata):
            self.assertEqual(rpc.calls, [])
            saved.append(metadata)

        txn = transaction()
        with self.assertRaises(TransactionStatusUnknown) as caught:
            await client(rpc).send_and_confirm(
                txn, on_submitting=persist, last_valid_block_height=300
            )
        self.assertEqual(caught.exception.signature, str(txn.signatures[0]))
        self.assertEqual(caught.exception.metadata, saved[0])
        self.assertEqual(saved[0]["recent_blockhash"], str(txn.message.recent_blockhash))
        self.assertEqual(saved[0]["last_valid_block_height"], 300)
        self.assertTrue(saved[0]["submitted_at"])
        self.assertEqual(rpc.calls, [("send",)])

    async def test_failed_persistence_prevents_network_submission(self):
        rpc = RecoveryRpc()

        def fail(_):
            raise OSError("disk full")

        with self.assertRaises(OSError):
            await client(rpc).send_and_confirm(transaction(), on_submitting=fail)
        self.assertEqual(rpc.calls, [])

    async def test_processed_status_is_pending_even_with_error(self):
        for error in (None, "InstructionError"):
            rpc = RecoveryRpc(statuses=[status("processed", error)])
            with self.assertRaises(TransactionStatusUnknown):
                await client(rpc).send_and_confirm(transaction())

    async def test_expiry_requires_finalized_horizon_and_two_absence_reads(self):
        rpc = RecoveryRpc()
        state, evidence = await client(rpc).get_expiry_evidence(
            str(transaction().signatures[0]), last_valid_block_height=300
        )
        self.assertEqual(state, "expired")
        self.assertEqual(evidence["expiry_slot"], 400)
        self.assertEqual(evidence["finalized_block_height"], 301)
        self.assertTrue(evidence["history_status_absent"])
        self.assertTrue(evidence["transaction_absent"])
        self.assertEqual(rpc.calls, [
            ("status", True), ("height", "finalized"), ("slot", "finalized"),
            ("status", True), ("transaction", "confirmed", 0),
        ])

    async def test_height_equal_to_horizon_is_still_pending(self):
        rpc = RecoveryRpc(height=300)
        state, _ = await client(rpc).get_expiry_evidence(
            str(transaction().signatures[0]), last_valid_block_height=300
        )
        self.assertEqual(state, "pending")
        self.assertEqual(rpc.calls, [("status", True), ("height", "finalized")])

    async def test_confirmation_race_prevents_expiry(self):
        for confirmation in ("processed", "confirmed", "finalized"):
            rpc = RecoveryRpc(statuses=[None, status(confirmation)])
            state, _ = await client(rpc).get_expiry_evidence(
                str(transaction().signatures[0]), last_valid_block_height=300
            )
            self.assertEqual(state, "pending" if confirmation == "processed" else "confirmed")
            self.assertFalse(any(call[0] == "transaction" for call in rpc.calls))

    async def test_transaction_lookup_prevents_false_expiry(self):
        for error in (None, "InstructionError"):
            found = SimpleNamespace(transaction=SimpleNamespace(meta=SimpleNamespace(err=error)))
            rpc = RecoveryRpc(found_transaction=found)
            state, evidence = await client(rpc).get_expiry_evidence(
                str(transaction().signatures[0]), last_valid_block_height=300
            )
            self.assertEqual(state, "confirmed" if error is None else "failed")
            self.assertTrue(evidence["transaction_found"])

    async def test_unavailable_evidence_never_expires_order(self):
        scenarios = [
            RecoveryRpc(statuses=[TimeoutError("status")]),
            RecoveryRpc(statuses=[None, TimeoutError("status")]),
            RecoveryRpc(found_transaction=TimeoutError("transaction")),
        ]
        for rpc in scenarios:
            state, _ = await client(rpc).get_expiry_evidence(
                str(transaction().signatures[0]), last_valid_block_height=300
            )
            self.assertEqual(state, "unknown")

    async def test_missing_metadata_never_expires_legacy_order(self):
        rpc = RecoveryRpc()
        state, evidence = await client(rpc).get_expiry_evidence(str(transaction().signatures[0]))
        self.assertEqual(state, "unknown")
        self.assertEqual(evidence["reason"], "missing_expiry_metadata")
        self.assertEqual(rpc.calls, [("status", True)])

    async def test_blockhash_alone_cannot_prove_expiry_before_finality(self):
        rpc = RecoveryRpc()
        state, evidence = await client(rpc).get_expiry_evidence(
            str(transaction().signatures[0]), recent_blockhash=str(Hash.default())
        )
        self.assertEqual(state, "unknown")
        self.assertEqual(evidence["reason"], "missing_expiry_metadata")
        self.assertEqual(rpc.calls, [("status", True)])

    async def test_inline_confirmation_has_short_deadline(self):
        rpc = RecoveryRpc()

        async def stalled(*_, **__):
            await asyncio.sleep(20)

        rpc.get_signature_statuses = stalled
        loop = asyncio.get_running_loop()
        started = loop.time()
        with self.assertRaises(TransactionStatusUnknown):
            await client(rpc).send_and_confirm(transaction())
        self.assertLess(loop.time() - started, 0.5)

    async def test_recovery_balance_query_enforces_finalized_slot_anchor(self):
        observed = []

        async def make_request(request, response_type):
            observed.append(json.loads(request.to_json()))
            return response([
                SimpleNamespace(account=SimpleNamespace(data=SimpleNamespace(parsed={
                    "info": {"tokenAmount": {"amount": amount}}
                })))
                for amount in ("100", "25")
            ])

        rpc = SimpleNamespace(_provider=SimpleNamespace(make_request=make_request))
        wallet = str(Keypair.from_seed(bytes(range(32))).pubkey())
        with patch("execution.rpc_handler.ExecutionConfig.get_wallet_address", return_value=wallet):
            balance = await client(rpc).get_token_balance(
                wallet, commitment="finalized", min_context_slot=400
            )
        self.assertEqual(balance, 125)
        self.assertEqual(observed[0]["method"], "getTokenAccountsByOwner")
        config = observed[0]["params"][2]
        self.assertEqual(config["commitment"], "finalized")
        self.assertEqual(config["minContextSlot"], 400)


if __name__ == "__main__":
    unittest.main()
