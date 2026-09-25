import json
import unittest
from unittest.mock import AsyncMock

import httpx2
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.core import RPCException
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import Message
from solders.transaction import VersionedTransaction
from solders.rpc.responses import SendTransactionResp

from execution.rpc_handler import (
    QuickNodeClient,
    TransactionPreflightRejected,
    TransactionStatusUnknown,
)


def _transaction():
    payer = Keypair.from_seed(bytes(range(32)))
    return VersionedTransaction(
        Message.new_with_blockhash([], payer.pubkey(), Hash.default()), [payer]
    )


def _response(*, error=None):
    if error is None:
        payload = {"jsonrpc": "2.0", "id": 0, "result": str(_transaction().signatures[0])}
    else:
        payload = {"jsonrpc": "2.0", "id": 0, "error": error}
    return httpx2.Response(
        200, content=json.dumps(payload).encode(),
        request=httpx2.Request("POST", "http://localhost:8899"),
    )


def _preflight_error(*, simulation_error=None):
    if simulation_error is None:
        simulation_error = {"InstructionError": [3, {"Custom": 36}]}
    return {
        "code": -32002,
        "message": "Transaction simulation failed: custom program error: 0x24",
        "data": {
            "err": simulation_error,
            "logs": [
                "Program goonuddtQRrWqqn5nFyczVKaie28f3kDkHWkHtURSLE failed: custom program error: 0x24",
                "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 failed: custom program error: 0x24",
            ],
        },
    }


class RpcPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = QuickNodeClient.__new__(QuickNodeClient)
        self.client.client = AsyncClient("http://localhost:8899", commitment=Confirmed)
        self.client._status_attempts = 1
        self.client._confirmation_timeout_seconds = 0.01

    async def asyncTearDown(self):
        await self.client.close()

    async def test_structured_preflight_is_reported_after_single_http_post(self):
        post = AsyncMock(return_value=_response(error=_preflight_error()))
        self.client.client._provider.session.post = post
        saved = []
        txn = _transaction()

        with self.assertRaises(TransactionPreflightRejected) as caught:
            await self.client.send_and_confirm(
                txn, on_submitting=saved.append, last_valid_block_height=300
            )

        self.assertEqual(post.await_count, 1)
        self.assertEqual(caught.exception.signature, str(txn.signatures[0]))
        self.assertEqual(caught.exception.metadata, saved[0])
        self.assertTrue(caught.exception.single_attempt_proven)
        self.assertEqual(
            caught.exception.failed_program_id,
            "goonuddtQRrWqqn5nFyczVKaie28f3kDkHWkHtURSLE",
        )
        self.assertEqual(len(caught.exception.logs), 2)

    async def test_transport_read_error_does_not_trigger_sdk_retry(self):
        post = AsyncMock(side_effect=httpx2.ReadError("response lost"))
        self.client.client._provider.session.post = post

        with self.assertRaises(TransactionStatusUnknown):
            await self.client.send_and_confirm(_transaction())

        self.assertEqual(post.await_count, 1)

    async def test_success_uses_one_post_and_normal_confirmation(self):
        post = AsyncMock(return_value=_response())
        self.client.client._provider.session.post = post
        self.client.get_signature_status = AsyncMock(
            return_value=("confirmed", "confirmed")
        )
        txn = _transaction()

        signature = await self.client.send_and_confirm(txn)

        self.assertEqual(signature, str(txn.signatures[0]))
        self.assertEqual(post.await_count, 1)

    async def test_unstructured_or_missing_simulation_error_stays_unknown(self):
        empty_simulation = _preflight_error()
        empty_simulation["data"]["err"] = None
        for error in (
            {"code": -32603, "message": "preflight failed"},
            empty_simulation,
        ):
            self.client.client._provider.session.post = AsyncMock(
                return_value=_response(error=error)
            )
            with self.assertRaises(TransactionStatusUnknown):
                await self.client.send_and_confirm(_transaction())

    async def test_fake_client_cannot_claim_single_attempt_evidence(self):
        detail = SendTransactionResp.from_json(json.dumps({
            "jsonrpc": "2.0", "id": 0, "error": _preflight_error(),
        }))

        class FakeRpc:
            async def send_transaction(self, txn, opts):
                raise RPCException(detail)

        real_client = self.client.client
        self.client.client = FakeRpc()
        try:
            with self.assertRaises(TransactionPreflightRejected) as caught:
                await self.client.send_and_confirm(_transaction())
        finally:
            self.client.client = real_client
        self.assertFalse(caught.exception.single_attempt_proven)


if __name__ == "__main__":
    unittest.main()
