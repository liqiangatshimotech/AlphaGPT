import asyncio
import unittest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import Message
from solders.transaction import VersionedTransaction

from execution.rpc_handler import (
    QuickNodeClient,
    TransactionStatusUnknown,
)


class _ConfirmedStatus:
    err = None
    confirmation_status = "confirmed"


class _Response:
    def __init__(self, value):
        self.value = value


class _FakeClient:
    async def send_transaction(self, txn, opts):
        return _Response(txn.signatures[0])

    async def get_signature_statuses(self, signatures, *, search_transaction_history):
        assert search_transaction_history is True
        return _Response([_ConfirmedStatus()])


class _UnavailableClient(_FakeClient):
    async def get_signature_statuses(self, signatures, *, search_transaction_history):
        raise TimeoutError("status endpoint timed out")


def _client(fake, attempts=1):
    client = QuickNodeClient.__new__(QuickNodeClient)
    client.client = fake
    client._status_attempts = attempts
    return client


def _signed_transaction():
    payer = Keypair.from_seed(bytes(range(32)))
    return VersionedTransaction(
        Message.new_with_blockhash([], payer.pubkey(), Hash.default()), [payer]
    )


class RpcConfirmationTests(unittest.TestCase):
    def test_confirmation_returns_signature_after_status_is_confirmed(self):
        transaction = _signed_transaction()
        result = asyncio.run(_client(_FakeClient()).send_and_confirm(transaction))
        self.assertEqual(result, str(transaction.signatures[0]))

    def test_confirmation_timeout_is_unknown_and_keeps_signature(self):
        transaction = _signed_transaction()
        with self.assertRaises(TransactionStatusUnknown) as caught:
            asyncio.run(_client(_UnavailableClient()).send_and_confirm(transaction))
        self.assertEqual(caught.exception.signature, str(transaction.signatures[0]))


if __name__ == "__main__":
    unittest.main()
