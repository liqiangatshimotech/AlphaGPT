"""Mint precision must be verified before converting an exit quote to price."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from solders.keypair import Keypair

from execution.config import ExecutionConfig
from execution.utils import get_mint_decimals


TOKEN = str(Keypair.from_seed(bytes(range(32))).pubkey())


class MintDecimalsTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_supply_supplies_verified_precision(self):
        client = SimpleNamespace(get_token_supply=AsyncMock(
            return_value=SimpleNamespace(value=SimpleNamespace(decimals=9))
        ))
        self.assertEqual(await get_mint_decimals(TOKEN, client), 9)
        client.get_token_supply.assert_awaited_once()

    async def test_rpc_error_never_defaults_to_six(self):
        client = SimpleNamespace(get_token_supply=AsyncMock(side_effect=TimeoutError("rpc unavailable")))
        with self.assertRaises(TimeoutError):
            await get_mint_decimals(TOKEN, client)

    async def test_missing_or_invalid_precision_is_rejected(self):
        client = SimpleNamespace(get_token_supply=AsyncMock(
            return_value=SimpleNamespace(value=None)
        ))
        with self.assertRaises(ValueError):
            await get_mint_decimals(TOKEN, client)
        client.get_token_supply.return_value = SimpleNamespace(
            value=SimpleNamespace(decimals=255)
        )
        with self.assertRaises(ValueError):
            await get_mint_decimals(TOKEN, client)

    async def test_native_sol_precision_needs_no_rpc(self):
        client = SimpleNamespace(get_token_supply=AsyncMock())
        self.assertEqual(await get_mint_decimals(ExecutionConfig.SOL_MINT, client), 9)
        client.get_token_supply.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
