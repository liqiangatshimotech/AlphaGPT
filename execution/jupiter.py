import asyncio
import base64
import email.utils
import os
import time
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv
from solders.message import to_bytes_versioned
from loguru import logger
from solders.transaction import VersionedTransaction
from .config import ExecutionConfig
from .rate_limit import wait_for_shared_slot

class JupiterAggregator:
    def __init__(self):
        load_dotenv()
        self.base_url = ExecutionConfig.JUPITER_BASE_URL
        self.headers = {"accept": "application/json"}
        if ExecutionConfig.JUPITER_API_KEY:
            self.headers["x-api-key"] = ExecutionConfig.JUPITER_API_KEY
        self.session = None
        self._min_request_interval = float(
            os.getenv("JUPITER_MIN_REQUEST_INTERVAL_SECONDS", "0.20")
        )
        self._max_retries = int(os.getenv("JUPITER_MAX_RETRIES", "3"))

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    @classmethod
    async def _wait_for_global_slot(cls, interval):
        await wait_for_shared_slot(interval)

    @staticmethod
    def _retry_after_seconds(headers, now=None):
        """Parse Retry-After or a gateway reset epoch without trusting it blindly."""
        now = time.time() if now is None else now
        raw = headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, min(float(raw), 60.0))
            except (TypeError, ValueError):
                try:
                    target = email.utils.parsedate_to_datetime(raw)
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    return max(0.0, min(target.timestamp() - now, 60.0))
                except (TypeError, ValueError, OverflowError):
                    pass
        raw_reset = headers.get("x-ratelimit-reset")
        try:
            return max(0.0, min(float(raw_reset) - now, 60.0)) if raw_reset else 0.0
        except (TypeError, ValueError):
            return 0.0

    async def _backoff_after_429(self, response, attempt):
        retry_after = self._retry_after_seconds(response.headers)
        delay = max(retry_after, min(2.0 ** attempt, 30.0))
        logger.warning(
            f"Jupiter rate limited (429); retrying in {delay:.2f}s "
            f"(attempt {attempt + 1}/{self._max_retries + 1})"
        )
        await asyncio.sleep(delay)

    async def get_quote(self, input_mint, output_mint, amount_integer, slippage_bps=None):
        session = await self._get_session()
        slippage = slippage_bps if slippage_bps else ExecutionConfig.DEFAULT_SLIPPAGE_BPS
        url = f"{self.base_url}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount_integer),
            "slippageBps": str(slippage),
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        for attempt in range(self._max_retries + 1):
            await self._wait_for_global_slot(self._min_request_interval)
            async with session.get(url, params=params, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()

                text = await resp.text()
                if resp.status == 429 and attempt < self._max_retries:
                    await self._backoff_after_429(resp, attempt)
                    continue

                logger.error(f"Jupiter Quote Error {resp.status}: {text}")
                return None

        return None

    async def get_swap_tx(self, quote_response, *, include_metadata=False):
        session = await self._get_session()
        url = f"{self.base_url}/swap"
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": ExecutionConfig.get_wallet_address(),
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": "auto"
        }
        for attempt in range(self._max_retries + 1):
            await self._wait_for_global_slot(self._min_request_interval)
            async with session.post(url, json=payload, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Keep the exact expiry returned with this transaction.
                    # A fresh blockhash from another request is not its expiry.
                    return data if include_metadata else data.get("swapTransaction")
                text = await resp.text()
                if resp.status == 429 and attempt < self._max_retries:
                    await self._backoff_after_429(resp, attempt)
                    continue
                logger.error(f"Jupiter Swap API Error {resp.status}: {text}")
                return None
        return None

    async def close(self):
        if self.session:
            await self.session.close()

    @staticmethod
    def deserialize_and_sign(b64_tx_str):
        try:
            tx_bytes = base64.b64decode(b64_tx_str)
            txn = VersionedTransaction.from_bytes(tx_bytes)
            signature = ExecutionConfig.get_payer_keypair().sign_message(
                to_bytes_versioned(txn.message)
            )
            txn = VersionedTransaction.populate(txn.message, [signature])
            return txn
        except Exception as e:
            logger.error(f"Signing Error: {e}")
            raise
