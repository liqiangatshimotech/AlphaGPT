import asyncio
import math
import time
from datetime import datetime

import aiohttp
from loguru import logger
from ..config import Config
from .base import DataProvider


class BirdeyeProvider(DataProvider):
    def __init__(self):
        self.base_url = Config.BIRDEYE_BASE_URL
        self.headers = Config.birdeye_headers()
        self.headers["x-chain"] = Config.CHAIN
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def _get_json(self, session, path, params):
        # All requests share a limiter. Retry outside the lock; never recurse
        # while holding a semaphore (which deadlocks with concurrency=1).
        for attempt in range(Config.BIRDEYE_MAX_RETRIES + 1):
            delay = min(2 ** attempt, 30)
            async with self._rate_lock:
                wait = Config.BIRDEYE_MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    async with session.get(
                        self.base_url + path, params=params,
                        allow_redirects=False, timeout=aiohttp.ClientTimeout(total=40),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("success") is False:
                                raise RuntimeError(f"Birdeye reported failure for {path}")
                            return data
                        if resp.status != 429 and resp.status < 500:
                            raise RuntimeError(f"Birdeye HTTP {resp.status} for {path}")
                        try:
                            delay = max(delay, min(float(resp.headers.get("Retry-After", 0)), 120))
                        except ValueError:
                            pass
                        logger.warning(f"Birdeye HTTP {resp.status}; retry {attempt + 1}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(f"Birdeye request failed: {type(exc).__name__}")
                finally:
                    # Space from completion, including trending -> OHLCV.
                    self._last_request_at = time.monotonic()
            if attempt < Config.BIRDEYE_MAX_RETRIES:
                await asyncio.sleep(delay)
        raise RuntimeError(f"Birdeye retries exhausted for {path}")

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            if value is None:
                return default
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    async def get_trending_tokens(self, limit=50):
        limit = min(max(int(limit), 1), 50)
        async with aiohttp.ClientSession(headers=self.headers) as session:
            data = await self._get_json(session, "/defi/token_trending", {
                "sort_by": "rank", "sort_type": "asc", "offset": 0, "limit": limit,
            })
        return [{
            "address": t["address"], "symbol": t.get("symbol", "UNKNOWN"),
            "name": t.get("name", "UNKNOWN"), "decimals": t.get("decimals", 6),
            "liquidity": self._as_float(t.get("liquidity")),
            "fdv": self._as_float(t.get("fdv")),
        } for t in data.get("data", {}).get("tokens", [])]

    async def get_token_history(self, session, address, days=Config.HISTORY_DAYS,
                                liquidity=None, fdv=None, end_time=None, start_time=None):
        intervals = {"1m": 60, "15m": 900}
        if Config.TIMEFRAME not in intervals:
            raise ValueError("Historical paging currently supports 1m and 15m")
        interval = intervals[Config.TIMEFRAME]
        # Exclude the current incomplete candle and use the same cutoff for all tokens.
        end = (int(time.time()) if end_time is None else int(end_time)) // interval * interval
        cursor = end - int(days * 86400)
        if start_time is not None:
            cursor = max(cursor, int(start_time) // interval * interval)
        records = {}
        while cursor < end:
            # At most 900 intervals per call, below the observed 1000-candle cap.
            page_end = min(cursor + 900 * interval, end)
            data = await self._get_json(session, "/defi/ohlcv", {
                "address": address, "type": Config.TIMEFRAME,
                "time_from": cursor, "time_to": page_end - 1,
            })
            for item in data.get("data", {}).get("items", []):
                stamp = int(item["unixTime"])
                if not cursor <= stamp < page_end:
                    continue
                values = [float(item[k]) for k in ("o", "h", "l", "c", "v")]
                if not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0 or values[4] < 0:
                    raise ValueError("Invalid OHLCV candle received")
                records[stamp] = (
                    datetime.fromtimestamp(stamp), address, *values,
                    # OHLCV responses often omit historical pool depth.  Do
                    # not copy the current token snapshot onto every old bar;
                    # unknown historical liquidity/FDV must remain NULL and
                    # be handled as unavailable by research/execution.
                    self._as_float(item.get("liquidity"), None),
                    self._as_float(item.get("fdv"), None), "birdeye",
                )
            cursor = page_end
        return [records[k] for k in sorted(records)]
