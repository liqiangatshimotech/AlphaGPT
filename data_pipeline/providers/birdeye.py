import asyncio
import math
import time
from datetime import datetime, timezone

import aiohttp
from loguru import logger

from ..config import Config
from .base import DataProvider


class BirdeyeProvider(DataProvider):
    """Birdeye HTTP provider used by the ingestion pipeline.

    The database stores ``TIMESTAMP`` (without time zone) values.  Birdeye
    timestamps are Unix seconds (UTC), so this provider deliberately converts
    them to *naive UTC* datetimes before returning records.  Using
    ``datetime.fromtimestamp`` here would use the host's local timezone and can
    make a healthy feed appear hours stale when the process is moved.
    """

    _INTERVALS = {"1m": 60, "15m": 900, "15min": 900}

    def __init__(self):
        self.base_url = Config.BIRDEYE_BASE_URL
        self.headers = Config.birdeye_headers()
        self.headers["x-chain"] = Config.CHAIN
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        # Birdeye rate limits apply across endpoints.  A single lock avoids a
        # burst when trending and OHLCV requests are made by the same process.
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0
        self.last_trending_status = "never_called"

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            if value is None:
                return default
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _utc_naive(unix_time):
        """Convert Unix seconds to the UTC-naive value expected by Postgres."""
        return datetime.fromtimestamp(int(unix_time), tz=timezone.utc).replace(tzinfo=None)

    @classmethod
    def _interval_seconds(cls):
        try:
            return cls._INTERVALS[Config.TIMEFRAME]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Birdeye timeframe {Config.TIMEFRAME!r}; "
                f"supported values are {', '.join(cls._INTERVALS)}"
            ) from exc

    @staticmethod
    def _retry_limit():
        # Keep the provider bounded even on older Config classes that do not
        # define the newer setting.  An unbounded recursive retry can keep a
        # pipeline cycle alive forever after a rate-limit or network outage.
        try:
            return min(max(int(getattr(Config, "BIRDEYE_MAX_RETRIES", 5)), 0), 10)
        except (TypeError, ValueError):
            return 5

    @staticmethod
    def _minimum_interval():
        try:
            return max(float(getattr(Config, "BIRDEYE_MIN_INTERVAL_SECONDS", 0.0)), 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _get_json(self, session, path, params):
        """Fetch a Birdeye response with a finite, rate-aware retry policy."""
        retries = self._retry_limit()
        minimum_interval = self._minimum_interval()

        for attempt in range(retries + 1):
            delay = min(2 ** attempt, 30)
            async with self.semaphore:
                # Coordinate only the optional request spacing.  Do not hold
                # this lock while waiting for the HTTP response: doing so
                # serialized every token's pages and made a first backfill
                # unnecessarily block live position monitoring.
                async with self._rate_lock:
                    wait = minimum_interval - (time.monotonic() - self._last_request_at)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_request_at = time.monotonic()
                try:
                    async with session.get(
                        self.base_url + path,
                        params=params,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=40),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, dict) and data.get("success") is False:
                                raise RuntimeError(f"Birdeye reported failure for {path}")
                            return data

                        if resp.status not in {429, 500, 502, 503, 504}:
                            body = (await resp.text())[:500]
                            raise RuntimeError(
                                f"Birdeye HTTP {resp.status} for {path}: {body}"
                            )

                        # Retry-After is advisory and may be malformed.  A
                        # cap keeps one bad header from blocking the runner.
                        try:
                            retry_after = float(resp.headers.get("Retry-After", 0))
                            delay = max(delay, min(retry_after, 120.0))
                        except (TypeError, ValueError):
                            pass
                        logger.warning(
                            f"Birdeye HTTP {resp.status} for {path}; "
                            f"retry {attempt + 1}/{retries}"
                        )
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        f"Birdeye request failed for {path}: {type(exc).__name__}; "
                        f"retry {attempt + 1}/{retries}"
                    )

            if attempt < retries:
                await asyncio.sleep(delay)

        raise RuntimeError(f"Birdeye retries exhausted for {path}")

    async def get_trending_tokens(self, limit=50):
        self.last_trending_status = "in_progress"
        limit = min(max(int(limit), 1), 50)
        params = {
            "sort_by": "rank",
            "sort_type": "asc",
            "offset": 0,
            "limit": limit,
        }
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                data = await self._get_json(session, "/defi/token_trending", params)
            payload = data.get("data") if isinstance(data, dict) else None
            raw_list = payload.get("tokens") if isinstance(payload, dict) else None
            if not isinstance(raw_list, list):
                raise ValueError("Birdeye trending response has no token list")
            self.last_trending_status = "success"
            return [
                {
                    "address": t["address"],
                    "symbol": t.get("symbol", "UNKNOWN"),
                    "name": t.get("name", "UNKNOWN"),
                    "decimals": t.get("decimals", 6),
                    "liquidity": self._as_float(t.get("liquidity")),
                    "fdv": self._as_float(t.get("fdv")),
                }
                for t in raw_list
            ]
        except Exception as exc:
            self.last_trending_status = "error"
            logger.error(f"Birdeye Trending Exception: {exc}")
            return []

    async def get_token_history(
        self,
        session,
        address,
        days=Config.HISTORY_DAYS,
        liquidity=None,
        fdv=None,
        end_time=None,
        start_time=None,
    ):
        """Return closed candles, paging under Birdeye's response-size cap.

        ``start_time`` and ``end_time`` are Unix seconds.  Supplying the
        latest stored candle as ``start_time`` lets the data manager request
        only the missing tail on each 15-minute cycle.  The latest boundary is
        intentionally included; the database's ``ON CONFLICT DO NOTHING``
        handles a concurrent/repeated cycle safely while the caller can still
        repair a partially written interval.
        """
        interval = self._interval_seconds()
        end = (int(time.time()) if end_time is None else int(end_time)) // interval * interval
        if end <= 0:
            return []

        history_seconds = max(int(float(days) * 86400), interval)
        cursor = end - history_seconds
        if start_time is not None:
            cursor = max(cursor, int(start_time) // interval * interval)
        if cursor >= end:
            return []

        snapshot_liquidity = self._as_float(liquidity)
        snapshot_fdv = self._as_float(fdv)
        records = {}

        try:
            while cursor < end:
                # Keep each page below Birdeye's ~1000-candle response cap.
                page_end = min(cursor + 900 * interval, end)
                data = await self._get_json(session, "/defi/ohlcv", {
                    "address": address,
                    "type": Config.TIMEFRAME,
                    "time_from": cursor,
                    # The upper bound is exclusive in our cursor handling;
                    # use page_end - 1 to prevent duplicate page boundaries.
                    "time_to": page_end - 1,
                })
                for item in data.get("data", {}).get("items", []):
                    try:
                        stamp = int(item["unixTime"])
                        if stamp < cursor or stamp >= page_end:
                            continue
                        values = [float(item[key]) for key in ("o", "h", "l", "c", "v")]
                        if (
                            not all(math.isfinite(value) for value in values)
                            or min(values[:4]) <= 0
                            or values[4] < 0
                        ):
                            raise ValueError("Invalid OHLCV candle received")
                        records[stamp] = (
                            self._utc_naive(stamp),
                            address,
                            *values,
                            # Birdeye often omits depth/FDV on historical
                            # candles.  Preserve the existing ingestion
                            # contract by falling back to the current token
                            # snapshot supplied by trending-token discovery.
                            self._as_float(item.get("liquidity"), snapshot_liquidity),
                            self._as_float(item.get("fdv"), snapshot_fdv),
                            "birdeye",
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning(f"Skipping malformed Birdeye candle for {address}: {exc}")
                cursor = page_end
        except Exception as exc:
            logger.error(f"Birdeye Fetch Error {address}: {exc}")
            return []

        return [records[stamp] for stamp in sorted(records)]
