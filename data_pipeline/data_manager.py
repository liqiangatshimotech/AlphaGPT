import time
from datetime import datetime, timezone

import aiohttp
from loguru import logger

from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider


class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()

    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    async def _latest_times(self, addresses):
        """Read the latest stored candle for each address.

        ``get_latest_ohlcv_times`` is the small query used by the repaired DB
        manager.  The fallback keeps this manager compatible with an older
        process during a rolling restart and is deliberately read-only.
        """
        getter = getattr(self.db, "get_latest_ohlcv_times", None)
        if getter is not None:
            values = await getter(addresses)
            return values or {}

        # Compatibility with the pre-fix DB manager.  This branch can be
        # removed after every worker has been restarted with the new manager.
        getter = getattr(self.db, "get_latest_candles", None)
        if getter is None:
            return {}
        rows = await getter()
        return {
            address: row.get("latest_time")
            for address, row in (rows or {}).items()
            if isinstance(row, dict)
        }

    @staticmethod
    def _to_epoch(value):
        """Interpret a DB timestamp as UTC, regardless of tz awareness."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return int(value.timestamp())
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            logger.warning(f"Ignoring invalid latest candle timestamp: {value!r}")
            return None

    async def pipeline_sync_daily(self, include_existing=False):
        logger.info("Step 1: Discovering trending tokens...")
        default_limit = 500 if Config.BIRDEYE_IS_PAID else 100
        limit = getattr(Config, "BIRDEYE_TRENDING_LIMIT", default_limit)
        candidates = await self.birdeye.get_trending_tokens(limit=limit)

        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for token in candidates:
            liq = token.get("liquidity", 0)
            fdv = token.get("fdv", 0)

            if liq < Config.MIN_LIQUIDITY_USD:
                continue
            if fdv < Config.MIN_FDV:
                continue
            if fdv > Config.MAX_FDV:
                continue

            selected_tokens.append(token)

        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        addresses = [token["address"] for token in selected_tokens]
        latest = await self._latest_times(addresses)

        # Existing tokens which temporarily leave the trending list can be
        # opted into explicitly.  New DB managers expose times only, so retain
        # the old metadata-aware path when available and otherwise leave the
        # default (trending-only) selection unchanged.
        if include_existing:
            legacy_getter = getattr(self.db, "get_latest_candles", None)
            if legacy_getter is not None:
                existing = await legacy_getter()
                selected_addresses = set(addresses)
                for address, row in (existing or {}).items():
                    if address in selected_addresses or not isinstance(row, dict):
                        continue
                    selected_tokens.append({
                        "address": address,
                        "symbol": row.get("symbol", "UNKNOWN"),
                        "name": row.get("name", "UNKNOWN"),
                        "decimals": row.get("decimals", 6),
                        "liquidity": row.get("liquidity"),
                        "fdv": row.get("fdv"),
                    })
                    latest[address] = row.get("latest_time")

        db_tokens = [
            (token["address"], token["symbol"], token["name"], token["decimals"], Config.CHAIN)
            for token in selected_tokens
        ]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Incremental OHLCV sync for {len(selected_tokens)} tokens...")
        interval = self.birdeye._interval_seconds()
        # Use one UTC epoch cutoff for the whole run and exclude the current
        # incomplete candle.  This avoids mixing local clock values between
        # tokens and keeps freshness calculations in UTC.
        end_time = int(time.time()) // interval * interval
        total_candles = 0

        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            for index, token in enumerate(selected_tokens, 1):
                start_time = self._to_epoch(latest.get(token["address"]))
                records = await self.birdeye.get_token_history(
                    session,
                    token["address"],
                    liquidity=token.get("liquidity"),
                    fdv=token.get("fdv"),
                    end_time=end_time,
                    start_time=start_time,
                )
                inserted = await self.db.batch_insert_ohlcv(records)
                # The repaired DB manager returns the count from ON CONFLICT
                # DO NOTHING.  Treat an old None return as unknown/zero rather
                # than claiming every fetched row was written.
                inserted = int(inserted or 0)
                total_candles += inserted
                logger.info(
                    f"History {index}/{len(selected_tokens)}: "
                    f"{len(records)} candles fetched, {inserted} new rows"
                )

        logger.success(f"Pipeline complete. New candles stored: {total_candles}")
        return {"inserted": total_candles, "tokens": len(selected_tokens), "end_time": end_time}
