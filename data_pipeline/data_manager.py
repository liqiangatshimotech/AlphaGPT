import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from loguru import logger

from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider
from .quote_snapshot import (
    DEFAULT_JUPITER_BASE_URL, JupiterQuoteOnlyClient, MAX_MINTS_PER_RUN,
    collect_many, validate_mint,
)


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

    @staticmethod
    def _snapshot_number(value):
        """Keep only finite numeric candidate fields in the local audit log."""
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _snapshot_address(value):
        """Reject malformed identifiers without copying arbitrary API text."""
        if not isinstance(value, str):
            return None
        address = value.strip()
        if not address or len(address) > 128 or not address.isascii() or not address.isprintable():
            return None
        return address

    @staticmethod
    def _record_universe_snapshot(observed_at, candidates, fetch_status):
        """Append one point-in-time discovery record per sync, including [].

        Configure the local path with TOKEN_UNIVERSE_SNAPSHOT_PATH. Each line
        contains only the explicitly whitelisted market fields; API responses,
        headers, and credentials are never serialized.
        """
        path = Path(os.getenv(
            "TOKEN_UNIVERSE_SNAPSHOT_PATH", "logs/token_universe_snapshots.jsonl"
        ))
        record = {
            "observed_at": observed_at,
            "source": "birdeye_trending",
            "fetch_status": fetch_status,
            "candidates": candidates,
        }
        data = (json.dumps(record, ensure_ascii=True, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(fd, data) != len(data):
                raise OSError("short token universe snapshot write")
            os.fsync(fd)
        finally:
            os.close(fd)

    async def _collect_quote_snapshots(self, selected_tokens, observed_at):
        """Sample a few executable round trips without delaying exits.

        The live runner executes this pipeline in a background task. Sampling
        rotates through the selected universe across 15-minute syncs and never
        builds, signs, or sends a swap transaction.
        """
        try:
            requested = int(os.getenv("TOKEN_QUOTE_SAMPLE_SIZE", "0"))
        except ValueError:
            logger.warning("Invalid TOKEN_QUOTE_SAMPLE_SIZE; skipping quote snapshots.")
            return
        if requested <= 0 or not selected_tokens:
            return
        key = os.getenv("JUPITER_API_KEY", "")
        if not key:
            logger.warning("JUPITER_API_KEY unavailable; skipping quote snapshots.")
            return
        count = min(requested, MAX_MINTS_PER_RUN, len(selected_tokens))
        observed_epoch = int(datetime.fromisoformat(observed_at.replace("Z", "+00:00")).timestamp())
        start = ((observed_epoch // 900) * count) % len(selected_tokens)
        mints = []
        seen = set()
        for offset in range(len(selected_tokens)):
            candidate = selected_tokens[(start + offset) % len(selected_tokens)]
            try:
                mint = validate_mint(candidate.get("address"))
            except ValueError:
                continue
            if mint not in seen:
                mints.append(mint)
                seen.add(mint)
            if len(mints) >= count:
                break
        if not mints:
            logger.warning("No valid mints for read-only quote snapshots.")
            return
        output = os.getenv("TOKEN_QUOTE_SNAPSHOT_PATH", "logs/quote_snapshots.jsonl")
        try:
            async with JupiterQuoteOnlyClient(
                key,
                base_url=os.getenv("JUPITER_BASE_URL", DEFAULT_JUPITER_BASE_URL),
            ) as client:
                rows = await collect_many(mints, output, client)
            available = sum(row["status"] == "available" for row in rows)
            logger.info(f"Read-only quote snapshots: {available}/{len(rows)} available")
        except Exception as error:
            # Quote collection is evidence gathering, never a prerequisite
            # for position monitoring or the market-data refresh.
            logger.warning(f"Quote snapshot collection deferred: {type(error).__name__}")

    async def pipeline_sync_daily(self, include_existing=False):
        logger.info("Step 1: Discovering trending tokens...")
        default_limit = 500 if Config.BIRDEYE_IS_PAID else 100
        limit = getattr(Config, "BIRDEYE_TRENDING_LIMIT", default_limit)
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        candidate_snapshot = []
        for token in candidates:
            address = self._snapshot_address(token.get("address"))
            liq = self._snapshot_number(token.get("liquidity"))
            fdv = self._snapshot_number(token.get("fdv"))
            selected = (
                address is not None and address == token.get("address")
                and liq is not None and liq >= Config.MIN_LIQUIDITY_USD
                and fdv is not None and Config.MIN_FDV <= fdv <= Config.MAX_FDV
            )
            candidate_snapshot.append({
                "address": address,
                "liquidity": liq,
                "fdv": fdv,
                "selected": selected,
            })
            if selected:
                selected_tokens.append(token)

        fetch_status = getattr(self.birdeye, "last_trending_status", "success")
        if fetch_status not in {"success", "error"}:
            fetch_status = "unknown"
        try:
            self._record_universe_snapshot(observed_at, candidate_snapshot, fetch_status)
        except Exception as exc:
            logger.warning(f"Token universe snapshot unavailable: {type(exc).__name__}: {exc}")

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

        await self._collect_quote_snapshots(selected_tokens, observed_at)
        logger.success(f"Pipeline complete. New candles stored: {total_candles}")
        return {"inserted": total_candles, "tokens": len(selected_tokens), "end_time": end_time}
