import time
from datetime import datetime
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

    async def pipeline_sync_daily(self, include_existing=False):
        logger.info("Step 1: Discovering trending tokens...")
        limit = Config.BIRDEYE_TRENDING_LIMIT
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长
            
            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        latest = await self.db.get_latest_candles()
        if include_existing:
            selected_addresses = {t['address'] for t in selected_tokens}
            selected_tokens.extend(t for addr, t in latest.items() if addr not in selected_addresses)
        
        if not selected_tokens:
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)

        logger.info(f"Incremental OHLCV sync for {len(selected_tokens)} tokens...")
        
        end_time = int(time.time()) // 60 * 60
        total_candles = 0
        async with aiohttp.ClientSession(headers=self.birdeye.headers) as session:
            for i, token in enumerate(selected_tokens, 1):
                last_time = latest.get(token['address'], {}).get('latest_time')
                # DB timestamps follow the provider's local-naive convention.
                # Recheck one boundary candle; insertion is conflict-safe.
                start_time = int(last_time.timestamp()) if last_time else None
                records = await self.birdeye.get_token_history(
                    session, token['address'], liquidity=token.get('liquidity'),
                    fdv=token.get('fdv'), end_time=end_time, start_time=start_time,
                )
                inserted = await self.db.batch_insert_ohlcv(records)
                total_candles += inserted
                logger.info(f"History {i}/{len(selected_tokens)}: {len(records)} candles fetched, {inserted} new rows")
            # Store current pool observations separately.  These values are
            # valid at snapshot time only; they are intentionally not copied
            # onto the historical OHLCV rows.
            snapshot_time = datetime.fromtimestamp(end_time)
            # Birdeye trending already provided a current pool snapshot;
            # persist it even when the optional Dexscreener supplement is
            # unreachable. This is point-in-time data only.
            snapshots = [
                (snapshot_time, item['address'], item.get('liquidity'),
                 item.get('fdv'), 'birdeye_trending')
                for item in selected_tokens
                if item.get('address') and item.get('liquidity') is not None
            ]
            try:
                # Use a separate session: Birdeye's API key header must never
                # be sent to the independent Dexscreener host.
                async with aiohttp.ClientSession() as snapshot_session:
                    details = await self.dexscreener.get_token_details_batch(
                        snapshot_session, [t['address'] for t in selected_tokens]
                    )
                snapshots.extend([
                    (snapshot_time, item['address'], item.get('liquidity'),
                     item.get('fdv'), 'dexscreener')
                    for item in details
                    if item.get('address') and item.get('liquidity') is not None
                ])
            except Exception as exc:
                # OHLCV collection remains usable when the optional snapshot
                # provider is unavailable; the missing depth is visible in
                # the separate table and never silently backfilled.
                logger.warning(f"Liquidity snapshot collection unavailable: {type(exc).__name__}")
            inserted_snapshots = await self.db.batch_insert_liquidity_snapshots(snapshots)
            logger.info(f"Liquidity snapshots stored: {inserted_snapshots}")
        logger.success(f"Pipeline complete. New candles stored: {total_candles}")
        return {'inserted': total_candles, 'tokens': len(selected_tokens),
                'end_time': end_time, 'liquidity_snapshots': inserted_snapshots if 'inserted_snapshots' in locals() else 0}
