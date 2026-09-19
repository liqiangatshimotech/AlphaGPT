"""Poll current token liquidity into liquidity_snapshots only.

This process deliberately does not rewrite OHLCV history.  Use a long-running
process outside the trainer (systemd/cron/container) for production capture.
"""
import argparse
import asyncio
from datetime import datetime

import aiohttp
from loguru import logger

from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider


async def snapshot_once(db, birdeye, dex, limit=50):
    tokens = await birdeye.get_trending_tokens(limit=limit)
    # Keep sub-minute precision so rapid diagnostics do not overwrite the
    # prior observation under the (time,address,source) primary key.
    now = datetime.now()
    rows = [
        (now, token['address'], token.get('liquidity'), token.get('fdv'), 'birdeye_trending')
        for token in tokens
        if token.get('address') and token.get('liquidity') is not None
    ]
    # Dexscreener is optional. Never share Birdeye's credential-bearing
    # session with this independent host.
    try:
        async with aiohttp.ClientSession() as session:
            details = await dex.get_token_details_batch(session, [t['address'] for t in tokens])
        rows.extend(
            (now, item['address'], item.get('liquidity'), item.get('fdv'), 'dexscreener')
            for item in details
            if item.get('address') and item.get('liquidity') is not None
        )
    except Exception as exc:
        logger.warning(f"Dexscreener snapshot unavailable: {type(exc).__name__}")
    inserted = await db.batch_insert_liquidity_snapshots(rows)
    return {'time': now.isoformat(), 'trending': len(tokens), 'snapshots': inserted}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 1 or args.interval < 0:
        raise ValueError('rounds must be positive and interval non-negative')
    if not Config.BIRDEYE_API_KEY:
        raise RuntimeError('BIRDEYE_API_KEY is missing')
    db, birdeye, dex = DBManager(), BirdeyeProvider(), DexScreenerProvider()
    await db.connect(); await db.init_schema()
    try:
        for index in range(args.rounds):
            logger.info(f"Liquidity snapshot round {index + 1}/{args.rounds}")
            logger.info(await snapshot_once(db, birdeye, dex, min(max(args.limit, 1), 50)))
            if index + 1 < args.rounds:
                await asyncio.sleep(args.interval)
    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
