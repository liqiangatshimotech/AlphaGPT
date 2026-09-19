import aiohttp
from loguru import logger
from .base import DataProvider
from ..config import Config

class DexScreenerProvider(DataProvider):
    def __init__(self):
        self.base_url = "https://api.dexscreener.com"

    async def get_trending_tokens(self, limit=50):
        url = f"{self.base_url}/tokens/v1/{Config.CHAIN}/solana"
        return []

    async def get_token_details_batch(self, session, addresses):
        valid_data = []
        chunk_size = 30

        for i in range(0, len(addresses), chunk_size):
            chunk = addresses[i:i+chunk_size]
            addr_str = ",".join(chunk)
            url = f"{self.base_url}/tokens/v1/{Config.CHAIN}/{addr_str}"

            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Current official endpoint returns a list; accept the
                        # legacy object shape for older gateways.
                        pairs = data if isinstance(data, list) else data.get('pairs', [])

                        best_pairs = {}
                        for p in pairs:
                            if p['chainId'] != Config.CHAIN: continue
                            base_addr = p['baseToken']['address']
                            liq = float(p.get('liquidity', {}).get('usd', 0))

                            volume = p.get('volume', {}) or {}
                            txns = p.get('txns', {}) or {}
                            m5 = txns.get('m5', {}) or {}
                            if base_addr not in best_pairs or liq > best_pairs[base_addr]['liquidity']:
                                best_pairs[base_addr] = {
                                    'address': base_addr,
                                    'symbol': p['baseToken']['symbol'],
                                    'name': p['baseToken']['name'],
                                    'liquidity': liq,
                                    'fdv': float(p.get('fdv', 0)),
                                    'volume_5m': float(volume.get('m5', 0) or 0),
                                    'txns_5m_buys': int(m5.get('buys', 0) or 0),
                                    'txns_5m_sells': int(m5.get('sells', 0) or 0),
                                    'pair_created_at': p.get('pairCreatedAt'),
                                    'decimals': 6 # 默认
                                }
                        valid_data.extend(best_pairs.values())
            except Exception as e:
                logger.error(f"DexScreener batch error: {e}")

        return valid_data

    async def get_token_history(self, session, address, days):
        return []
