import asyncpg
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION, 
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    PRIMARY KEY (time, address)
                );
            """)
            
            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

            # Pool depth and FDV are point-in-time observations.  They are not
            # OHLCV candle fields and must never be copied across historical
            # bars when a provider omits them.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS liquidity_snapshots (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    liquidity DOUBLE PRECISION,
                    fdv DOUBLE PRECISION,
                    source TEXT NOT NULL,
                    PRIMARY KEY (time, address, source)
                );
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_liquidity_snapshots_address_time "
                "ON liquidity_snapshots (address, time);"
            )

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (address) DO UPDATE 
                SET symbol = EXCLUDED.symbol, last_updated = NOW();
            """, tokens)

    async def get_latest_candles(self):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT t.address, t.symbol, t.name, t.decimals,
                       o.time AS latest_time, o.liquidity, o.fdv
                FROM tokens t
                JOIN LATERAL (
                    SELECT time, liquidity, fdv FROM ohlcv
                    WHERE address=t.address ORDER BY time DESC LIMIT 1
                ) o ON TRUE
                WHERE t.chain=$1
            """, Config.CHAIN)
            return {r['address']: dict(r) for r in rows}

    async def batch_insert_ohlcv(self, records):
        if not records:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Ignore only conflicting rows, not an entire mixed old/new batch.
                await conn.execute("CREATE TEMP TABLE ohlcv_stage (LIKE ohlcv INCLUDING DEFAULTS) ON COMMIT DROP")
                await conn.copy_records_to_table(
                    'ohlcv_stage', records=records,
                    columns=['time', 'address', 'open', 'high', 'low', 'close',
                             'volume', 'liquidity', 'fdv', 'source'], timeout=60,
                )
                result = await conn.execute(
                    "INSERT INTO ohlcv SELECT * FROM ohlcv_stage ON CONFLICT (time, address) DO NOTHING"
                )
                return int(result.split()[-1])

    async def batch_insert_liquidity_snapshots(self, records):
        """Insert point-in-time pool observations without filling OHLCV gaps."""
        if not records:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.executemany("""
                INSERT INTO liquidity_snapshots (time, address, liquidity, fdv, source)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (time, address, source) DO UPDATE SET
                    liquidity=EXCLUDED.liquidity, fdv=EXCLUDED.fdv
            """, records)
            return len(records)
