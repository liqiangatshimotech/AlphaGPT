from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import asyncpg
from loguru import logger

from .config import Config


_OHLCV_COLUMNS = (
    "time",
    "address",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "liquidity",
    "fdv",
    "source",
)

class DBManager:
    def __init__(self, pool=None):
        """Create a database manager.

        ``pool`` is injectable for tests and for callers that already own a
        pool.  Normal application code should leave it unset and call
        :meth:`connect`.
        """
        self.pool = pool

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

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

    @staticmethod
    def _normalise_ohlcv_records(records):
        """Return records with UTC-aware datetimes made timestamp-compatible.

        ``ohlcv.time`` is a PostgreSQL ``TIMESTAMP`` (without time zone), and
        this project treats that value as UTC.  Existing providers emit naive
        UTC values.  If a caller supplies an aware value, convert it to UTC
        before stripping the timezone so asyncpg does not reject it.
        """
        normalised = []
        for record in records:
            values = list(record)
            if len(values) != len(_OHLCV_COLUMNS):
                raise ValueError(
                    f"Each OHLCV record must contain {len(_OHLCV_COLUMNS)} values; "
                    f"got {len(values)}"
                )
            candle_time = values[0]
            if isinstance(candle_time, datetime) and candle_time.tzinfo is not None:
                values[0] = candle_time.astimezone(timezone.utc).replace(tzinfo=None)
            normalised.append(tuple(values))
        return normalised

    async def batch_insert_ohlcv(self, records) -> int:
        """Insert candles and return the number of newly inserted rows.

        A direct ``COPY`` into ``ohlcv`` aborts the whole copy when even one
        primary-key duplicate is present.  We copy into a transaction-scoped
        temporary staging table, then perform one set-based insert with
        ``ON CONFLICT DO NOTHING``.  Existing candles are never overwritten,
        and the count comes from PostgreSQL's ``RETURNING`` rows rather than
        from the number fetched from the provider.

        Any database error is allowed to propagate to the caller.  The
        transaction rolls back both the staged data and the target insert.
        """
        if records is None:
            return 0
        records = self._normalise_ohlcv_records(records)
        if not records:
            return 0
        if self.pool is None:
            raise RuntimeError("Database is not connected")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # ON COMMIT DROP keeps staging isolated and avoids leaking a
                # temporary relation across pooled connections.
                await conn.execute(
                    """
                    CREATE TEMP TABLE _ohlcv_stage (
                        LIKE ohlcv INCLUDING DEFAULTS
                    ) ON COMMIT DROP;
                    """
                )
                await conn.copy_records_to_table(
                    "_ohlcv_stage",
                    records=records,
                    columns=_OHLCV_COLUMNS,
                    timeout=60,
                )
                inserted = await conn.fetchval(
                    """
                    WITH inserted AS (
                        INSERT INTO ohlcv (
                            time, address, open, high, low, close,
                            volume, liquidity, fdv, source
                        )
                        SELECT
                            time, address, open, high, low, close,
                            volume, liquidity, fdv, source
                        FROM _ohlcv_stage
                        ON CONFLICT (time, address) DO NOTHING
                        RETURNING 1
                    )
                    SELECT count(*)::integer FROM inserted;
                    """
                )
        return int(inserted or 0)

    async def get_latest_ohlcv_times(
        self, addresses: Optional[Sequence[str]] = None
    ) -> Mapping[str, Optional[datetime]]:
        """Return the latest stored candle time for each requested token.

        The returned datetimes are naive UTC values, matching PostgreSQL's
        ``TIMESTAMP`` column and the provider records used by this project.
        When ``addresses`` is omitted, every token present in ``ohlcv`` is
        returned.  Requested addresses with no candles map to ``None``.
        """
        if self.pool is None:
            raise RuntimeError("Database is not connected")

        requested = None if addresses is None else list(dict.fromkeys(addresses))
        if requested == []:
            return {}

        async with self.pool.acquire() as conn:
            if requested is None:
                rows = await conn.fetch(
                    """
                    SELECT address, MAX(time) AS latest_time
                    FROM ohlcv
                    GROUP BY address;
                    """
                )
                return {row["address"]: row["latest_time"] for row in rows}

            rows = await conn.fetch(
                """
                SELECT address, MAX(time) AS latest_time
                FROM ohlcv
                WHERE address = ANY($1::text[])
                GROUP BY address;
                """,
                requested,
            )
            latest_by_address = {
                row["address"]: row["latest_time"] for row in rows
            }
            return {address: latest_by_address.get(address) for address in requested}

    async def get_latest_ohlcv_time(self, address: str) -> Optional[datetime]:
        """Return one token's latest stored candle time, or ``None``."""
        latest = await self.get_latest_ohlcv_times([address])
        return latest[address]
