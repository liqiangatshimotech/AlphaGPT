"""PostgreSQL regression tests for the OHLCV database writer.

The test connects to the configured local PostgreSQL instance when it is
available, but all relations are temporary and are therefore dropped with the
connection.  A missing local database skips the integration test instead of
creating or mutating application tables.
"""

import asyncio
from datetime import datetime, timezone

import asyncpg
import pytest

from data_pipeline.config import Config
from data_pipeline.db_manager import DBManager


class _SingleConnectionPool:
    """Minimal pool adapter that keeps temporary tables on one connection."""

    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return pool.connection

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Acquire()


async def _connect_or_skip():
    try:
        # Keep the DSN local to this call; never include it in test output.
        return await asyncpg.connect(Config.DB_DSN, timeout=2)
    except Exception as exc:  # pragma: no cover - depends on local services
        pytest.skip(f"PostgreSQL unavailable ({type(exc).__name__})")


def test_batch_insert_skips_duplicates_and_reports_real_count():
    asyncio.run(_test_batch_insert_skips_duplicates_and_reports_real_count())


async def _test_batch_insert_skips_duplicates_and_reports_real_count():
    connection = await _connect_or_skip()
    try:
        await connection.execute(
            """
            CREATE TEMP TABLE ohlcv (
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
            ) ON COMMIT PRESERVE ROWS;
            """
        )
        # CREATE runs in its own implicit transaction. ON COMMIT DROP here
        # would immediately remove the test table, exposing the real ohlcv
        # table through search_path. Keep it until this connection closes.
        assert await connection.fetchval("SELECT to_regclass('pg_temp.ohlcv')") is not None
        db = DBManager(pool=_SingleConnectionPool(connection))
        t1 = datetime(2026, 9, 20, 8, 0)
        t2 = datetime(2026, 9, 20, 8, 1)
        t3 = datetime(2026, 9, 20, 8, 2)

        def candle(candle_time, address, close):
            return (
                candle_time,
                address,
                close,
                close + 1,
                close - 1,
                close,
                10.0,
                20.0,
                30.0,
                "test",
            )

        assert await db.batch_insert_ohlcv(
            [candle(t1, "A", 1), candle(t2, "A", 2), candle(t1, "B", 3)]
        ) == 3

        # The first row is an existing key with changed values.  It must stay
        # untouched while the two genuinely new keys are inserted.
        assert await db.batch_insert_ohlcv(
            [
                candle(t1, "A", 100),
                candle(t3, "A", 4),
                candle(t2, "B", 5),
            ]
        ) == 2

        existing = await connection.fetchrow(
            "SELECT close FROM ohlcv WHERE time = $1 AND address = $2", t1, "A"
        )
        assert existing["close"] == 1

        latest = await db.get_latest_ohlcv_times(["A", "B", "MISSING"])
        assert latest == {"A": t3, "B": t2, "MISSING": None}
        assert await db.get_latest_ohlcv_time("A") == t3

        # The staging relation is transaction-scoped and must not remain on a
        # pooled connection after the operation commits.
        assert await connection.fetchval(
            "SELECT to_regclass('pg_temp._ohlcv_stage')"
        ) is None
    finally:
        await connection.close()


def test_aware_ohlcv_timestamps_are_normalised_to_utc():
    aware = datetime(2026, 9, 20, 16, 0, tzinfo=timezone.utc)
    records = DBManager._normalise_ohlcv_records(
        [
            (
                aware,
                "A",
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                "test",
            )
        ]
    )
    assert records[0][0] == datetime(2026, 9, 20, 16, 0)
    assert records[0][0].tzinfo is None
