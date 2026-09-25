"""Read-only research dataset for the causal v3 path.

Unlike ``CryptoDataLoader`` (used by the live runner), this loader never
forward-fills: unobserved (token, minute) cells hold zero and ``observed`` is
false. Nothing downstream may treat those cells as tradable prices.

Time convention: ``times`` holds UTC epoch seconds of each candle's *start*
(Birdeye ``unixTime``). A 1m candle is complete at ``time + 60``. The grid is
the sorted union of all stored minutes, so adjacent columns need not be one
minute apart; execution code must compare timestamps, not column offsets.

Known limitation: the ``ohlcv`` table has no ingestion timestamp, so nothing
proves a historical candle was available to the live system at ``time + 60``.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import torch

SOL_MINT = "So11111111111111111111111111111111111111112"
CANDLE_SECONDS = 60
RAW_FIELDS = ("open", "high", "low", "close", "volume")
SPLIT_META_KEYS = ("split_range", "selection_end_epoch", "max_tokens")


@dataclass
class ResearchDataset:
    addresses: list
    times: np.ndarray            # int64 [T], UTC candle-start seconds
    observed: torch.Tensor       # bool [N, T]
    raw: dict                    # name -> float32 tensor [N, T], zero if unobserved
    prices: dict                 # "open"/"close" -> float64 ndarray [N, T]
    sol_times: np.ndarray        # int64, SOL/USD candle starts
    sol_close: np.ndarray        # float64 SOL/USD closes
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self):
        return len(self.addresses)

    def fingerprint(self):
        """Digest of every array a formula, label or simulation can read.

        Covers the universe, time axis, observation mask, *all* raw feature
        inputs (OHLCV), the execution prices and the SOL/USD series, together
        with each array's name, dtype and shape, so changing any of them makes
        a frozen selection refuse to run.
        """
        digest = hashlib.sha256()

        def add(label, array):
            array = np.ascontiguousarray(array)
            digest.update(f"{label}|{array.dtype.str}|{array.shape}|".encode())
            digest.update(array.tobytes())

        digest.update(("\x1f".join(self.addresses) + "\x1e").encode())
        add("times", self.times)
        add("observed", self.observed.cpu().numpy())
        for name in sorted(self.raw):
            add(f"raw.{name}", self.raw[name].cpu().numpy())
        for name in sorted(self.prices):
            add(f"prices.{name}", self.prices[name])
        add("sol_times", self.sol_times)
        add("sol_close", self.sol_close)
        # Metadata that decides split boundaries / token selection. Hashed
        # whenever any is present (every current loader writes split_range),
        # so editing or deleting it changes the fingerprint; datasets that
        # never had it (older snapshots, synthetic) keep their fingerprint.
        split_meta = {key: self.meta.get(key) for key in SPLIT_META_KEYS}
        if any(value is not None for value in split_meta.values()):
            digest.update(("split_meta|" + json.dumps(split_meta, sort_keys=True)).encode())
        return digest.hexdigest()[:24]

    def check_consistency(self):
        """Raise ValueError unless the arrays describe one coherent market.

        Features read ``raw`` while fills read ``prices``; on observed cells
        raw open/close must be the float32 image of the execution prices, and
        unobserved cells must hold zero everywhere.
        """
        observed = self.observed.cpu().numpy()
        shape = (self.num_tokens, len(self.times))
        if observed.shape != shape:
            raise ValueError(f"observed shape {observed.shape} != {shape}")
        missing = [name for name in RAW_FIELDS if name not in self.raw]
        if missing or set(self.prices) != {"open", "close"}:
            raise ValueError(f"dataset fields incomplete: raw missing {missing}, prices {sorted(self.prices)}")
        for name in RAW_FIELDS:
            if tuple(self.raw[name].shape) != shape:
                raise ValueError(f"raw.{name} shape {tuple(self.raw[name].shape)} != {shape}")
        for name in ("open", "close"):
            price = self.prices[name]
            raw = self.raw[name].cpu().numpy()
            if price.shape != shape:
                raise ValueError(f"prices.{name} shape {price.shape} != {shape}")
            if not np.array_equal(raw[observed], price[observed].astype(np.float32)):
                raise ValueError(f"raw.{name} disagrees with execution prices on observed candles")
            if np.any(price[~observed] != 0) or np.any(raw[~observed] != 0):
                raise ValueError(f"{name} has values on unobserved cells")
        if len(self.sol_times) != len(self.sol_close) or np.any(np.diff(self.sol_times) <= 0):
            raise ValueError("SOL/USD series must be strictly increasing and aligned")
        return True

    def sol_usd_asof(self, when):
        """(close, age_seconds) of the latest SOL candle completed by ``when``.

        Returns (None, None) before the first completed SOL candle. Callers
        decide whether ``age_seconds`` is acceptable.
        """
        index = np.searchsorted(self.sol_times + CANDLE_SECONDS, when, side="right") - 1
        if index < 0:
            return None, None
        completed = int(self.sol_times[index]) + CANDLE_SECONDS
        return float(self.sol_close[index]), int(when - completed)

    def sol_usd_asof_many(self, when, max_age_seconds):
        """Vectorised ``sol_usd_asof`` with the age limit applied.

        Returns (price, valid): ``price`` is NaN wherever no SOL candle had
        completed by ``when`` or the latest one is older than the limit.
        """
        when = np.asarray(when, dtype=np.int64)
        index = np.searchsorted(self.sol_times + CANDLE_SECONDS, when, side="right") - 1
        safe = np.maximum(index, 0)
        if len(self.sol_times) == 0:
            return np.full(when.shape, np.nan), np.zeros(when.shape, dtype=bool)
        age = when - (self.sol_times[safe] + CANDLE_SECONDS)
        valid = (index >= 0) & (age <= max_age_seconds)
        return np.where(valid, self.sol_close[safe], np.nan), valid

    def to_snapshot(self):
        return {
            "addresses": self.addresses,
            "times": torch.from_numpy(self.times.copy()),
            "observed": self.observed.cpu(),
            "raw": {name: value.cpu() for name, value in self.raw.items()},
            "prices": {name: torch.from_numpy(value.copy()) for name, value in self.prices.items()},
            "sol_times": torch.from_numpy(self.sol_times.copy()),
            "sol_close": torch.from_numpy(self.sol_close.copy()),
            "meta": self.meta,
        }

    @classmethod
    def from_snapshot(cls, data):
        return cls(
            addresses=list(data["addresses"]),
            times=data["times"].numpy().astype(np.int64),
            observed=data["observed"].bool(),
            raw={name: value.float() for name, value in data["raw"].items()},
            prices={name: value.numpy().astype(np.float64) for name, value in data["prices"].items()},
            sol_times=data["sol_times"].numpy().astype(np.int64),
            sol_close=data["sol_close"].numpy().astype(np.float64),
            meta=dict(data.get("meta", {})),
        )


def train_end_for_range(start, end, train_fraction):
    """Training cutoff (epoch seconds) of the range [start, end).

    The single definition shared by ``--max-tokens`` token selection and
    ``compute_splits``, so tokens are ranked on exactly the training window.
    """
    return int(start) + int((int(end) - int(start)) * train_fraction)


def _naive_utc(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).replace(tzinfo=None)


def _epoch(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def iso(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def build_dataset(rows, sol_rows, meta=None):
    """Build a dataset from (time_s, address, o, h, l, c, v) rows.

    ``rows`` must already exclude invalid candles. Duplicate (address, time)
    rows are ambiguous provenance and rejected.
    """
    import pandas as pd

    df = pd.DataFrame(rows, columns=["time", "address", "open", "high", "low", "close", "volume"])
    if df.empty:
        raise ValueError("no OHLCV rows")
    if df.duplicated(["time", "address"]).any():
        raise ValueError("duplicate (address, time) candles")
    df["time"] = df["time"].astype(np.int64)
    addresses = sorted(df["address"].unique().tolist())
    times = np.sort(df["time"].unique()).astype(np.int64)
    columns = {address: index for index, address in enumerate(addresses)}
    t_index = np.searchsorted(times, df["time"].to_numpy())
    a_index = df["address"].map(columns).to_numpy()
    shape = (len(addresses), len(times))
    observed = np.zeros(shape, dtype=bool)
    observed[a_index, t_index] = True
    arrays = {}
    for name in RAW_FIELDS:
        grid = np.zeros(shape, dtype=np.float64)
        grid[a_index, t_index] = df[name].to_numpy(dtype=np.float64)
        arrays[name] = grid
    sol = pd.DataFrame(sol_rows, columns=["time", "close"]) if len(sol_rows) else pd.DataFrame(columns=["time", "close"])
    sol = sol.sort_values("time")
    return ResearchDataset(
        addresses=addresses,
        times=times,
        observed=torch.from_numpy(observed),
        raw={name: torch.from_numpy(value.astype(np.float32)) for name, value in arrays.items()},
        prices={"open": arrays["open"], "close": arrays["close"]},
        sol_times=sol["time"].to_numpy(dtype=np.int64),
        sol_close=sol["close"].to_numpy(dtype=np.float64),
        meta=dict(meta or {}),
    )


def valid_candle_sql(alias):
    """Validity and provenance filter for ``ohlcv`` rows under ``alias``."""
    a = alias
    return f"""
    {a}.open > 0 AND {a}.high > 0 AND {a}.low > 0 AND {a}.close > 0
    AND {a}.open < 1e30 AND {a}.high < 1e30 AND {a}.low < 1e30 AND {a}.close < 1e30
    AND {a}.volume >= 0 AND {a}.volume < 1e30
    AND {a}.high >= {a}.open AND {a}.high >= {a}.close
    AND {a}.low <= {a}.open AND {a}.low <= {a}.close
    AND COALESCE({a}.source, '') <> 'test'
"""


VALID_CANDLE_SQL = valid_candle_sql("o")


def candle_query_sql(*, limit_tokens):
    """Token candle query. With ``limit_tokens`` the universe is the tokens
    with the most *valid, non-test* candles before ``:selection_end``; ranking
    on raw row counts would let test or invalid rows take slots that the
    outer filter then drops."""
    selection_sql = ""
    if limit_tokens:
        selection_sql = f"""
            AND o.address IN (
                SELECT o2.address FROM ohlcv AS o2
                WHERE o2.time >= :start AND o2.time < :selection_end AND o2.address <> :sol
                  AND {valid_candle_sql("o2")}
                GROUP BY o2.address ORDER BY COUNT(*) DESC, o2.address ASC LIMIT :limit
            )
        """
    return f"""
        SELECT o.time, o.address, o.open, o.high, o.low, o.close, o.volume
        FROM ohlcv AS o
        WHERE o.time >= :start AND o.time < :end AND o.address <> :sol
          AND {VALID_CANDLE_SQL} {selection_sql}
        ORDER BY o.time ASC
    """


def database_url(environ=None):
    """Database URL from the environment *at call time*.

    ``ModelConfig.DB_URL`` is fixed when ``model_core.config`` is imported,
    which for the research CLI happens before ``.env`` is loaded; reading it
    here would silently fall back to the defaults.
    """
    import os

    env = os.environ if environ is None else environ
    return (f"postgresql://{env.get('DB_USER', 'postgres')}:{env.get('DB_PASSWORD', 'password')}"
            f"@{env.get('DB_HOST', 'localhost')}:5432/{env.get('DB_NAME', 'crypto_quant')}")


def load_from_database(*, start=None, end=None, max_tokens=None, selection_end=None,
                       train_fraction=0.6, clip_to_sol_coverage=True):
    """Load candles read-only from the configured database.

    Tokens are ranked by candle count *before* the training cutoff only, so
    the later validation/test windows do not influence which tokens are
    studied. The requested range is frozen *before* selection and stored as
    ``meta["split_range"]``; splits are computed from it, not from the loaded
    tokens' first/last candles, so a selected token that stops early cannot
    move the training cutoff before the selection cutoff.
    The SOL mint is excluded from the tradable universe and loaded separately
    as the SOL/USD conversion series. When ``clip_to_sol_coverage`` is set the
    dataset ends where SOL/USD coverage ends, since SOL accounting after that
    point would require an invented conversion rate.
    """
    from dotenv import load_dotenv
    import pandas as pd
    import sqlalchemy

    load_dotenv()
    engine = sqlalchemy.create_engine(database_url())
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("SET TRANSACTION READ ONLY"))
        bounds = conn.execute(sqlalchemy.text(
            f"SELECT MIN(o.time), MAX(o.time) FROM ohlcv AS o WHERE {VALID_CANDLE_SQL}"
        )).one()
        sol = pd.read_sql(sqlalchemy.text(
            f"SELECT o.time, o.close FROM ohlcv AS o WHERE o.address = :sol AND {VALID_CANDLE_SQL} ORDER BY o.time"
        ), conn, params={"sol": SOL_MINT})
        db_min, db_max = bounds
        start = start or db_min
        end = end or (db_max + pd.Timedelta(minutes=1))
        clipped_from = None
        if clip_to_sol_coverage and not sol.empty:
            sol_end = sol["time"].max() + pd.Timedelta(minutes=1)
            if sol_end < end:
                clipped_from = end
                end = sol_end
        range_start, range_end = _epoch(start), _epoch(end)
        params = {"start": start, "end": end, "sol": SOL_MINT}
        selection_end_epoch = None
        if max_tokens is not None:
            selection_end_epoch = (_epoch(selection_end) if selection_end is not None
                                   else train_end_for_range(range_start, range_end, train_fraction))
            selection_end = _naive_utc(selection_end_epoch)
            params.update(selection_end=selection_end, limit=int(max_tokens))
        df = pd.read_sql(sqlalchemy.text(candle_query_sql(limit_tokens=max_tokens is not None)),
                         conn, params=params)
        conn.rollback()
    to_seconds = lambda series: (pd.to_datetime(series).astype("int64") // 10**9).astype(np.int64)
    df["time"] = to_seconds(df["time"])
    sol_rows = list(zip(to_seconds(sol["time"]), sol["close"].astype(float)))
    meta = {
        "source": "postgres ohlcv (read-only)",
        "db_min_time": str(db_min),
        "db_max_time": str(db_max),
        "start": str(start),
        "end_exclusive": str(end),
        "clipped_end_from": str(clipped_from) if clipped_from is not None else None,
        "max_tokens": max_tokens,
        "selection_end": str(selection_end) if selection_end is not None else None,
        "selection_end_epoch": selection_end_epoch,
        "split_range": [range_start, range_end],
        "rows": int(len(df)),
        "sol_rows": int(len(sol)),
        "excluded": ["SOL mint (conversion series only)", "source='test' rows", "invalid OHLCV"],
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    return build_dataset(df.itertuples(index=False, name=None), sol_rows, meta)


def synthetic_dataset(*, tokens=12, minutes=2400, seed=0, gap_probability=0.03,
                      start=1_758_000_000):
    """Irregular synthetic market for tests and smoke runs (no database)."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(tokens):
        listing = int(rng.integers(0, minutes // 5))
        delisting = minutes if rng.random() < 0.7 else int(rng.integers(minutes // 2, minutes))
        price = float(np.exp(rng.normal(-6, 3)))
        drift = rng.normal(0, 0.0004)
        for minute in range(listing, delisting):
            ret = drift + rng.standard_t(3) * 0.01
            open_ = price
            close = max(price * float(np.exp(ret)), 1e-12)
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            price = close
            if rng.random() < gap_probability:
                continue
            volume = float(np.exp(rng.normal(8, 1.5)))
            rows.append((start + minute * 60, f"TOKEN{index:03d}", open_, high, low, close, volume))
    sol_price = 150.0
    sol_rows = []
    for minute in range(minutes):
        sol_price *= float(np.exp(rng.normal(0, 0.0008)))
        sol_rows.append((start + minute * 60, sol_price))
    return build_dataset(rows, sol_rows, {"source": "synthetic", "seed": seed, "tokens": tokens, "minutes": minutes})
