"""Load OHLCV features for training and live strategy inference.

The database stores naive UTC candle times. In live mode, callers can pass a
maximum candle age so old tokens are excluded before the top-N selection and
before any missing values are forward filled into the feature tensor.
"""

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd
import sqlalchemy
import torch

from .config import ModelConfig
from .factors import FeatureEngineer


class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.addresses = []
        # Shared UTC time axis and actual SQL-row provenance for the tensors.
        # observed_mask is [token, time] and stays false for both pre-listing
        # zeros and forward-filled gaps or stale tails.
        self.timestamps = []
        self.observed_mask = None
        # Actual last stored candle for each selected address, never a time
        # inferred from the forward-filled feature matrix.
        self.latest_candle_times = {}

    @staticmethod
    def _utc_naive(value):
        """Translate a caller's clock to the database's naive UTC format."""
        if not isinstance(value, datetime):
            raise TypeError("as_of must be a datetime")
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def load_data(self, limit_tokens=500, *, max_candle_age_seconds=None, as_of=None):
        """Load the most populated token histories.

        ``max_candle_age_seconds`` is intended for live scans. Tokens whose
        *actual* newest candle is older than ``as_of - max_candle_age_seconds``
        (or has a future timestamp) are excluded in SQL before applying
        ``limit_tokens``. Omitting it preserves historical training behavior.
        Naive ``as_of`` values are interpreted as UTC.
        """
        if isinstance(limit_tokens, bool) or not isinstance(limit_tokens, int) or limit_tokens < 1:
            raise ValueError("limit_tokens must be a positive integer")
        if max_candle_age_seconds is not None:
            try:
                age = float(max_candle_age_seconds)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_candle_age_seconds must be positive and finite") from exc
            if not math.isfinite(age) or age <= 0:
                raise ValueError("max_candle_age_seconds must be positive and finite")
            now = self._utc_naive(as_of or datetime.now(timezone.utc))
            cutoff = now - timedelta(seconds=age)

        # A failed refresh must not leave the previous tensor available to a
        # later live scan. Position monitoring can continue without AI data.
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.addresses = []
        self.timestamps = []
        self.observed_mask = None
        self.latest_candle_times = {}

        print("Loading data from SQL...")
        valid_candle_sql = """
            o.open > 0 AND o.high > 0 AND o.low > 0 AND o.close > 0
            AND o.open < 1e30 AND o.high < 1e30
            AND o.low < 1e30 AND o.close < 1e30
            AND o.volume >= 0 AND o.volume < 1e30
            AND o.high >= o.open AND o.high >= o.close
            AND o.low <= o.open AND o.low <= o.close
        """
        freshness_clause = "HAVING MAX(o.time) BETWEEN :cutoff AND :as_of" if max_candle_age_seconds is not None else ""
        top_query = sqlalchemy.text(f"""
            SELECT t.address, MAX(o.time) AS latest_candle_time
            FROM tokens AS t
            JOIN ohlcv AS o ON o.address = t.address AND {valid_candle_sql}
            GROUP BY t.address
            {freshness_clause}
            ORDER BY COUNT(*) DESC, t.address ASC
            LIMIT :limit
        """)
        params = {"limit": limit_tokens}
        if max_candle_age_seconds is not None:
            params.update(cutoff=cutoff, as_of=now)
        selected = pd.read_sql(top_query, self.engine, params=params)
        addresses = selected["address"].tolist()
        if not addresses:
            raise ValueError("No fresh tokens found." if max_candle_age_seconds is not None else "No tokens found.")

        # Select by bound parameters rather than interpolating token addresses
        # supplied by the database into a second SQL statement.
        data_query = sqlalchemy.text(f"""
            SELECT o.time, o.address, o.open, o.high, o.low, o.close,
                   o.volume, o.liquidity, o.fdv
            FROM ohlcv AS o
            WHERE o.address IN :addresses AND {valid_candle_sql}
            ORDER BY time ASC
        """).bindparams(sqlalchemy.bindparam("addresses", expanding=True))
        df = pd.read_sql(data_query, self.engine, params={"addresses": addresses})
        if df.empty:
            raise ValueError("Selected tokens have no OHLCV rows.")

        latest_candle_times = {
            row.address: pd.Timestamp(row.latest_candle_time).to_pydatetime().replace(tzinfo=timezone.utc)
            for row in selected.itertuples(index=False)
        }

        # Build provenance before forward fill. A database row with a missing
        # or invalid price/volume is not an observed candle for causal models.
        values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
        open_, high, low, close, volume = values.T
        valid_candle = (
            np.isfinite(values).all(axis=1)
            & (open_ > 0) & (high > 0) & (low > 0) & (close > 0)
            & (volume >= 0)
            & (high >= np.maximum(open_, close))
            & (low <= np.minimum(open_, close))
        )
        observed = (df[["time", "address"]].assign(_observed=valid_candle)
                    .pivot(index="time", columns="address", values="_observed")
                    .reindex(columns=addresses)
                    .astype("boolean")
                    .fillna(False).astype(bool))
        timestamps = [stamp.to_pydatetime()
                      for stamp in pd.to_datetime(observed.index, utc=True)]
        observed_mask = torch.tensor(
            observed.to_numpy(dtype=bool).T.copy(),
            dtype=torch.bool,
            device=ModelConfig.DEVICE,
        )

        def to_tensor(col):
            pivot = df.pivot(index="time", columns="address", values=col)
            pivot = pivot.reindex(index=observed.index, columns=addresses)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": to_tensor("close"),
            "volume": to_tensor("volume"),
            "liquidity": to_tensor("liquidity"),
            "fdv": to_tensor("fdv"),
        }
        feat_tensor = FeatureEngineer.compute_features(raw_data_cache)
        op = raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        target_ret = torch.log(t2 / (t1 + 1e-9))
        target_ret[:, -2:] = 0.0

        self.addresses = addresses
        self.timestamps = timestamps
        self.observed_mask = observed_mask
        self.latest_candle_times = latest_candle_times
        self.raw_data_cache = raw_data_cache
        self.feat_tensor = feat_tensor
        self.target_ret = target_ret
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")
