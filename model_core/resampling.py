"""Causal resampling helpers for short-lived token OHLCV data.

The data pipeline stores one-minute candles, while research may use a short
bar such as three or five minutes.  This module deliberately keeps gaps
visible: an interval with missing one-minute candles is returned with its
partial aggregate, but ``observed`` and ``complete`` are ``False``.  Callers
must use that mask before calculating features or placing orders.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
OPTIONAL_COLUMNS = ("volume", "liquidity", "fdv")


def _as_timedelta(value: str | pd.Timedelta) -> pd.Timedelta:
    """Parse a short-bar interval and reject non-positive values."""

    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid resampling interval: {value!r}") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError("resampling interval must be positive")
    return delta


def _row_valid(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Return a strict per-candle data-quality mask.

    Prices must be positive.  Volume, liquidity, and FDV are allowed to be
    zero (zero liquidity is useful information for the execution filter), but
    all values must be finite.  An explicitly supplied ``observed`` mask is
    respected as well.
    """

    mask = pd.Series(True, index=frame.index, dtype=bool)
    if "observed" in frame:
        mask &= frame["observed"].fillna(False).astype(bool)
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        mask &= np.isfinite(values)
        if name in PRICE_COLUMNS:
            mask &= values > 0
        else:
            mask &= values >= 0
    return mask


def resample_ohlcv(
    data: pd.DataFrame,
    interval: str | pd.Timedelta = "5min",
    *,
    time_col: str = "time",
    group_col: str = "address",
    source_interval: str | pd.Timedelta = "1min",
    include_empty: bool = True,
) -> pd.DataFrame:
    """Aggregate one-minute candles into short OHLCV bars.

    Parameters
    ----------
    data:
        A frame containing ``time``, optional ``address``, and ``open``,
        ``high``, ``low``, ``close``.  ``volume``, ``liquidity`` and ``fdv``
        are aggregated when present.  An input ``observed`` boolean can be
        supplied to mark source rows that are known to be real candles.
    interval:
        Target bar width, for example ``"3min"`` or ``"5min"``.  The target
        must be an integer multiple of ``source_interval``.
    source_interval:
        Width of one source candle.  It is one minute for the current data
        pipeline and is explicit here so completeness is not guessed.
    include_empty:
        Include empty buckets between the first and last bucket of each
        address.  Empty buckets have NaN values and ``observed=False``.  Set
        this to ``False`` when only populated buckets are wanted.

    Returns
    -------
    pandas.DataFrame
        One row per address and target bucket.  ``open/high/low/close`` are
        first/max/min/last valid source prices; ``volume`` is summed;
        ``liquidity`` and ``fdv`` are the last valid values in the bucket.
        ``observed`` is true only when every expected source interval is
        present and valid.  ``complete`` is an alias retained for callers
        that want to make the quality check explicit.  ``source_count`` and
        ``valid_count`` expose why a bucket was marked incomplete.

    Notes
    -----
    A partial bucket is *not* silently dropped by default.  Its aggregate is
    useful for diagnostics, while the false mask prevents look-ahead or
    fabricated signals in downstream research.  Empty buckets are explicit
    rows only when ``include_empty=True``.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if time_col not in data.columns:
        raise ValueError(f"missing time column: {time_col!r}")
    missing = [name for name in PRICE_COLUMNS if name not in data.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {', '.join(missing)}")

    target = _as_timedelta(interval)
    source = _as_timedelta(source_interval)
    ratio = target / source
    expected = int(round(ratio))
    if expected < 1 or not np.isclose(ratio, expected):
        raise ValueError("interval must be an integer multiple of source_interval")

    frame = data.copy()
    try:
        frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("time column contains invalid timestamps") from exc
    if frame[time_col].isna().any():
        raise ValueError("time column contains null timestamps")

    # Keep only fields that actually exist.  Missing optional fields should not
    # be manufactured as zeros: a caller can then distinguish unavailable
    # liquidity/FDV from a measured zero.
    value_columns = [name for name in PRICE_COLUMNS + OPTIONAL_COLUMNS if name in frame]
    valid_columns = [name for name in value_columns if name in frame]
    frame["__valid"] = _row_valid(frame, valid_columns)
    frame["__bucket"] = frame[time_col].dt.floor(target)
    has_group = group_col in frame.columns
    if not has_group:
        frame["__group"] = "__single__"
        group_name = "__group"
    else:
        group_name = group_col
        # Null addresses are not useful for joins or model tensors.
        frame = frame[frame[group_col].notna()].copy()

    output_columns = ([time_col] + ([group_col] if has_group else []) + value_columns +
                      ["observed", "complete", "source_count", "valid_count"])
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict] = []
    # Stable sort makes first/last deterministic if a source has duplicate
    # timestamps.  Duplicate timestamps count once toward completeness.
    frame = frame.sort_values([group_name, "__bucket", time_col], kind="mergesort")
    for group_value, grouped in frame.groupby(group_name, sort=False, dropna=False):
        buckets = grouped["__bucket"]
        if include_empty:
            start, end = buckets.min(), buckets.max()
            bucket_values = pd.date_range(start=start, end=end, freq=target)
        else:
            bucket_values = pd.Index(pd.unique(buckets)).sort_values()
        for bucket in bucket_values:
            part = grouped[grouped["__bucket"] == bucket]
            valid = part[part["__valid"]]
            # Keep unique source timestamps for the completeness check.  A
            # duplicated source row must not make a partial bucket complete.
            valid_times = valid[time_col].dt.floor(source).drop_duplicates()
            valid_count = int(len(valid_times))
            source_count = int(part[time_col].dt.floor(source).drop_duplicates().size)
            complete = valid_count == expected
            row: dict = {time_col: bucket}
            if has_group:
                row[group_col] = group_value
            if valid.empty:
                for name in value_columns:
                    row[name] = np.nan
            else:
                # Value columns are numeric in the database; coercion here
                # keeps malformed rows from leaking strings into aggregates.
                for name in PRICE_COLUMNS:
                    values = pd.to_numeric(valid[name], errors="coerce")
                    if name == "open":
                        row[name] = float(values.iloc[0])
                    elif name == "high":
                        row[name] = float(values.max())
                    elif name == "low":
                        row[name] = float(values.min())
                    elif name == "close":
                        row[name] = float(values.iloc[-1])
                if "volume" in value_columns:
                    row["volume"] = float(pd.to_numeric(valid["volume"], errors="coerce").sum())
                for name in ("liquidity", "fdv"):
                    if name in value_columns:
                        values = pd.to_numeric(valid[name], errors="coerce")
                        row[name] = float(values.iloc[-1])
            row.update({"observed": bool(complete), "complete": bool(complete),
                        "source_count": source_count, "valid_count": valid_count})
            records.append(row)

    result = pd.DataFrame.from_records(records, columns=output_columns)
    if has_group:
        result = result.sort_values([group_col, time_col], kind="mergesort")
    else:
        result = result.sort_values([time_col], kind="mergesort")
    return result.reset_index(drop=True)


# Descriptive aliases make the helper convenient for callers that use either
# terminology while keeping one implementation and one set of semantics.
aggregate_ohlcv = resample_ohlcv
resample_token_ohlcv = resample_ohlcv


__all__ = ["resample_ohlcv", "aggregate_ohlcv", "resample_token_ohlcv"]
