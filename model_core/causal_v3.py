"""Experimental causal formula features and VM for offline strategy research.

This module deliberately does not replace the v2 live strategy. Every enabled
factor at candle ``t`` depends only on candles observed through ``t``. The mask
must come from the database's actual (token, time) rows, before forward filling.

LIQ_SCORE (token 1) is disabled. The historical Birdeye importer sometimes
filled old candles with a *later* trending snapshot's liquidity and FDV. A
row-observation mask alone cannot establish that those values were known at
that candle. v3 rejects any formula using LIQ_SCORE until a separately proven
historical provenance source is available.
"""

import torch

from .vocab import FORMULA_VOCAB


CAUSAL_V3_VERSION = 3
FEATURE_COUNT = FORMULA_VOCAB.feature_count


def _check_mask(mask, shape):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != shape:
        raise ValueError("observed_mask must be a bool tensor matching [tokens, time]")


def _observed_lag(x, observed_mask, distance):
    """Value at the preceding observed candle, skipping absent grid rows."""
    if distance < 1:
        raise ValueError("distance must be positive")
    ranks = observed_mask.long().cumsum(dim=1)
    target_ranks = ranks - distance
    indices = torch.searchsorted(ranks.contiguous(), target_ranks.contiguous())
    indices = indices.clamp(max=x.shape[1] - 1)
    lagged = x.gather(1, indices)
    return torch.where(observed_mask & (target_ranks > 0), lagged, 0.0)


def _rolling_observed_stats(x, observed_mask, window):
    """Rolling mean/std over the latest ``window`` actual observations."""
    if window < 1:
        raise ValueError("window must be positive")
    ranks = observed_mask.long().cumsum(dim=1)
    before_ranks = (ranks - window).clamp(min=0)
    before_indices = torch.searchsorted(
        ranks.contiguous(), before_ranks.contiguous(), right=True,
    ) - 1
    before_indices = before_indices.clamp(min=0)

    values = torch.where(observed_mask, x, 0.0).to(torch.float64)
    total = values.cumsum(dim=1)
    total_sq = values.square().cumsum(dim=1)
    prior = torch.where(before_ranks > 0, total.gather(1, before_indices), 0.0)
    prior_sq = torch.where(before_ranks > 0, total_sq.gather(1, before_indices), 0.0)
    count = ranks.clamp(max=window)
    denominator = count.clamp(min=1)
    mean = (total - prior) / denominator
    variance = ((total_sq - prior_sq) / denominator - mean.square()).clamp(min=0.0)
    return mean.to(x.dtype), variance.sqrt().to(x.dtype), count


def _rolling_zscore(x, observed_mask, *, window=120, min_count=5):
    mean, std, count = _rolling_observed_stats(x, observed_mask, window)
    # The floor also prevents a constant run of candles from magnifying the
    # first tiny change into a very large formula input.
    z = ((x - mean) / std.clamp(min=1e-3)).clamp(-5.0, 5.0)
    return torch.where(observed_mask & (count >= min_count), z, 0.0)


class CausalV3FeatureEngineer:
    """Six slots in the v2 token order; LIQ_SCORE is a disabled zero slot."""

    INPUT_DIM = FEATURE_COUNT

    @staticmethod
    def compute_features(raw_dict, observed_mask):
        required = ("open", "high", "low", "close", "volume")
        if any(name not in raw_dict for name in required):
            raise ValueError("raw_dict is missing an OHLCV field")
        close = raw_dict["close"]
        if close.ndim != 2 or close.shape[1] < 1:
            raise ValueError("OHLCV tensors must have shape [tokens, time]")
        _check_mask(observed_mask, close.shape)
        if observed_mask.device != close.device:
            raise ValueError("observed_mask and OHLCV tensors must use one device")
        for name in required:
            value = raw_dict[name]
            if not isinstance(value, torch.Tensor) or value.shape != close.shape:
                raise ValueError("OHLCV tensor shapes differ")
            if value.device != close.device:
                raise ValueError("OHLCV tensor devices differ")
            if not torch.isfinite(value[observed_mask]).all().item():
                raise ValueError(f"observed {name} contains non-finite values")
        for name in ("open", "high", "low", "close"):
            if (raw_dict[name][observed_mask] <= 0).any().item():
                raise ValueError(f"observed {name} must be positive")
        open_values = raw_dict["open"][observed_mask]
        high_values = raw_dict["high"][observed_mask]
        low_values = raw_dict["low"][observed_mask]
        close_values = raw_dict["close"][observed_mask]
        if ((high_values < torch.maximum(open_values, close_values))
                | (low_values > torch.minimum(open_values, close_values))).any().item():
            raise ValueError("observed candle has an invalid high-low range")
        if (raw_dict["volume"][observed_mask] < 0).any().item():
            raise ValueError("observed volume must be nonnegative")

        def observed(name):
            return torch.where(observed_mask, raw_dict[name], 0.0)

        open_ = observed("open")
        high = observed("high")
        low = observed("low")
        close = observed("close")
        volume = observed("volume")

        ranks = observed_mask.long().cumsum(dim=1)
        previous_close = _observed_lag(close, observed_mask, 1)
        # Subtract logs instead of forming a price ratio, which can overflow
        # float32 for two individually valid prices far apart in scale.
        tiny = torch.finfo(close.dtype).tiny
        ret = torch.where(
            observed_mask & (ranks > 1),
            torch.log(close.clamp(min=tiny))
            - torch.log(previous_close.clamp(min=tiny)),
            0.0,
        )
        liq_score = torch.zeros_like(close)
        pressure = torch.where(
            observed_mask,
            torch.tanh(((close - open_) / (high - low).clamp(min=1e-9)) * 3.0),
            0.0,
        )
        previous_volume = _observed_lag(volume, observed_mask, 1)
        volume_change = torch.where(
            observed_mask & (ranks > 1),
            (volume - previous_volume) / (previous_volume + 1.0),
            0.0,
        )
        previous_change = _observed_lag(volume_change, observed_mask, 1)
        fomo = (volume_change - previous_change).clamp(-5.0, 5.0)
        close_ma, _, _ = _rolling_observed_stats(close, observed_mask, 20)
        deviation = torch.where(
            observed_mask,
            (close - close_ma) / close_ma.clamp(min=1e-12),
            0.0,
        )
        log_volume = torch.log1p(volume)
        features = torch.stack((
            _rolling_zscore(ret, observed_mask),
            liq_score,
            pressure,
            _rolling_zscore(fomo, observed_mask),
            _rolling_zscore(deviation, observed_mask),
            _rolling_zscore(log_volume, observed_mask),
        ), dim=1)
        if not torch.isfinite(features).all().item():
            raise ValueError("causal features contain non-finite values")
        return features


class CausalV3StackVM:
    """Observation-aware version of the v2 postfix operators.

    DIV returns zero when the denominator is near zero; JUMP normalizes over
    preceding actual observations only. LIQ_SCORE formulas are rejected because
    old candle liquidity/FDV provenance is unknown. Invalid arithmetic rejects
    the whole formula instead of silently replacing non-finite results.
    """

    @staticmethod
    def execute(formula_tokens, feat_tensor, observed_mask):
        if feat_tensor.ndim != 3 or feat_tensor.shape[1] != FEATURE_COUNT:
            raise ValueError("features must have shape [tokens, 6, time]")
        _check_mask(observed_mask, (feat_tensor.shape[0], feat_tensor.shape[2]))
        if observed_mask.device != feat_tensor.device:
            raise ValueError("observed_mask and features must use one device")
        if not torch.isfinite(feat_tensor.masked_select(observed_mask[:, None, :])).all().item():
            return None

        stack = []
        for token in formula_tokens:
            if type(token) is not int or not 0 <= token < FORMULA_VOCAB.size:
                return None
            if token == 1:  # Historical LIQ_SCORE has no trusted as-of source.
                return None
            if token < FEATURE_COUNT:
                stack.append(torch.where(observed_mask, feat_tensor[:, token, :], 0.0))
                continue

            operator = FORMULA_VOCAB.operator_names[token - FEATURE_COUNT]
            arity = 3 if operator == "GATE" else 2 if operator in {"ADD", "SUB", "MUL", "DIV"} else 1
            if len(stack) < arity:
                return None
            args = stack[-arity:]
            del stack[-arity:]
            if operator == "ADD":
                result = args[0] + args[1]
            elif operator == "SUB":
                result = args[0] - args[1]
            elif operator == "MUL":
                result = args[0] * args[1]
            elif operator == "DIV":
                safe = args[1].abs() >= 1e-3
                denominator = torch.where(safe, args[1], 1.0)
                result = torch.where(safe, args[0] / denominator, 0.0)
            elif operator == "NEG":
                result = -args[0]
            elif operator == "ABS":
                result = args[0].abs()
            elif operator == "SIGN":
                result = args[0].sign()
            elif operator == "GATE":
                result = torch.where(args[0] > 0, args[1], args[2])
            elif operator == "JUMP":
                result = torch.relu(_rolling_zscore(args[0], observed_mask, window=60) - 3.0)
            elif operator == "DECAY":
                result = args[0] + 0.8 * _observed_lag(args[0], observed_mask, 1) + 0.6 * _observed_lag(args[0], observed_mask, 2)
            elif operator == "DELAY1":
                result = _observed_lag(args[0], observed_mask, 1)
            elif operator == "MAX3":
                result = torch.maximum(args[0], torch.maximum(
                    _observed_lag(args[0], observed_mask, 1),
                    _observed_lag(args[0], observed_mask, 2),
                ))
            else:
                return None
            result = torch.where(observed_mask, result, 0.0)
            if not torch.isfinite(result).all().item():
                return None
            stack.append(result)

        return stack[0] if len(stack) == 1 else None
