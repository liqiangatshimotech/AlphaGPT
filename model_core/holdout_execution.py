"""Causal, single-position evaluator for frozen research signals.

The evaluator is intentionally independent of ``MemeBacktest``: a decision at
bar ``t`` can only use fields at ``t`` and enters at the next observed open.
Missing execution data is reported as unavailable instead of being treated as a
zero return.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def simulate(
    scores: np.ndarray,
    raw: Dict[str, np.ndarray],
    start: int,
    end: int,
    entry_threshold: float,
    exit_threshold: float | None = None,
    hold_bars: int = 15,
    cooldown: int = 3,
    notional: float = 1000.0,
    fee: float = 0.006,
) -> Dict[str, Any]:
    """Simulate one top-ranked position with causal next-open fills.

    ``scores`` and arrays are shaped ``[tokens, bars]``.  ``raw['observed']``
    is a data-quality mask, not a future tradability oracle.  ``end`` is
    exclusive.  Returns a JSON-serialisable diagnostic dictionary.
    """
    scores = np.asarray(scores, dtype=float)
    open_ = np.asarray(raw['open'], dtype=float)
    close = np.asarray(raw.get('close', open_), dtype=float)
    liq = np.asarray(raw['liquidity'], dtype=float)
    observed = np.asarray(raw['observed'], dtype=bool)
    n, total = scores.shape
    if any(x.shape != (n, total) for x in (open_, close, liq, observed)):
        raise ValueError('scores and raw arrays must have the same shape')
    if not 0 <= start < end <= total or hold_bars < 1 or cooldown < 0:
        raise ValueError('invalid simulation range or holding parameters')
    exit_threshold = entry_threshold if exit_threshold is None else exit_threshold

    cash = float(notional)
    qty = 0.0
    token = -1
    entry_price = 0.0
    entry_bar = -1
    cooldown_left = 0
    entries = 0
    gross_pnl = 0.0
    costs = 0.0
    unresolved = []
    events = []
    per_token = np.zeros(n, dtype=float)
    equity_path = [cash]
    available = True
    bankrupt = False

    def rate(i: int, bar: int, size: float) -> float:
        if not math.isfinite(liq[i, bar]) or liq[i, bar] <= 0:
            raise ValueError('missing or non-positive liquidity at execution')
        return float(fee + min(0.05, size / liq[i, bar]))

    def close_position(i: int, why: str) -> None:
        nonlocal cash, qty, token, entry_price, entry_bar, available, gross_pnl, costs
        if token < 0:
            return
        if not observed[token, i] or not math.isfinite(open_[token, i]) or open_[token, i] <= 0:
            available = False; unresolved.append({'bar': i, 'token': int(token), 'reason': 'missing_exit_open'}); return
        px = float(open_[token, i]); size = qty * px
        try: r = rate(token, i, size)
        except ValueError:
            available = False; unresolved.append({'bar': i, 'token': int(token), 'reason': 'missing_exit_liquidity'}); return
        proceeds = size * (1.0 - r)
        principal = qty * entry_price
        pnl = proceeds - principal
        gross_pnl += size - principal
        costs += size * r
        cash += proceeds
        per_token[token] += pnl / notional
        events.append({'bar': int(i), 'type': 'exit', 'token': int(token), 'reason': why, 'pnl_fraction': float(pnl / notional)})
        qty = 0.0; token = -1; entry_price = 0.0; entry_bar = -1

    for t in range(start, end - 1):
        # An active position is exited at the next open after minimum duration,
        # or after a causal score/eligibility failure at the current bar.
        if token >= 0:
            age = t - entry_bar
            bad_signal = (not observed[token, t]) or (not math.isfinite(scores[token, t])) or scores[token, t] < exit_threshold
            if age >= hold_bars or bad_signal:
                close_position(t + 1, 'hold' if age >= hold_bars else 'signal')
                if token < 0:
                    cooldown_left = cooldown
        if token < 0:
            if cooldown_left:
                cooldown_left -= 1
            else:
                current = scores[:, t]
                eligible = observed[:, t] & np.isfinite(current) & (liq[:, t] > 0) & (current >= entry_threshold)
                if np.any(eligible):
                    i = int(np.nanargmax(np.where(eligible, current, -np.inf)))
                    if cash < notional * 0.05:
                        bankrupt = True
                        break
                    fill = t + 1
                    if (not observed[i, fill]) or (not math.isfinite(open_[i, fill])) or open_[i, fill] <= 0:
                        available = False; unresolved.append({'bar': fill, 'token': i, 'reason': 'missing_entry_open'}); continue
                    try: r = rate(i, fill, notional)
                    except ValueError:
                        available = False; unresolved.append({'bar': fill, 'token': i, 'reason': 'missing_entry_liquidity'}); continue
                    px = float(open_[i, fill]); spend = min(notional, cash / (1.0 + r))
                    if spend <= 0:
                        bankrupt = True; available = False; break
                    qty = spend / px; cash -= spend * (1.0 + r); costs += spend * r
                    token, entry_price, entry_bar = i, px, t
                    entries += 1
                    events.append({'bar': int(fill), 'type': 'entry', 'token': i, 'price': px})
        mark = cash
        if token >= 0 and observed[token, t + 1] and math.isfinite(open_[token, t + 1]) and open_[token, t + 1] > 0:
            mark += qty * float(open_[token, t + 1])
        equity_path.append(mark)

    # Force-close any outstanding position at the final available open.
    if token >= 0:
        close_position(end - 1, 'segment_end')
    final_equity = cash
    curve = np.asarray(equity_path, dtype=float) / notional
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max(peak - curve)) if curve.size else 0.0
    net = float(final_equity / notional - 1.0)
    return {
        'available': bool(available and not unresolved), 'unresolved_missing': unresolved,
        'bankrupt': bool(bankrupt), 'net_pnl_fraction': net,
        'gross_pnl_fraction': float(gross_pnl / notional), 'cost_fraction': float(costs / notional),
        'max_drawdown_fraction': max_dd, 'entries': int(entries),
        'active_tokens': int(np.count_nonzero(per_token)), 'exposure': float(np.mean(np.asarray(equity_path) > notional)),
        'per_token_pnl_fraction': per_token.tolist(), 'entry_events': events,
        'equity_path': curve.tolist(),
    }
