"""Causal, single-position evaluator with next-open order execution.

Signal/data-quality masks describe the completed current bar.  Fill availability
is deliberately separate: a future bar's completed OHLCV mask cannot decide
whether its opening price was available.  Missing fills or valuations are
reported, never converted into a fictitious zero-price liquidation.
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
    eligible: np.ndarray | None = None,
    min_hold_bars: int = 1,
    min_liquidity: float = 0.0,
    max_impact: float = 0.05,
    stop_loss: float | None = None,
    liquidity_drop_fraction: float | None = None,
) -> Dict[str, Any]:
    """Simulate one top-ranked position using information available at close t.

    Decisions at ``t`` execute at open ``t+1``.  A position entered at open
    ``f`` reaches its maximum duration at open ``f+hold_bars``.  The minimum
    duration restricts signal exits; missing data, liquidity risk, stop loss,
    maximum duration, and end-of-segment liquidation take precedence.

    Arrays have shape ``[tokens, bars]``. ``eligible`` is an optional current
    signal mask; it never changes ``raw['observed']``.  ``open_available`` in
    raw controls fills (default: a finite positive open).  Execution liquidity
    comes from ``execution_liquidity`` when supplied, else ``liquidity``; callers
    must provide the liquidity known at that open, not end-of-bar snapshots.

    After an exit at open ``e``, ``cooldown`` full decision bars beginning at
    ``e`` are skipped.  The next decision is at ``e+cooldown`` and the next
    possible entry is open ``e+cooldown+1``.  No decision can both exit and enter.

    Entry impact above ``max_impact`` is rejected. Exits above it are flagged
    unavailable as reliable executions; an uncapped cost diagnostic is retained
    if proceeds can be valued. Missing equity marks are represented by None;
    an unresolved final position makes final P&L None instead of -100%.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError('scores must have shape [tokens, bars]')
    n, total = scores.shape

    def array(name, default=None, dtype=float):
        value = raw[name] if name in raw else default
        if value is None:
            raise ValueError(f'missing raw array: {name}')
        result = np.asarray(value, dtype=dtype)
        if result.shape != (n, total):
            raise ValueError('scores and raw arrays must have the same shape')
        return result

    open_ = array('open')
    close = array('close', open_)
    liq = array('liquidity')
    observed = array('observed', dtype=bool)
    open_available = array('open_available', np.isfinite(open_) & (open_ > 0), bool)
    execution_liq = array('execution_liquidity', liq)
    signal_eligible = np.ones_like(observed) if eligible is None else np.asarray(eligible, dtype=bool)
    if signal_eligible.shape != (n, total):
        raise ValueError('eligible must have the same shape as scores')
    if (not 0 <= start < end <= total or n < 1 or
            not isinstance(hold_bars, (int, np.integer)) or hold_bars < 1 or
            not isinstance(min_hold_bars, (int, np.integer)) or not 1 <= min_hold_bars <= hold_bars or
            not isinstance(cooldown, (int, np.integer)) or cooldown < 0):
        raise ValueError('invalid simulation range or holding parameters')
    if (not math.isfinite(notional) or notional <= 0 or
            not math.isfinite(fee) or not 0 <= fee < 1 or
            not math.isfinite(min_liquidity) or min_liquidity < 0 or
            not math.isfinite(max_impact) or not 0 <= max_impact < 1):
        raise ValueError('invalid notional, cost, or liquidity parameter')
    for name, value in [('stop_loss', stop_loss), ('liquidity_drop_fraction', liquidity_drop_fraction)]:
        if value is not None and (not math.isfinite(value) or not 0 < value < 1):
            raise ValueError(f'{name} must be a fraction between zero and one')
    exit_threshold = entry_threshold if exit_threshold is None else exit_threshold
    if not math.isfinite(entry_threshold) or math.isnan(exit_threshold):
        raise ValueError('invalid entry or exit threshold')

    cash = float(notional)
    qty = 0.0
    token = -1
    entry_price = 0.0
    entry_fill = -1
    entry_liquidity = 0.0
    next_entry_decision = start
    entries = 0
    gross_pnl = 0.0
    costs = 0.0
    unresolved = []
    risk_events = []
    rejections = []
    events = []
    per_token = np.zeros(n, dtype=float)
    traded_tokens = set()
    equity_path: list[float | None] = [1.0]
    equity_bars = [int(start)]
    available = True
    bankrupt = False
    exposed_intervals = 0
    pending_exit_reason: str | None = None

    def missing(bar: int, i: int, reason: str) -> None:
        nonlocal available
        available = False
        unresolved.append({'bar': int(bar), 'token': int(i), 'reason': reason})

    def valid_open(i: int, bar: int) -> bool:
        return bool(observed[i, bar] and open_available[i, bar]
                    and math.isfinite(open_[i, bar]) and open_[i, bar] > 0)

    def valid_close(i: int, bar: int) -> bool:
        return bool(observed[i, bar] and math.isfinite(close[i, bar]) and close[i, bar] > 0)

    def close_position(bar: int, why: str) -> bool:
        nonlocal cash, qty, token, entry_price, entry_fill, entry_liquidity
        nonlocal available, gross_pnl, costs, next_entry_decision, pending_exit_reason
        if not valid_open(token, bar):
            missing(bar, token, 'missing_exit_open')
            return False
        liquidity = float(execution_liq[token, bar])
        if not math.isfinite(liquidity) or liquidity <= 0:
            missing(bar, token, 'missing_exit_liquidity')
            return False
        px = float(open_[token, bar])
        size = qty * px
        impact = size / liquidity
        if liquidity < min_liquidity or impact > max_impact:
            risk_events.append({'bar': int(bar), 'token': int(token),
                                'reason': 'exit_liquidity_limit', 'impact': float(impact),
                                'liquidity': liquidity})
        rate = fee + impact
        if rate >= 1:
            missing(bar, token, 'exit_impact_unpriceable')
            return False
        principal = qty * entry_price
        exit_cost = size * rate
        gross = size - principal
        cash += size - exit_cost
        gross_pnl += gross
        costs += exit_cost
        # Entry costs were charged when bought, so the token total and the
        # gross-minus-cost identity both include each side exactly once.
        per_token[token] += (gross - exit_cost) / notional
        events.append({'bar': int(bar), 'type': 'exit', 'token': int(token),
                       'reason': why, 'price': px, 'held_bars': int(bar - entry_fill),
                       'pnl_fraction': float((gross - exit_cost) / notional),
                       'cost_fraction': float(exit_cost / notional), 'impact': float(impact)})
        qty = 0.0
        token = -1
        entry_price = 0.0
        entry_fill = -1
        entry_liquidity = 0.0
        pending_exit_reason = None
        next_entry_decision = bar + cooldown
        return True

    def close_at_final(bar: int) -> bool:
        """Force-liquidate at the last observed close (no future open)."""
        nonlocal cash, qty, token, entry_price, entry_fill, entry_liquidity
        nonlocal gross_pnl, costs
        if token < 0:
            return True
        i = token
        if not valid_close(i, bar):
            missing(bar, i, 'missing_final_close')
            return False
        liquidity = float(execution_liq[i, bar])
        if not math.isfinite(liquidity) or liquidity <= 0:
            missing(bar, i, 'missing_final_liquidity')
            return False
        px = float(close[i, bar]); size = qty * px; impact = size / liquidity
        rate = fee + impact
        if rate >= 1:
            missing(bar, i, 'final_impact_unpriceable')
            return False
        exit_cost = size * rate; gross = size - qty * entry_price
        cash += size - exit_cost; gross_pnl += gross; costs += exit_cost
        per_token[i] += (gross - exit_cost) / notional
        events.append({'bar': int(bar), 'type': 'exit', 'token': int(i),
                       'reason': 'segment_end', 'price': px,
                       'held_bars': int(bar - entry_fill),
                       'pnl_fraction': float((gross - exit_cost) / notional),
                       'cost_fraction': float(exit_cost / notional),
                       'impact': float(impact)})
        qty = 0.0; token = -1; entry_price = 0.0; entry_fill = -1; entry_liquidity = 0.0
        return True

    def mark_equity(bar: int) -> float | None:
        if token < 0:
            return float(cash / notional)
        if not valid_close(token, bar) or not open_available[token, bar]:
            missing(bar, token, 'missing_mark_close')
            return None
        return float((cash + qty * close[token, bar]) / notional)

    for t in range(start, end - 1):
        fill = t + 1
        held_at_decision = token >= 0
        # Existing holdings are exposed to the movement from open t to open
        # t+1; positions first bought at t+1 are not exposed during this span.
        exposed_intervals += int(held_at_decision)
        if held_at_decision:
            held_after_fill = fill - entry_fill
            why = pending_exit_reason
            current_liq = float(liq[token, t])
            # A missing completed holding bar is an unresolved future
            # observation.  It is not an exit signal and must not be silently
            # replaced by a zero/flat return.
            if (not observed[token, t] or not open_available[token, t]
                    or not math.isfinite(close[token, t]) or close[token, t] <= 0):
                missing(t, token, 'missing_holding_close')
                break
            if not math.isfinite(current_liq) or current_liq <= 0:
                missing(t, token, 'missing_holding_liquidity')
                break
            # Safety checks may bypass the minimum hold, and do not promise a
            # realizable exit when the pool or quote has disappeared.
            if why is None:
                if not observed[token, t]:
                    why = 'missing_signal_data'
                elif current_liq < min_liquidity:
                    why = 'liquidity'
                elif (stop_loss is not None and math.isfinite(close[token, t]) and
                      close[token, t] > 0 and close[token, t] / entry_price - 1 <= -stop_loss):
                    why = 'stop_loss'
                elif liquidity_drop_fraction is not None:
                    reference = entry_liquidity
                    if t > 0 and observed[token, t - 1] and math.isfinite(liq[token, t - 1]):
                        reference = max(reference, float(liq[token, t - 1]))
                    if reference > 0 and current_liq / reference <= 1 - liquidity_drop_fraction:
                        why = 'liquidity_drop'
            if why is None and held_after_fill >= hold_bars:
                why = 'hold'
            # Eligibility loss and an exit-threshold breach are safety exits;
            # minimum holding duration must never prevent them.
            if why is None and (not signal_eligible[token, t] or not math.isfinite(scores[token, t]) or
                                scores[token, t] < exit_threshold):
                why = 'signal'
            if fill == end - 1:
                why = why or 'segment_end'
            if why is not None:
                pending_exit_reason = why
                # A pure segment-end liquidation uses the final close.  A
                # scheduled signal/hold exit still uses next-open execution.
                if why == 'segment_end':
                    if not close_at_final(fill):
                        break
                elif not close_position(fill, why):
                    break
        elif fill < end - 1 and t >= next_entry_decision:
            if cash < notional * 0.05:
                bankrupt = True
            else:
                current = scores[:, t]
                candidates = (observed[:, t] & signal_eligible[:, t] & np.isfinite(current) &
                              np.isfinite(liq[:, t]) & (liq[:, t] > 0) &
                              (liq[:, t] >= min_liquidity) & (current >= entry_threshold))
                if np.any(candidates):
                    i = int(np.argmax(np.where(candidates, current, -np.inf)))
                    if not valid_open(i, fill):
                        missing(fill, i, 'missing_entry_open')
                        break
                    else:
                        liquidity = float(execution_liq[i, fill])
                        if not math.isfinite(liquidity) or liquidity <= 0:
                            missing(fill, i, 'missing_entry_liquidity')
                            break
                        else:
                            # Solve spend*(1+fee)+spend**2/liquidity <= cash,
                            # avoiding an implicit excess debit as cash shrinks.
                            affordable = 2 * cash / ((1 + fee) + math.sqrt((1 + fee) ** 2 + 4 * cash / liquidity))
                            spend = min(notional, affordable)
                            impact = spend / liquidity
                            if liquidity < min_liquidity or impact > max_impact:
                                rejections.append({'bar': int(fill), 'token': i,
                                                   'reason': 'entry_liquidity_limit',
                                                   'liquidity': liquidity, 'impact': float(impact)})
                            else:
                                px = float(open_[i, fill])
                                entry_cost = spend * (fee + impact)
                                qty = spend / px
                                cash = max(0.0, cash - spend - entry_cost)
                                costs += entry_cost
                                per_token[i] -= entry_cost / notional
                                token, entry_price, entry_fill = i, px, fill
                                entry_liquidity = liquidity
                                entries += 1
                                traded_tokens.add(i)
                                events.append({'bar': int(fill), 'decision_bar': int(t),
                                               'type': 'entry', 'token': i, 'price': px,
                                               'cost_fraction': float(entry_cost / notional),
                                               'impact': float(impact)})
        marked = mark_equity(fill)
        if marked is None:
            break
        equity_path.append(marked)
        equity_bars.append(int(fill))

    curve = np.asarray([float(x) for x in equity_path], dtype=float)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max(np.divide(peak - curve, peak, out=np.zeros_like(curve), where=peak > 0)))
    # If a future mark/fill was missing, these are the realized cash figures
    # only and ``available`` is false.  Keeping them finite makes JSON reports
    # safe while preventing callers from mistaking them for a valid backtest.
    final_equity = float(cash)
    net = float(final_equity / notional - 1.0)
    gross = float(gross_pnl / notional)
    return {
        'available': bool(available and not unresolved and token < 0),
        'unresolved_missing': int(len(unresolved)), 'missing_events': unresolved,
        'risk_events': risk_events,
        'entry_rejections': rejections, 'bankrupt': bool(bankrupt),
        'net_pnl_fraction': net, 'gross_pnl_fraction': gross,
        'cost_fraction': float(costs / notional), 'max_drawdown_fraction': max_dd,
        'entries': int(entries), 'active_tokens': len(traded_tokens),
        'exposure': float(exposed_intervals / max(1, end - start - 1)),
        'per_token_pnl_fraction': per_token.tolist(), 'entry_events': events,
        'equity_path': equity_path, 'equity_bars': equity_bars,
        'final_equity': final_equity, 'open_position_token': int(token),
    }
