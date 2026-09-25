"""Executable-rule labels and portfolio simulation for causal v3 research.

Rules mirror ``strategy_manager/runner.py`` where OHLCV data allows:

* A signal from candle ``d`` is known at its close (``time + 60``) plus
  ``decision_latency_seconds``. The buy fills at the *open* of the token's
  first observed candle starting at or after that ready time, if it starts no
  later than ``max_entry_wait_seconds`` after it; otherwise the order is
  counted as unfilled. Grid columns are never assumed to be one minute apart.
* Positions are monitored at each observed candle *close* in the live order:
  stop loss (full), first take profit (sell ``take_profit_ratio`` once, then
  moonbag), trailing stop on the full-exit value high-water mark (initialised
  at the first monitor, as live), then a time exit. A triggered exit fills at
  the next observed open. The v3 time exit replaces the v2 AI exit, whose
  ``abs``-shaped live formula could never score below 0.45.
* SOL accounting: every buy, monitor, sell and period-end valuation converts
  USD candles with the latest SOL/USD candle completed by that moment and
  refuses rates older than ``sol_price_max_age_seconds``. Training labels use
  the same conversion, so rewards optimise the simulated wallet's target.
* Costs: quoted per-side cost (pool fees + price impact, from read-only
  round-trip quotes, halved) + extra adverse slippage per side + a network
  fee per transaction. Nothing else is added, so quoted pool fees are not
  counted twice.

Limitations (reported, not modelled): one-minute OHLCV cannot reproduce
intra-minute stop ordering or real-time full-position quotes; no candle has a
recorded ingestion time; liquidity history is not trustworthy, so the live
500k USD liquidity gate cannot be applied historically.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math

import numpy as np

from .v3_artifact import EXIT_POLICY_VERSION
from .v3_data import CANDLE_SECONDS

_KEY_SCALE = np.int64(10**11)

# Pending-exit / exit-reason codes shared by labels and the simulator.
_NONE, _FULL, _PARTIAL = 0, 1, 2
REASONS = {1: "StopLoss", 2: "Moonbag", 3: "TrailingStop", 4: "TimeExit"}


@dataclass(frozen=True)
class ExitPolicy:
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.10
    take_profit_ratio: float = 0.5
    trailing_activation: float = 0.05
    trailing_drop: float = 0.03
    max_hold_seconds: int = 4 * 3600
    stop_loss_cooldown_seconds: int = 86400
    version: str = EXIT_POLICY_VERSION

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CostModel:
    name: str
    quote_bps_per_side: float
    slippage_bps_per_side: float
    network_fee_sol: float

    @property
    def side_fraction(self):
        return (self.quote_bps_per_side + self.slippage_bps_per_side) / 10_000.0

    def to_dict(self):
        return {**asdict(self), "side_fraction": self.side_fraction}


@dataclass(frozen=True)
class ExecutionSettings:
    max_positions: int = 5
    entry_sol: float = 1.0
    balance_buffer_sol: float = 0.1
    initial_sol: float = 6.0
    buy_threshold: float = 0.85
    decision_latency_seconds: int = 0
    max_entry_wait_seconds: int = 120
    max_exit_wait_seconds: int = 120
    stale_after_seconds: int = 600
    sol_price_max_age_seconds: int = 1800
    min_eligible: int = 2
    min_score_spread: float = 0.01
    max_buy_fraction: float = 0.5
    route_failure_rate: float = 0.0
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def cost_models_from_quotes(path="logs/quote_snapshots.jsonl"):
    """Baseline/adverse/severe cost scenarios from read-only quote snapshots.

    The quote round trip includes both sides' pool fees and price impact at
    the quote's own time; it is halved per side and is *not* a realised fill.
    """
    rows = []
    try:
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError:
        pass
    available = [r for r in rows if r.get("status") == "available"
                 and isinstance(r.get("round_trip_cost_bps"), (int, float))
                 and math.isfinite(r["round_trip_cost_bps"])]
    costs = np.array([r["round_trip_cost_bps"] for r in available], dtype=float)
    if costs.size:
        median, p90, worst = (float(np.median(costs)), float(np.quantile(costs, 0.9)), float(costs.max()))
    else:  # No evidence: fall back to the old backtest's 0.6% per side.
        median = p90 = worst = 120.0
    summary = {
        "path": path,
        "rows": len(rows),
        "available": len(available),
        "availability_rate": (len(available) / len(rows)) if rows else None,
        "round_trip_bps_median": median,
        "round_trip_bps_p90": p90,
        "round_trip_bps_max": worst,
        "first_observed_at": rows[0].get("observed_at") if rows else None,
        "last_observed_at": rows[-1].get("observed_at") if rows else None,
        "mints": len({r.get("token_mint") for r in rows}),
    }
    models = {
        "baseline": CostModel("baseline", median / 2, 25.0, 0.0002),
        "adverse": CostModel("adverse", p90 / 2, 100.0, 0.001),
        "severe": CostModel("severe", worst / 2, 200.0, 0.002),
    }
    return models, summary


class CompactSeries:
    """Observed candles only, concatenated token by token in time order."""

    def __init__(self, dataset):
        observed = dataset.observed.cpu().numpy()
        tok, col = np.nonzero(observed)  # row-major: token, then time
        self.tok = tok.astype(np.int64)
        self.col = col.astype(np.int64)
        self.t = dataset.times[col].astype(np.int64)
        self.open = dataset.prices["open"][tok, col]
        self.close = dataset.prices["close"][tok, col]
        self.key = self.tok * _KEY_SCALE + self.t
        self.size = len(self.tok)
        self.num_tokens = dataset.num_tokens
        self.dataset = dataset
        self.index = np.full(observed.shape, -1, dtype=np.int64)
        self.index[tok, col] = np.arange(self.size)
        previous_same = np.zeros(self.size, dtype=bool)
        previous_same[1:] = self.tok[1:] == self.tok[:-1]
        self.has_previous = previous_same

    def first_at_or_after(self, tokens, when):
        """Compact index of each token's first candle starting >= when (or -1)."""
        found = np.searchsorted(self.key, tokens * _KEY_SCALE + when, side="left")
        clipped = np.minimum(found, self.size - 1)
        ok = (found < self.size) & (self.tok[clipped] == tokens)
        return np.where(ok, found, -1)


def trade_labels(series, policy, cost, settings, split_start, split_end, sol=None):
    """Per-entry outcome of one 1-SOL trade under the portfolio's rules.

    Returns a dict of arrays over decision candles ``d`` (compact indices).
    ``net`` is the SOL return after costs and network fees, using the same
    as-of SOL/USD conversion as ``simulate_portfolio``: the fill candle's open
    for buys and sells, each monitored candle's close for exit checks, and
    the same ``sol_price_max_age_seconds`` limit. Without a fresh rate a buy
    is unfilled, a monitor is skipped and a pending exit waits for the next
    candle, exactly as in the simulator. ``status``: 1 resolved, 2 purged
    (outcome crosses ``split_end`` or the step bound; excluded from training),
    3 written off (token stopped updating with the position open), 0 never
    filled. ``sol`` is the ResearchDataset providing ``sol_usd_asof_many``
    (defaults to ``series.dataset``).
    """
    if policy.max_hold_seconds is None or policy.max_hold_seconds <= 0:
        raise ValueError("vectorised labels require a positive max_hold_seconds")
    sol = series.dataset if sol is None else sol
    t, tok = series.t, series.tok
    max_age = settings.sol_price_max_age_seconds
    sol_open, sol_open_ok = sol.sol_usd_asof_many(t, max_age)
    sol_close, sol_close_ok = sol.sol_usd_asof_many(t + CANDLE_SECONDS, max_age)

    decisions = np.nonzero((t >= split_start) & (t < split_end))[0]
    ready = t[decisions] + CANDLE_SECONDS + settings.decision_latency_seconds
    fills = series.first_at_or_after(tok[decisions], ready)
    safe = np.maximum(fills, 0)
    filled = ((fills >= 0) & (t[safe] - ready <= settings.max_entry_wait_seconds)
              & (t[safe] < split_end) & sol_open_ok[safe])

    count = len(decisions)
    status = np.zeros(count, dtype=np.int8)
    net = np.full(count, np.nan)
    exit_time = np.zeros(count, dtype=np.int64)
    reason = np.zeros(count, dtype=np.int8)
    entries = np.nonzero(filled)[0]
    f = fills[entries]
    e_tok = tok[f]
    side = cost.side_fraction
    fee_fraction = cost.network_fee_sol / settings.entry_sol

    n = len(entries)
    units = (1.0 - side) * sol_open[f] / series.open[f]   # tokens per 1 SOL
    entry_price = 1.0 / units                             # SOL per token
    fill_time = t[f]
    remaining = np.ones(n)
    proceeds = np.zeros(n)
    txs = np.ones(n)
    moonbag = np.zeros(n, dtype=bool)
    highest = np.zeros(n)
    initialized = np.zeros(n, dtype=bool)
    pending = np.zeros(n, dtype=np.int8)
    pending_reason = np.zeros(n, dtype=np.int8)
    active = np.ones(n, dtype=bool)
    e_status = np.zeros(n, dtype=np.int8)
    e_exit = np.zeros(n, dtype=np.int64)
    e_reason = np.zeros(n, dtype=np.int8)

    # Missing SOL rates can defer exits past max_hold; allow for that, and
    # purge (never guess) anything still open at the bound.
    steps = 2 * (policy.max_hold_seconds // CANDLE_SECONDS) + 3
    for k in range(steps + 1):
        live = np.nonzero(active)[0]
        if live.size == 0:
            break
        m = f[live] + k
        mc = np.minimum(m, series.size - 1)
        same = (m < series.size) & (tok[mc] == e_tok[live])
        inside = same & (t[mc] < split_end)
        if k == steps:
            inside[:] = False
        stopped = live[~inside]
        if stopped.size:
            # Judge staleness only from candles before the stop point, so data
            # after split_end (e.g. the token resuming) cannot change a label.
            # m - 1 >= f is always this token's candle once k >= 1; at k = 0
            # the fill candle is inside the split by construction.
            last = t[np.maximum(m[~inside] - 1, 0)]
            writeoff = (k > 0) & (k < steps) & (last + CANDLE_SECONDS + settings.stale_after_seconds < split_end)
            e_status[stopped] = np.where(writeoff, 3, 2)
            e_exit[stopped] = np.where(writeoff, last + CANDLE_SECONDS, 0)
            active[stopped] = False
        g = live[inside]
        mg = mc[inside]

        # Fill exits triggered at the previous observed close; without a fresh
        # SOL rate the exit stays pending and this candle is not monitored.
        has_pending = pending[g] != _NONE
        blocked = has_pending & ~sol_open_ok[mg]
        fill_now = has_pending & ~blocked
        if fill_now.any():
            gi, mi = g[fill_now], mg[fill_now]
            full = pending[gi] == _FULL
            sold = np.where(full, remaining[gi], remaining[gi] * policy.take_profit_ratio)
            proceeds[gi] += sold * units[gi] * series.open[mi] * (1.0 - side) / sol_open[mi]
            remaining[gi] -= sold
            txs[gi] += 1
            moonbag[gi] |= ~full
            done = gi[full]
            e_status[done] = 1
            e_exit[done] = t[mi[full]]
            e_reason[done] = pending_reason[done]
            active[done] = False
            pending[gi] = _NONE

        watch = active[g] & ~blocked & sol_close_ok[mg]
        g2, m2 = g[watch], mg[watch]
        value = series.close[m2] * (1.0 - side) / sol_close[m2]
        pnl = value / entry_price[g2] - 1.0
        stop = pnl <= policy.stop_loss_pct
        take = ~stop & ~moonbag[g2] & (pnl >= policy.take_profit_pct)
        update = ~stop & ~take
        high = np.where(initialized[g2], np.maximum(highest[g2], value), value)
        highest[g2] = np.where(update, high, highest[g2])
        initialized[g2] |= update
        gain = highest[g2] / entry_price[g2] - 1.0
        # highest is 0 only where update is False, so the guard changes nothing.
        drawdown = np.divide(highest[g2] - value, highest[g2], out=np.zeros_like(value), where=highest[g2] > 0)
        trail = update & (gain > policy.trailing_activation) & (drawdown > policy.trailing_drop)
        timeout = update & ~trail & (t[m2] + CANDLE_SECONDS - fill_time[g2] >= policy.max_hold_seconds)
        full_exit = stop | trail | timeout
        pending[g2] = np.where(full_exit, _FULL, np.where(take, _PARTIAL, _NONE))
        pending_reason[g2] = np.select([stop, take, trail, timeout], [1, 2, 3, 4], 0)

    e_net = proceeds - 1.0 - txs * fee_fraction
    status[entries] = e_status
    net[entries] = np.where(np.isin(e_status, (1, 3)), e_net, np.nan)
    exit_time[entries] = e_exit
    reason[entries] = e_reason
    return {
        "decision_index": decisions,
        "status": status,
        "net": net,
        "exit_time": exit_time,
        "reason": reason,
    }


def label_grid(series, labels, shape):
    grid = np.full(shape, np.nan)
    d = labels["decision_index"]
    grid[series.tok[d], series.col[d]] = labels["net"]
    return grid


@dataclass
class _Position:
    token: int
    units: float
    entry_price: float          # SOL spent per token, as live entry_price
    entry_time: int
    decision_time: int
    score: float
    highest: float = 0.0
    initialized: bool = False
    moonbag: bool = False
    pending: int = _NONE
    pending_reason: int = 0
    pending_ready: int = 0
    proceeds: float = 0.0
    fees: float = 0.0
    exits: list = field(default_factory=list)


def _iso(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def simulate_portfolio(dataset, scores, settings, policy, cost, start, end, *,
                       exclude_tokens=(), tradable=None):
    """Event-driven SOL-accounted simulation over grid columns in [start, end).

    ``scores`` is the sigmoid score grid [tokens, time] (NaN if unavailable).
    Capital, slots (open positions + pending buys), cooldowns and partial
    exits are shared across all tokens as in the live runner.
    """
    times = dataset.times
    observed = dataset.observed.cpu().numpy()
    opens, closes = dataset.prices["open"], dataset.prices["close"]
    columns = np.nonzero((times >= start) & (times < end))[0]
    tradable_rows = np.ones(dataset.num_tokens, dtype=bool) if tradable is None else tradable.copy()
    for token in exclude_tokens:
        tradable_rows[token] = False
    side = cost.side_fraction
    fee = cost.network_fee_sol
    rng = np.random.default_rng(settings.seed)

    cash = settings.initial_sol
    positions = {}
    pending_buys = {}
    cooldowns = {}
    last_seen = {}
    counters = Counter()
    trades, decisions, equity_curve = [], [], []
    peak = settings.initial_sol
    max_drawdown = 0.0
    last_sol = None

    def sol_price(when):
        nonlocal last_sol
        price, age = dataset.sol_usd_asof(when)
        if price is None or age > settings.sol_price_max_age_seconds:
            return None
        last_sol = price
        return price

    def finish(pos, when, status):
        pnl = pos.proceeds - settings.entry_sol - pos.fees
        trades.append({
            "mint": dataset.addresses[pos.token],
            "token": pos.token,
            "decision_time": _iso(pos.decision_time),
            "entry_time": _iso(pos.entry_time),
            "exit_time": _iso(when),
            "score": pos.score,
            "exits": pos.exits,
            "status": status,
            "pnl_sol": pnl,
            "return": pnl / settings.entry_sol,
        })

    for c in columns:
        now = int(times[c])
        seen = observed[:, c]

        for token in list(pending_buys):
            order = pending_buys[token]
            if now < order["ready"]:
                continue
            if now > order["ready"] + settings.max_entry_wait_seconds:
                del pending_buys[token]
                counters["entry_unfilled_timeout"] += 1
                continue
            if not seen[token]:
                continue
            del pending_buys[token]
            sol = sol_price(now)
            if sol is None:
                counters["entry_unfilled_no_sol_price"] += 1
                continue
            if settings.route_failure_rate and rng.random() < settings.route_failure_rate:
                counters["entry_route_failure"] += 1
                continue
            if cash < settings.entry_sol + fee:
                counters["entry_unfilled_cash"] += 1
                continue
            units = settings.entry_sol * (1.0 - side) / (opens[token, c] / sol)
            cash -= settings.entry_sol + fee
            positions[token] = _Position(
                token=token, units=units, entry_price=settings.entry_sol / units,
                entry_time=now, decision_time=order["decision"], score=order["score"], fees=fee,
            )
            counters["entries"] += 1

        for token, pos in list(positions.items()):
            if not seen[token]:
                continue
            last_seen[token] = (now, closes[token, c])
            if pos.pending != _NONE and now >= pos.pending_ready:
                sol = sol_price(now)
                if sol is None:
                    counters["exit_deferred_no_sol_price"] += 1
                else:
                    if now - pos.pending_ready > settings.max_exit_wait_seconds:
                        counters["exit_delayed_fill"] += 1
                    full = pos.pending == _FULL
                    sold = pos.units if full else pos.units * policy.take_profit_ratio
                    received = sold * opens[token, c] * (1.0 - side) / sol
                    cash += received - fee
                    pos.units -= sold
                    pos.proceeds += received
                    pos.fees += fee
                    reason = REASONS[pos.pending_reason]
                    pos.exits.append({"time": _iso(now), "reason": reason, "fraction": 1.0 if full else policy.take_profit_ratio,
                                      "sol": received})
                    decisions.append((now, token, "fill:" + reason))
                    counters["exit_" + reason] += 1
                    pos.pending = _NONE
                    if full:
                        del positions[token]
                        if reason == "StopLoss":
                            cooldowns[token] = now + policy.stop_loss_cooldown_seconds
                        finish(pos, now, "closed")
                        continue
                    pos.moonbag = True
            if pos.pending != _NONE:
                continue
            close_time = now + CANDLE_SECONDS
            sol = sol_price(close_time)
            if sol is None:
                counters["monitor_skipped_no_sol_price"] += 1
                continue
            value = closes[token, c] * (1.0 - side) / sol
            pnl = value / pos.entry_price - 1.0
            code = 0
            if pnl <= policy.stop_loss_pct:
                code, kind = 1, _FULL
            elif not pos.moonbag and pnl >= policy.take_profit_pct:
                code, kind = 2, _PARTIAL
            else:
                pos.highest = max(pos.highest, value) if pos.initialized else value
                pos.initialized = True
                gain = pos.highest / pos.entry_price - 1.0
                drawdown = (pos.highest - value) / pos.highest
                if gain > policy.trailing_activation and drawdown > policy.trailing_drop:
                    code, kind = 3, _FULL
                elif close_time - pos.entry_time >= policy.max_hold_seconds:
                    code, kind = 4, _FULL
            if code:
                pos.pending, pos.pending_reason, pos.pending_ready = kind, code, close_time
                decisions.append((close_time, token, "exit:" + REASONS[code]))

        # Entry scan at this column's close.
        decision_time = now + CANDLE_SECONDS
        if len(positions) + len(pending_buys) < settings.max_positions:
            eligible = np.nonzero(seen & tradable_rows)[0]
            column_scores = scores[eligible, c]
            if eligible.size < settings.min_eligible:
                counters["scan_blocked_few_eligible"] += 1
            elif not np.isfinite(column_scores).all():
                counters["scan_blocked_nonfinite"] += 1
            elif column_scores.max() - column_scores.min() < settings.min_score_spread:
                counters["scan_blocked_no_spread"] += 1
            elif (column_scores >= settings.buy_threshold).sum() > eligible.size * settings.max_buy_fraction:
                counters["scan_blocked_buys_over_half"] += 1
            else:
                counters["scans"] += 1
                order = np.lexsort((eligible, -column_scores))
                for position in order:
                    score = float(column_scores[position])
                    if score < settings.buy_threshold:
                        break
                    token = int(eligible[position])
                    counters["signals"] += 1
                    if len(positions) + len(pending_buys) >= settings.max_positions:
                        counters["signals_skipped_slots"] += 1
                        continue
                    if token in positions or token in pending_buys:
                        continue
                    if cooldowns.get(token, -1) > decision_time:
                        counters["signals_skipped_cooldown"] += 1
                        continue
                    reserved = len(pending_buys) * (settings.entry_sol + fee)
                    if cash - reserved < settings.entry_sol + settings.balance_buffer_sol:
                        counters["signals_skipped_cash"] += 1
                        continue
                    pending_buys[token] = {
                        "decision": decision_time,
                        "ready": decision_time + settings.decision_latency_seconds,
                        "score": score,
                    }
                    counters["orders"] += 1
                    decisions.append((decision_time, token, "buy"))

        sol = sol_price(decision_time) or last_sol
        marked = cash
        if sol:
            for token, pos in positions.items():
                price = last_seen[token][1]  # set on the fill candle at the latest
                marked += pos.units * price * (1.0 - side) / sol
        equity_curve.append((decision_time, marked))
        peak = max(peak, marked)
        max_drawdown = max(max_drawdown, (peak - marked) / peak if peak > 0 else 0.0)

    counters["entry_unfilled_period_end"] += len(pending_buys)
    for token, pos in list(positions.items()):
        seen_time, price = last_seen.get(token, (None, None))
        sol, age = dataset.sol_usd_asof(end)
        stale = seen_time is None or end - (seen_time + CANDLE_SECONDS) > settings.stale_after_seconds
        if stale:
            counters["period_end_writeoff"] += 1
            pos.exits.append({"time": _iso(end), "reason": "PeriodEndWriteOff", "fraction": 1.0, "sol": 0.0})
            finish(pos, end, "written_off_stale")
        elif sol is None or age > settings.sol_price_max_age_seconds:
            # Same age rule as entries, monitors and exits: without a fresh
            # SOL/USD rate the position cannot be valued, so it is not counted
            # as a successful close. Conservatively valued at zero.
            counters["period_end_unvalued_no_sol_price"] += 1
            pos.exits.append({"time": _iso(end), "reason": "PeriodEndUnvalued", "fraction": 1.0, "sol": 0.0,
                              "sol_price_age_seconds": age})
            finish(pos, end, "unvalued_no_sol_price")
        else:
            received = pos.units * price * (1.0 - side) / sol
            cash += received - fee
            pos.proceeds += received
            pos.fees += fee
            pos.exits.append({"time": _iso(end), "reason": "PeriodEndClose", "fraction": 1.0, "sol": received})
            counters["period_end_close"] += 1
            finish(pos, end, "closed_at_period_end")
        del positions[token]
    final = cash
    peak = max(peak, final)
    max_drawdown = max(max_drawdown, (peak - final) / peak if peak > 0 else 0.0)
    equity_curve.append((int(end), final))
    return summarize(trades, counters, settings, final, max_drawdown, equity_curve, decisions)


def summarize(trades, counters, settings, final, max_drawdown, equity_curve, decisions):
    by_mint = defaultdict(float)
    by_day = defaultdict(float)
    for trade in trades:
        by_mint[trade["mint"]] += trade["pnl_sol"]
        by_day[trade["entry_time"][:10]] += trade["pnl_sol"]
    positive = sum(v for v in by_mint.values() if v > 0)
    top_mint = max(by_mint, key=by_mint.get) if by_mint else None
    orders = counters.get("orders", 0)
    returns = [t["return"] for t in trades]
    return {
        "initial_sol": settings.initial_sol,
        "final_sol": final,
        "net_return": final / settings.initial_sol - 1.0,
        "net_pnl_sol": final - settings.initial_sol,
        "max_drawdown": max_drawdown,
        "entries": counters.get("entries", 0),
        "distinct_mints": len(by_mint),
        "win_rate": (sum(r > 0 for r in returns) / len(returns)) if returns else None,
        "mean_trade_return": float(np.mean(returns)) if returns else None,
        "fill_rate": (counters.get("entries", 0) / orders) if orders else None,
        "top_mint": top_mint,
        "top_mint_pnl_sol": by_mint.get(top_mint) if top_mint else None,
        "top_mint_share_of_positive_pnl": (by_mint[top_mint] / positive) if top_mint and positive > 0 and by_mint[top_mint] > 0 else None,
        "pnl_by_day": dict(sorted(by_day.items())),
        "pnl_by_mint": dict(sorted(by_mint.items(), key=lambda item: -item[1])),
        "counters": dict(sorted(counters.items())),
        "trades": trades,
        "equity_curve": equity_curve[:: max(1, len(equity_curve) // 500)],
        "decisions": decisions,
    }
