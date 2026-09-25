"""Executable-rule labels and portfolio simulation for the v3 research path."""

from dataclasses import replace
from datetime import datetime
import unittest

import numpy as np

from model_core.v3_data import build_dataset, iso, synthetic_dataset
from model_core.v3_execution import (
    REASONS, CompactSeries, CostModel, ExecutionSettings, ExitPolicy, simulate_portfolio, trade_labels,
)

T0 = 1_758_000_000
FREE = CostModel("free", 0.0, 0.0, 0.0)


def market(paths, *, sol=100.0, minutes=None):
    """paths: {name: {minute: (open, close)}}; high/low bracket open/close."""
    rows = []
    for name, candles in paths.items():
        for minute, (open_, close) in candles.items():
            rows.append((T0 + 60 * minute, name, open_, max(open_, close), min(open_, close), close, 10.0))
    last = minutes or max(max(c) for c in paths.values()) + 1
    sol_rows = [(T0 + 60 * m, sol) for m in range(-5, last + 1)]
    return build_dataset(rows, sol_rows)


def flat(start, end, price=1.0, overrides=None):
    candles = {m: (price, price) for m in range(start, end)}
    candles.update(overrides or {})
    return candles


def at(minute):
    return iso(T0 + 60 * minute)


def epoch(stamp):
    return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())


def score_grid(dataset, signals):
    """signals: {(name, minute): score}; everything else observed scores 0.5."""
    scores = np.where(dataset.observed.numpy(), 0.5, np.nan)
    for (name, minute), value in signals.items():
        row = dataset.addresses.index(name)
        col = int(np.searchsorted(dataset.times, T0 + 60 * minute))
        assert dataset.times[col] == T0 + 60 * minute
        scores[row, col] = value
    return scores


def run(dataset, scores, settings=None, policy=None, cost=FREE, end_minute=None):
    end = T0 + 60 * end_minute if end_minute is not None else int(dataset.times[-1]) + 60
    return simulate_portfolio(dataset, scores, settings or ExecutionSettings(), policy or ExitPolicy(),
                              cost, int(dataset.times[0]), end)


class PortfolioRuleTests(unittest.TestCase):
    def test_stop_loss_fills_next_open_and_starts_cooldown(self):
        a = flat(0, 60, overrides={m: (0.92, 0.92) for m in range(12, 60)})
        a[11] = (1.0, 0.93)
        data = market({"A": a, "B": flat(0, 60), "C": flat(0, 60)})
        result = run(data, score_grid(data, {("A", 10): 0.9, ("A", 30): 0.95}))
        self.assertEqual(result["entries"], 1)
        trade = result["trades"][0]
        self.assertEqual(trade["entry_time"], at(11))   # open of the candle after the signal
        self.assertEqual(trade["exits"][0]["reason"], "StopLoss")
        self.assertEqual(trade["exits"][0]["time"], at(12))
        self.assertAlmostEqual(trade["return"], 0.92 - 1.0)
        self.assertEqual(result["counters"]["signals_skipped_cooldown"], 1)

    def test_partial_take_profit_then_trailing_stop(self):
        a = flat(0, 60, 1.0)
        a.update({11: (1.0, 1.12), 12: (1.12, 1.20), 13: (1.20, 1.15)})
        for m in range(14, 60):
            a[m] = (1.15, 1.15)
        data = market({"A": a, "B": flat(0, 60), "C": flat(0, 60)})
        result = run(data, score_grid(data, {("A", 10): 0.9}))
        exits = result["trades"][0]["exits"]
        self.assertEqual([e["reason"] for e in exits], ["Moonbag", "TrailingStop"])
        self.assertEqual(exits[0]["fraction"], 0.5)
        # Half sold at 1.12 (open after TP close), rest at 1.15 after the 1.20 high.
        self.assertAlmostEqual(result["trades"][0]["pnl_sol"], 0.5 * 1.12 + 0.5 * 1.15 - 1.0)

    def test_time_exit_and_gap_uses_timestamps_not_columns(self):
        # A is missing minutes 11-14: the fill must wait for minute 15 and the
        # 120 s entry wait is exceeded, so the order is unfilled.
        a = flat(0, 60)
        for m in range(11, 15):
            del a[m]
        data = market({"A": a, "B": flat(0, 60), "C": flat(0, 60)})
        result = run(data, score_grid(data, {("A", 10): 0.9}))
        self.assertEqual(result["entries"], 0)
        self.assertEqual(result["counters"]["entry_unfilled_timeout"], 1)

        data = market({"A": flat(0, 400), "B": flat(0, 400), "C": flat(0, 400)})
        result = run(data, score_grid(data, {("A", 10): 0.9}), policy=ExitPolicy(max_hold_seconds=3600))
        exit_ = result["trades"][0]["exits"][0]
        self.assertEqual(exit_["reason"], "TimeExit")
        self.assertEqual(exit_["time"], at(71))   # 60 min after the minute-11 fill

    def test_decision_latency_delays_fill(self):
        data = market({"A": flat(0, 60), "B": flat(0, 60), "C": flat(0, 60)})
        settings = ExecutionSettings(decision_latency_seconds=900)
        result = run(data, score_grid(data, {("A", 10): 0.9}), settings=settings)
        self.assertEqual(result["trades"][0]["entry_time"], at(26))

    def test_capacity_cash_and_quality_gate(self):
        names = [f"T{i}" for i in range(12)]
        data = market({name: flat(0, 60) for name in names})
        signals = {(name, 10): 0.9 + i / 1000 for i, name in enumerate(names[:6])}
        result = run(data, score_grid(data, signals))
        self.assertEqual(result["entries"], 5)
        self.assertEqual(result["counters"]["signals_skipped_slots"], 1)
        # Highest scores are bought first.
        self.assertNotIn("T0", {t["mint"] for t in result["trades"]})

        result = run(data, score_grid(data, signals), settings=ExecutionSettings(initial_sol=2.5))
        self.assertEqual(result["entries"], 2)
        self.assertGreaterEqual(result["counters"]["signals_skipped_cash"], 1)

        everything = {(name, 10): 0.9 + i / 1000 for i, name in enumerate(names[:7])}
        result = run(data, score_grid(data, everything))
        self.assertEqual(result["entries"], 0)
        self.assertEqual(result["counters"]["scan_blocked_buys_over_half"], 1)

    def test_unobserved_and_prelisting_cells_never_trade(self):
        data = market({"A": flat(30, 60), "B": flat(0, 60), "C": flat(0, 60)})
        scores = score_grid(data, {})
        row = data.addresses.index("A")
        scores[row, :30] = 0.99   # would-be signal on pre-listing (unobserved) cells
        result = run(data, scores)
        self.assertEqual(result["entries"], 0)

    def test_period_end_stale_position_is_written_off_not_sold(self):
        data = market({"A": flat(0, 20), "B": flat(0, 120), "C": flat(0, 120)})
        result = run(data, score_grid(data, {("A", 10): 0.9}), policy=ExitPolicy(max_hold_seconds=10**9))
        trade = result["trades"][0]
        self.assertEqual(trade["status"], "written_off_stale")
        self.assertAlmostEqual(trade["pnl_sol"], -1.0)

        data = market({"A": flat(0, 120), "B": flat(0, 120), "C": flat(0, 120)})
        result = run(data, score_grid(data, {("A", 10): 0.9}), policy=ExitPolicy(max_hold_seconds=10**9))
        self.assertEqual(result["trades"][0]["status"], "closed_at_period_end")
        self.assertAlmostEqual(result["final_sol"], ExecutionSettings().initial_sol)

    def test_sol_accounting_uses_asof_rate(self):
        data = market({"A": flat(0, 60), "B": flat(0, 60), "C": flat(0, 60)})
        # SOL doubles after entry: a flat USD token loses half its SOL value.
        data.sol_close = np.where(data.sol_times >= T0 + 60 * 20, 200.0, 100.0)
        result = run(data, score_grid(data, {("A", 10): 0.9}))
        self.assertEqual(result["trades"][0]["exits"][0]["reason"], "StopLoss")
        self.assertLess(result["trades"][0]["return"], -0.4)

    def test_period_end_close_refuses_stale_sol_rate(self):
        data = market({"A": flat(0, 120), "B": flat(0, 120), "C": flat(0, 120)})
        keep = data.sol_times < T0 + 60 * 20          # SOL/USD stops updating at minute 20
        data.sol_times, data.sol_close = data.sol_times[keep], data.sol_close[keep]
        result = run(data, score_grid(data, {("A", 10): 0.9}), policy=ExitPolicy(max_hold_seconds=10**9))
        trade = result["trades"][0]
        self.assertGreater(result["counters"]["monitor_skipped_no_sol_price"], 50)
        self.assertEqual(trade["status"], "unvalued_no_sol_price")
        self.assertEqual(trade["exits"][-1]["reason"], "PeriodEndUnvalued")
        self.assertNotIn("period_end_close", result["counters"])
        self.assertAlmostEqual(trade["pnl_sol"], -1.0)

    def test_costs_are_monotonic(self):
        data = synthetic_dataset(tokens=10, minutes=1500, seed=3)
        rng = np.random.default_rng(0)
        scores = np.where(data.observed.numpy(), rng.uniform(0.3, 0.9, data.observed.shape), np.nan)
        # Time exits only, so every cost level trades the same path.
        policy = ExitPolicy(stop_loss_pct=-10.0, take_profit_pct=1e9, trailing_activation=1e9, max_hold_seconds=1800)
        settings = ExecutionSettings(max_buy_fraction=1.0, initial_sol=100.0)
        previous = None
        for bps in (0, 20, 60, 150):
            cost = CostModel(str(bps), bps, 0.0, 0.0001)
            result = run(data, scores, cost=cost, settings=settings, policy=policy)
            self.assertGreater(result["entries"], 20)
            if previous is not None:
                self.assertLess(result["final_sol"], previous)
            previous = result["final_sol"]


def single_trade(data, name, minute, policy, cost=FREE, settings=None):
    """(label, simulated trade) for one signal on a token, no capacity limits."""
    settings = settings or ExecutionSettings(max_buy_fraction=1.0, min_score_spread=0.0)
    start, end = int(data.times[0]), int(data.times[-1]) + 60
    result = simulate_portfolio(data, score_grid(data, {(name, minute): 0.9}), settings, policy, cost, start, end)
    series = CompactSeries(data)
    labels = trade_labels(series, policy, cost, settings, start, end)
    row = data.addresses.index(name)
    index = series.index[row, int(np.searchsorted(data.times, T0 + 60 * minute))]
    position = int(np.nonzero(labels["decision_index"] == index)[0][0])
    label = {key: labels[key][position] for key in ("status", "net", "exit_time", "reason")}
    return label, result


class LabelConsistencyTests(unittest.TestCase):
    def test_labels_use_sol_accounting_like_the_portfolio(self):
        # Review repro: flat USD token, SOL 100 -> 106 USD after entry. In SOL
        # the position loses 5.66% and stops out; USD labels saw 0% / TimeExit.
        policy = ExitPolicy(max_hold_seconds=3600)
        paths = {"A": flat(0, 200), "B": flat(0, 200), "C": flat(0, 200)}
        cases = {
            "sol_up": lambda m: 106.0 if m >= 20 else 100.0,
            "sol_down": lambda m: 94.0 if m >= 20 else 100.0,
            "sol_flat": lambda m: 100.0,
        }
        for name, curve in cases.items():
            with self.subTest(case=name):
                data = market(paths)
                data.sol_close = np.array([curve((t - T0) // 60) for t in data.sol_times], dtype=float)
                label, result = single_trade(data, "A", 10, policy)
                trade = result["trades"][0]
                self.assertEqual(label["status"], 1)
                self.assertAlmostEqual(float(label["net"]), trade["return"], places=12)
                self.assertEqual(int(label["exit_time"]), epoch(trade["exit_time"]))
                self.assertEqual(REASONS[int(label["reason"])], trade["exits"][-1]["reason"])
        data = market(paths)
        data.sol_close = np.where(data.sol_times >= T0 + 60 * 20, 106.0, 100.0)
        label, result = single_trade(data, "A", 10, policy)
        self.assertEqual(REASONS[int(label["reason"])], "StopLoss")
        self.assertEqual(int(label["exit_time"]), T0 + 60 * 21)
        self.assertAlmostEqual(float(label["net"]), 100.0 / 106.0 - 1.0, places=12)

    def test_labels_follow_missing_sol_rates_like_the_portfolio(self):
        policy = ExitPolicy(max_hold_seconds=3600)
        paths = {"A": flat(0, 300, overrides={m: (0.9, 0.9) for m in range(40, 300)}),
                 "B": flat(0, 300), "C": flat(0, 300)}
        settings = ExecutionSettings(max_buy_fraction=1.0, min_score_spread=0.0, sol_price_max_age_seconds=300)
        # (a) no SOL rate at the fill: unfilled in both.
        data = market(paths)
        keep = (data.sol_times < T0 + 60 * 3) | (data.sol_times >= T0 + 60 * 30)
        data.sol_times, data.sol_close = data.sol_times[keep], data.sol_close[keep]
        label, result = single_trade(data, "A", 10, policy, settings=settings)
        self.assertEqual(result["entries"], 0)
        self.assertEqual(result["counters"]["entry_unfilled_no_sol_price"], 1)
        self.assertEqual(label["status"], 0)
        # (b) SOL gap while the stop-loss would trigger: monitors skipped and
        # the exit happens later, at the same time and value in both.
        data = market(paths)
        keep = (data.sol_times < T0 + 60 * 30) | (data.sol_times >= T0 + 60 * 60)
        data.sol_times, data.sol_close = data.sol_times[keep], data.sol_close[keep]
        label, result = single_trade(data, "A", 10, policy, settings=settings)
        trade = result["trades"][0]
        self.assertGreater(result["counters"]["monitor_skipped_no_sol_price"], 0)
        self.assertEqual(label["status"], 1)
        self.assertEqual(int(label["exit_time"]), epoch(trade["exit_time"]))
        self.assertAlmostEqual(float(label["net"]), trade["return"], places=12)
        # (c) Stop triggers at minute 40's close with a fresh rate; the token
        # skips 41-44 and SOL skips 41-47, so the exit waits until minute 49.
        a = dict(paths["A"])
        for m in range(41, 45):
            del a[m]
        data = market({**paths, "A": a})
        keep = (data.sol_times < T0 + 60 * 41) | (data.sol_times >= T0 + 60 * 48)
        data.sol_times, data.sol_close = data.sol_times[keep], data.sol_close[keep]
        settings_c = replace(settings, sol_price_max_age_seconds=120)
        label, result = single_trade(data, "A", 10, policy, settings=settings_c)
        trade = result["trades"][0]
        self.assertGreaterEqual(result["counters"]["exit_deferred_no_sol_price"], 1)
        self.assertEqual(trade["exit_time"], at(49))
        self.assertEqual(int(label["exit_time"]), T0 + 60 * 49)
        self.assertAlmostEqual(float(label["net"]), trade["return"], places=12)

    def test_vectorised_labels_match_portfolio_trades(self):
        for sol_mode in ("pinned", "moving", "gappy"):
            with self.subTest(sol=sol_mode):
                self._compare_with_portfolio(sol_mode)

    def _compare_with_portfolio(self, sol_mode):
        data = synthetic_dataset(tokens=8, minutes=3000, seed=5, gap_probability=0.05)
        if sol_mode == "pinned":
            data.sol_close = np.full_like(data.sol_close, 150.0)
        elif sol_mode == "gappy":
            rng = np.random.default_rng(9)
            keep = np.ones(len(data.sol_times), dtype=bool)
            for begin in rng.integers(0, len(keep) - 60, 15):
                keep[begin: begin + int(rng.integers(5, 60))] = False
            data.sol_times, data.sol_close = data.sol_times[keep], data.sol_close[keep]
        cost = CostModel("c", 30.0, 10.0, 0.0005)
        settings = ExecutionSettings(max_positions=100, initial_sol=1000.0, max_buy_fraction=1.0,
                                     sol_price_max_age_seconds=1800 if sol_mode != "gappy" else 600)
        policy = ExitPolicy(max_hold_seconds=3 * 3600, stop_loss_cooldown_seconds=0)
        rng = np.random.default_rng(1)
        observed = data.observed.numpy()
        scores = np.where(observed, 0.5 + 0.001 * rng.random(observed.shape), np.nan)
        scores[observed & (rng.random(observed.shape) < 0.01)] = 0.9
        start, end = int(data.times[0]), int(data.times[-1]) + 60
        result = simulate_portfolio(data, scores, settings, policy, cost, start, end)
        series = CompactSeries(data)
        labels = trade_labels(series, policy, cost, settings, start, end)
        by_decision = {}
        for index, net, status in zip(labels["decision_index"], labels["net"], labels["status"]):
            by_decision[(int(series.tok[index]), int(series.t[index]) + 60)] = (net, status)
        compared = 0
        for trade in result["trades"]:
            if trade["status"] in ("closed_at_period_end", "unvalued_no_sol_price"):
                continue
            decision = epoch(trade["decision_time"])
            net, status = by_decision[(trade["token"], decision)]
            self.assertIn(status, (1, 3))
            self.assertAlmostEqual(net, trade["return"], places=9)
            compared += 1
        self.assertGreater(compared, 50)

    def test_labels_purge_outcomes_crossing_split_end(self):
        data = synthetic_dataset(tokens=6, minutes=2000, seed=2)
        series = CompactSeries(data)
        policy = ExitPolicy(max_hold_seconds=3600)
        split_end = int(data.times[0]) + 1000 * 60
        labels = trade_labels(series, policy, FREE, ExecutionSettings(), int(data.times[0]), split_end)
        resolved = labels["status"] == 1
        self.assertTrue(resolved.any())
        self.assertTrue((labels["exit_time"][resolved] < split_end).all())
        self.assertTrue(np.isnan(labels["net"][labels["status"] == 2]).all())
        # Data after the split end cannot change any label.
        altered = synthetic_dataset(tokens=6, minutes=2000, seed=2)
        late = altered.times >= split_end
        altered.prices["open"][:, late] *= 3.0
        altered.prices["close"][:, late] *= 0.2
        again = trade_labels(CompactSeries(altered), policy, FREE, ExecutionSettings(), int(data.times[0]), split_end)
        np.testing.assert_array_equal(again["status"], labels["status"])
        np.testing.assert_allclose(again["net"], labels["net"], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
