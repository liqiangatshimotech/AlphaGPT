import unittest
import numpy as np
from model_core.holdout_execution import simulate


def raw(n=40, price=100.0):
    return {'open': np.full((1, n), price), 'close': np.full((1, n), price),
            'liquidity': np.full((1, n), 1_000_000.0),
            'observed': np.ones((1, n), dtype=bool)}


def events(result, kind):
    return [event for event in result['entry_events'] if event['type'] == kind]


class HoldoutExecutionTests(unittest.TestCase):
    def test_flat_market_accounting_and_exposure(self):
        result = simulate(np.full((1, 30), 2.0), raw(30), 0, 30, 1.0,
                          hold_bars=5, cooldown=100)
        self.assertEqual(result['entries'], 1)
        self.assertAlmostEqual(result['gross_pnl_fraction'], 0)
        self.assertAlmostEqual(result['net_pnl_fraction'], -result['cost_fraction'])
        self.assertAlmostEqual(sum(result['per_token_pnl_fraction']), result['net_pnl_fraction'])
        self.assertAlmostEqual(result['equity_path'][-1], 1 + result['net_pnl_fraction'])
        self.assertAlmostEqual(result['max_drawdown_fraction'], -result['net_pnl_fraction'])
        self.assertAlmostEqual(result['exposure'], 5 / 29)

    def test_hold_return_and_exact_exit_open(self):
        x = raw(30)
        x['open'][0, 16:] = 110
        x['close'][0, 16:] = 110
        result = simulate(np.full((1, 30), 2.0), x, 0, 30, 1.0,
                          hold_bars=15, cooldown=100)
        self.assertEqual(events(result, 'entry')[0]['bar'], 1)
        self.assertEqual(events(result, 'exit')[0]['bar'], 16)
        self.assertEqual(events(result, 'exit')[0]['held_bars'], 15)
        self.assertGreater(result['gross_pnl_fraction'], 0.09)
        self.assertLess(result['gross_pnl_fraction'], 0.101)
        self.assertAlmostEqual(result['gross_pnl_fraction'] - result['cost_fraction'],
                               result['net_pnl_fraction'])

    def test_future_change_cannot_change_earlier_events(self):
        a, b = raw(30), raw(30)
        b['open'][0, 25:] = 999
        scores = np.full((1, 30), -1.0)
        scores[0, 2] = 2.0
        first = simulate(scores, a, 0, 20, 1.0, hold_bars=5, cooldown=100)
        second = simulate(scores, b, 0, 20, 1.0, hold_bars=5, cooldown=100)
        self.assertEqual(first['entry_events'], second['entry_events'])

    def test_open_fill_does_not_read_whole_future_observed_mask(self):
        x = raw(8)
        x['observed'][0, 1] = False
        x['open_available'] = np.ones((1, 8), dtype=bool)
        result = simulate(np.full((1, 8), 2.0), x, 0, 8, 1.0, cooldown=100)
        self.assertEqual(events(result, 'entry')[0]['bar'], 1)
        self.assertEqual(events(result, 'exit')[0]['bar'], 2)
        self.assertEqual(events(result, 'exit')[0]['reason'], 'missing_signal_data')
        self.assertTrue(result['available'])

    def test_eligibility_is_separate_from_data_quality(self):
        x = raw(10)
        quality = x['observed'].copy()
        eligible = np.ones((1, 10), dtype=bool)
        eligible[:, 1:] = False
        result = simulate(np.full((1, 10), 2.0), x, 0, 10, 1.0, eligible=eligible,
                          hold_bars=5, min_hold_bars=3, cooldown=100)
        self.assertEqual(events(result, 'exit')[0]['bar'], 4)
        self.assertEqual(events(result, 'exit')[0]['reason'], 'signal')
        np.testing.assert_array_equal(x['observed'], quality)

    def test_execution_liquidity_is_used_at_fill(self):
        x = raw(8)
        x['execution_liquidity'] = np.full((1, 8), 1_000_000.0)
        x['liquidity'][0, 1] = 10.0  # Completed fill-bar state must not price its open.
        result = simulate(np.full((1, 8), 2.0), x, 0, 8, 1.0, cooldown=100)
        self.assertEqual(result['entries'], 1)
        self.assertLess(events(result, 'entry')[0]['impact'], 0.002)

    def test_missing_entry_fill_is_unavailable_without_losing_cash(self):
        x = raw(8)
        x['open_available'] = np.ones((1, 8), dtype=bool)
        x['open_available'][0, 1] = False
        scores = np.full((1, 8), -1.0)
        scores[0, 0] = 2.0
        result = simulate(scores, x, 0, 8, 1.0)
        self.assertFalse(result['available'])
        self.assertEqual(result['entries'], 0)
        self.assertEqual(result['net_pnl_fraction'], 0)
        self.assertEqual(result['unresolved_missing'][0]['reason'], 'missing_entry_open')

    def test_unresolved_final_position_has_unknown_pnl(self):
        x = raw(7)
        x['open_available'] = np.ones((1, 7), dtype=bool)
        x['open_available'][0, 6] = False
        result = simulate(np.full((1, 7), 2.0), x, 0, 7, 1.0, hold_bars=5)
        self.assertFalse(result['available'])
        self.assertIsNone(result['net_pnl_fraction'])
        self.assertIsNone(result['gross_pnl_fraction'])
        self.assertIsNone(result['max_drawdown_fraction'])
        self.assertIsNone(result['equity_path'][-1])
        self.assertEqual(result['open_position_token'], 0)

    def test_missing_intermediate_mark_does_not_become_zero_equity(self):
        x = raw(8)
        x['open_available'] = np.ones((1, 8), dtype=bool)
        x['open_available'][0, 3] = False
        result = simulate(np.full((1, 8), 2.0), x, 0, 8, 1.0, hold_bars=5, cooldown=100)
        self.assertIsNone(result['equity_path'][3])
        self.assertIsNone(result['max_drawdown_fraction'])
        self.assertFalse(result['available'])
        self.assertIsNone(result['net_pnl_fraction'])

    def test_max_duration_cooldown_and_no_same_decision_reentry(self):
        result = simulate(np.full((1, 25), 2.0), raw(25), 0, 25, 1.0,
                          hold_bars=5, cooldown=3)
        self.assertEqual([e['bar'] for e in events(result, 'entry')], [1, 10, 19])
        self.assertEqual([e['bar'] for e in events(result, 'exit')], [6, 15, 24])
        no_cooldown = simulate(np.full((1, 9), 2.0), raw(9), 0, 9, 1.0,
                               hold_bars=1, cooldown=0)
        self.assertEqual([e['bar'] for e in events(no_cooldown, 'entry')], [1, 3, 5, 7])
        self.assertEqual([e['bar'] for e in events(no_cooldown, 'exit')], [2, 4, 6, 8])

    def test_stop_loss_overrides_minimum_hold(self):
        x = raw(10)
        x['close'][0, 1] = 80
        x['open'][0, 2:] = 75
        result = simulate(np.full((1, 10), 2.0), x, 0, 10, 1.0,
                          hold_bars=5, min_hold_bars=3, stop_loss=0.1, cooldown=100)
        self.assertEqual(events(result, 'exit')[0]['bar'], 2)
        self.assertEqual(events(result, 'exit')[0]['price'], 75)
        self.assertEqual(events(result, 'exit')[0]['reason'], 'stop_loss')

    def test_liquidity_drop_and_minimum_override_minimum_hold(self):
        for settings, reason in [({'liquidity_drop_fraction': 0.5}, 'liquidity_drop'),
                                 ({'min_liquidity': 500_000}, 'liquidity')]:
            x = raw(10)
            x['liquidity'][0, 1:] = 400_000
            x['execution_liquidity'] = np.full((1, 10), 1_000_000.0)
            result = simulate(np.full((1, 10), 2.0), x, 0, 10, 1.0,
                              hold_bars=5, min_hold_bars=3, cooldown=100, **settings)
            self.assertEqual(events(result, 'exit')[0]['bar'], 2)
            self.assertEqual(events(result, 'exit')[0]['reason'], reason)

    def test_impact_limit_rejects_entry_and_does_not_cap_exit_cost(self):
        x = raw(5)
        x['execution_liquidity'] = np.full((1, 5), 10_000.0)
        rejected = simulate(np.full((1, 5), 2.0), x, 0, 5, 1.0)
        self.assertEqual(rejected['entries'], 0)
        self.assertTrue(rejected['entry_rejections'])
        self.assertEqual(rejected['net_pnl_fraction'], 0)
        x['execution_liquidity'][0, 1] = 1e12
        result = simulate(np.full((1, 5), 2.0), x, 0, 5, 1.0, hold_bars=1, cooldown=100, fee=0)
        self.assertFalse(result['available'])
        self.assertTrue(result['risk_events'])
        self.assertGreater(result['cost_fraction'], 0.09)
        self.assertAlmostEqual(result['gross_pnl_fraction'] - result['cost_fraction'],
                               result['net_pnl_fraction'])

    def test_terminal_liquidation_is_in_equity_and_drawdown(self):
        result = simulate(np.full((1, 4), 2.0), raw(4), 0, 4, 1.0,
                          hold_bars=20, cooldown=100)
        self.assertEqual(events(result, 'exit')[0]['reason'], 'segment_end')
        self.assertAlmostEqual(result['equity_path'][-1], 1 + result['net_pnl_fraction'])
        self.assertAlmostEqual(result['max_drawdown_fraction'], -result['net_pnl_fraction'])

    def test_drawdown_is_relative_to_equity_peak(self):
        x = raw(5)
        x['open'][0] = [100, 100, 200, 150, 150]
        x['close'][0] = x['open'][0]
        x['liquidity'][:] = 1e15
        result = simulate(np.full((1, 5), 2.0), x, 0, 5, 1.0, hold_bars=3, fee=0)
        curve = np.array(result['equity_path'])
        expected = np.max((np.maximum.accumulate(curve) - curve) / np.maximum.accumulate(curve))
        self.assertAlmostEqual(result['max_drawdown_fraction'], expected)
        self.assertAlmostEqual(result['max_drawdown_fraction'], 0.25, places=7)

    def test_no_entry_at_final_open(self):
        scores = np.full((1, 5), -1.0)
        scores[0, 3] = 2.0
        result = simulate(scores, raw(5), 0, 5, 1.0)
        self.assertEqual(result['entries'], 0)
        self.assertEqual(result['exposure'], 0)
        self.assertEqual(result['net_pnl_fraction'], 0)

    def test_invalid_parameters_and_masks_fail(self):
        with self.assertRaises(ValueError):
            simulate(np.ones((1, 5)), raw(5), 0, 5, 1.0, hold_bars=2, min_hold_bars=3)
        with self.assertRaises(ValueError):
            simulate(np.ones((1, 5)), raw(5), 0, 5, 1.0, eligible=np.ones((2, 5)))


if __name__ == '__main__':
    unittest.main()
