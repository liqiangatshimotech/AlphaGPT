import unittest
import numpy as np
from model_core.holdout_execution import simulate


def raw(n=40, price=100.0):
    return {'open': np.full((1,n), price), 'close': np.full((1,n), price),
            'liquidity': np.full((1,n), 1_000_000.0),
            'observed': np.ones((1,n), dtype=bool)}


class HoldoutExecutionTests(unittest.TestCase):
    def test_constant_price_cost_is_counted_once_per_roundtrip(self):
        r=simulate(np.full((1,30),2.0),raw(30),0,30,1.0,hold_bars=5,cooldown=100)
        self.assertEqual(r['entries'],1)
        self.assertLess(r['net_pnl_fraction'],0)
        self.assertGreater(r['cost_fraction'],0)

    def test_hold_return_is_not_multiplied_by_horizon(self):
        x=raw(30); x['open'][0,16:]=110; x['close'][0,16:]=110
        r=simulate(np.full((1,30),2.0),x,0,30,1.0,hold_bars=15,cooldown=100)
        self.assertGreater(r['gross_pnl_fraction'],0.05)
        self.assertLess(r['gross_pnl_fraction'],0.2)

    def test_future_change_cannot_change_earlier_entry_event(self):
        a=raw(30); b=raw(30); b['open'][0,25:]=999
        s=np.full((1,30),-1.0); s[0,2]=2.0
        ra=simulate(s,a,0,20,1.0,hold_bars=5,cooldown=100)
        rb=simulate(s,b,0,20,1.0,hold_bars=5,cooldown=100)
        self.assertEqual(ra['entry_events'][0]['bar'],rb['entry_events'][0]['bar'])

    def test_missing_future_execution_is_unavailable(self):
        x=raw(30); x['observed'][0,8]=False
        s=np.full((1,30),-1.0); s[0,2]=2.0
        r=simulate(s,x,0,20,1.0,exit_threshold=-2.0,hold_bars=5,cooldown=100)
        self.assertFalse(r['available'])
        self.assertTrue(r['unresolved_missing'])

    def test_cooldown_and_minimum_hold_limit_entries(self):
        s=np.full((1,40),2.0)
        r=simulate(s,raw(40),0,40,1.0,hold_bars=5,cooldown=3)
        self.assertLessEqual(r['entries'],7)


if __name__=='__main__': unittest.main()
