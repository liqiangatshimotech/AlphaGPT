import unittest
import numpy as np
from model_core.short_cycle import event_features, forward_labels


class ShortCycleTests(unittest.TestCase):
    def test_event_features_are_short_and_gap_aware(self):
        raw={k:np.ones((1,12),dtype=float) for k in ('open','high','low','close','volume','liquidity','fdv')}
        raw['high'] += .1; raw['observed']=np.ones((1,12),dtype=bool); raw['observed'][0,5]=False
        raw['close'][0,6]=2
        f=event_features(raw)
        self.assertTrue(np.isnan(f['ret1'][0,5]))
        self.assertTrue(np.isnan(f['ret1'][0,6]))
        self.assertTrue(np.isfinite(f['ret1'][0,4]))

    def test_forward_label_is_one_horizon_return_not_repeated_pnl(self):
        raw={k:np.ones((1,20),dtype=float) for k in ('open','high','low','close','volume','liquidity','fdv')}
        raw['observed']=np.ones((1,20),dtype=bool); raw['open'][0,6:]=1.1
        y, mask=forward_labels(raw,5)
        self.assertTrue(mask[0,0]); self.assertAlmostEqual(y[0,0],.1,places=6)
        self.assertFalse(mask[0,-1]); self.assertTrue(np.isnan(y[0,-1]))


if __name__=='__main__': unittest.main()
