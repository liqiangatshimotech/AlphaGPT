import unittest
import torch
from model_core.research_train import legal_mask, causal_features, metrics
from model_core.vocab import load_formula, FORMULA_VOCAB as V
from model_core.ops import OPS_CONFIG

class ResearchTests(unittest.TestCase):
    def test_generated_formulas(self):
        torch.manual_seed(1)
        d=torch.zeros(1000,dtype=torch.long); seq=[]
        arity=torch.tensor([0]*V.feature_count+[x[2] for x in OPS_CONFIG])
        for t in range(12):
            a=torch.multinomial(legal_mask(d,11-t).float(),1).squeeze(1); seq.append(a); d+=1-arity[a]
        for row in torch.stack(seq,1).tolist(): load_formula(row)
        self.assertTrue(torch.all(d==1))
    def test_prefix_invariance(self):
        torch.manual_seed(2)
        raw={k:torch.rand(2,100)+1 for k in ['close','open','high','low','volume','liquidity','fdv']}; raw['high']=raw['low']+1; raw['observed']=torch.ones(2,100,dtype=torch.bool)
        full=causal_features(raw); prefix=causal_features({k:v[:,:60] for k,v in raw.items()})
        torch.testing.assert_close(full[:,:,:60],prefix)
    def test_costs_and_purge(self):
        raw={'liquidity':torch.full((1,10),1e6),'tradable':torch.ones(1,10,dtype=torch.bool)}
        t=torch.zeros(1,10); t[:,-2:]=100
        m=metrics(torch.full((1,10),5.),raw,t,0,10)
        self.assertAlmostEqual(m['net_pnl_per_initial_notional'],-.014,places=6)
        self.assertAlmostEqual(m['drawdown_additive'],.014,places=6)

if __name__=='__main__': unittest.main()
