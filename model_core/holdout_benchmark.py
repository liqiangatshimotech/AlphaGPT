"""Pre-registered baseline benchmark for address holdout and horizons."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .data_loader import CryptoDataLoader
from .research_train import causal_features
from .holdout_execution import simulate
from .vm import StackVM


def address_split(addresses):
    train, test = [], []
    for i, address in enumerate(addresses):
        bucket = int(hashlib.sha256(str(address).encode()).hexdigest()[:8], 16) % 5
        (train if bucket < 3 else test).append(i)
    return train, test


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--timeframe',choices=['1m','15m'],default='1m'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=False,exist_ok=False)
    from data_pipeline.config import Config
    Config.TIMEFRAME=args.timeframe
    loader=CryptoDataLoader(); loader.load_data(); raw=loader.raw_data_cache; feat=causal_features(raw)
    vm=StackVM(); formulas={'NEG_RET':[0,10], 'NEG_PRESSURE':[2,10], 'NEG_RET_PRESSURE':[0,2,6,10], 'RET':[0]}
    train_idx,test_idx=address_split(loader.addresses)
    arr={k:v.detach().cpu().numpy() for k,v in raw.items() if torch.is_tensor(v)}
    feat_np=feat.detach().cpu(); results=[]
    for name,seq in formulas.items():
        score=vm.execute(seq,feat_np).detach().cpu().numpy()
        for threshold_q in (0.80,0.90):
            vals=score[train_idx]
            signal_valid = arr['observed'][train_idx] & (np.abs(vals) > 1e-8)
            eligible=signal_valid & (arr['liquidity'][train_idx]>500000) & np.isfinite(vals)
            threshold=float(np.quantile(vals[eligible],threshold_q)) if eligible.any() else 0.0
            for horizon in (5,15,30):
                common=dict(entry_threshold=threshold,exit_threshold=-np.inf,hold_bars=horizon,cooldown=3,notional=1000.,fee=.006)
                tr={k:v[train_idx] for k,v in arr.items()}; te={k:v[test_idx] for k,v in arr.items()}
                tr['observed'] = tr['observed'] & (np.abs(score[train_idx]) > 1e-8)
                te['observed'] = te['observed'] & (np.abs(score[test_idx]) > 1e-8)
                train=simulate(score[train_idx],tr,20,score.shape[1]-horizon,**common)
                test=simulate(score[test_idx],te,20,score.shape[1]-horizon,**common)
                results.append({'formula':name,'tokens_train':len(train_idx),'tokens_test':len(test_idx),'threshold_q':threshold_q,'threshold':threshold,'horizon':horizon,'train':train,'address_holdout':test})
    report={'seed':42,'timeframe':args.timeframe,'bars':int(score.shape[1]),'addresses':len(loader.addresses),'train_indices':train_idx,'test_indices':test_idx,'formulas':formulas,'horizons':[5,15,30],'threshold_quantiles':[.80,.90],'results':results,'limitations':['Historical address split is a robustness diagnostic, not a fresh blind test: this database and all 41 addresses were previously inspected.','Address universe is still selected by all-history candle count in CryptoDataLoader.','No formula/threshold was fitted on held-out addresses; threshold is frozen from train addresses.','15m mode samples aligned 15-minute records; it is not a full OHLC aggregation. Use a production resampler before deployment.','Use a post-freeze forward data snapshot before deployment.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
