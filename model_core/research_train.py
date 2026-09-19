"""Isolated causal formula-search experiment; never overwrites live strategy."""
import argparse
import json
from pathlib import Path
import torch
from torch.distributions import Categorical
from .alphagpt import AlphaGPT
from .data_loader import CryptoDataLoader
from .ops import OPS_CONFIG
from .vm import StackVM
from .vocab import FORMULA_VOCAB as V, FORMULA_VOCAB_VERSION, load_formula


def legal_mask(depth, remaining):
    arities = torch.tensor([0]*V.feature_count + [x[2] for x in OPS_CONFIG], device=depth.device)
    nxt = depth[:, None] + 1 - arities
    allowed = (depth[:, None] >= arities) & (nxt >= 1) & (nxt <= 1 + 2*remaining)
    allowed[:, V.token_names.index('JUMP')] = False  # Uses future time statistics.
    return allowed


def causal_features(raw):
    c, v = raw['close'], raw['volume']
    observed = raw['observed']
    prev_obs = torch.cat([torch.zeros_like(observed[:, :1]), observed[:, :-1]], 1)
    prev2_obs = torch.cat([torch.zeros_like(observed[:, :2]), observed[:, :-2]], 1)
    contiguous1 = observed & prev_obs
    contiguous2 = contiguous1 & prev2_obs
    prev = torch.cat([c[:, :1], c[:, :-1]], 1)
    ret = torch.where(contiguous1, torch.log(c.clamp_min(1e-12)/prev.clamp_min(1e-12)), torch.zeros_like(c))
    vp = torch.cat([v[:, :1],v[:, :-1]],1)
    growth = torch.where(contiguous1, ((v-vp)/(vp+1)).clamp(-5,5), torch.zeros_like(v))
    growth_prev = torch.cat([growth[:, :1], growth[:, :-1]], 1)
    fomo = torch.where(contiguous2, growth-growth_prev, torch.zeros_like(growth))
    # Rolling statistics use only the current contiguous run.  A late-listed
    # token therefore has no DEV signal until it has 20 real observations.
    run = torch.zeros_like(observed, dtype=torch.long)
    for t in range(observed.shape[1]):
        run[:, t] = observed[:, t] * (run[:, t-1] + 1 if t else 1)
    cs = torch.cat([torch.zeros_like(c[:, :1]), c.cumsum(1)], 1)
    sum20 = cs[:, 20:] - cs[:, :-20]
    ma = torch.cat([torch.zeros_like(c[:, :19]), sum20 / 20.0], 1)
    ma = torch.where(run >= 20, ma, torch.zeros_like(ma))
    channels = [ret, (raw['liquidity']/(raw['fdv']+1e-6)*4).clamp(0,1),
                torch.tanh((c-raw['open'])/(raw['high']-raw['low']+1e-9)*3),
                fomo, (c-ma)/(ma+1e-9), torch.log1p(v)]
    out=[]
    valid=raw['observed']; count=valid.cumsum(1).clamp_min(1)
    for i,x in enumerate(channels):
        x=torch.where(valid,x,0)
        if i in (0,3,4,5):
            mean=x.cumsum(1)/count
            var=(x.square().cumsum(1)/count-mean.square()).clamp_min(0)
            x=((x-mean)/(var.sqrt()+1e-6)).clamp(-5,5)
        # Keep a conservative warm-up mask for every feature.  This prevents
        # one-bar listing artifacts from becoming tradable formulas.
        out.append(torch.where(valid & (run >= 20),x,0))
    return torch.stack(out,1)


def _stateful_top1(scores, eligible, threshold, min_hold=5, cooldown=3):
    """One-position execution with entry threshold, minimum hold and cooldown."""
    n, tmax = scores.shape
    pos = torch.zeros_like(scores)
    active, age, wait = -1, 0, 0
    for t in range(tmax):
        if active >= 0:
            keep = bool(eligible[active, t] and scores[active, t] >= threshold)
            if age < min_hold or keep:
                pos[active, t] = 1.0; age += 1; continue
            active, age, wait = -1, 0, cooldown
        if wait > 0:
            wait -= 1; continue
        column = scores[:, t].masked_fill(~eligible[:, t], -torch.inf)
        winner = int(torch.argmax(column))
        if torch.isfinite(column[winner]) and float(column[winner]) >= threshold:
            active, age = winner, 1
            pos[active, t] = 1.0
    return pos


def metrics(factor,raw,target,a,b,threshold=1.734601,top_k=0,min_hold=5,cooldown=3):
    # Targets use t+1 and t+2: purge the final two decision rows of each segment.
    b-=2
    liq=raw['liquidity'][:,a:b]
    valid=raw['tradable'][:,a:b]
    eligible=(liq>500000)&valid
    scores=factor[:,a:b]
    if top_k:
        if top_k == 1:
            pos=_stateful_top1(scores,eligible,threshold,min_hold=min_hold,cooldown=cooldown)
        else:
            k=min(top_k,scores.shape[0])
            rank_scores=scores.masked_fill(~eligible,-torch.inf)
            winners=torch.topk(rank_scores,k,dim=0).indices
            pos=torch.zeros_like(scores)
            pos.scatter_(0,winners,torch.gather((scores>threshold).float(),0,winners))
            pos*=eligible.float()
    else:
        pos=((scores>threshold)&eligible).float()
    prev=torch.cat([torch.zeros_like(pos[:,:1]),pos[:,:-1]],1)
    changes=(pos-prev).abs()
    rate=.006+(1000/(liq+1e-9)).clamp(0,.05)
    pnl=pos*target[:,a:b]-changes*rate
    pnl[:,-1]-=pos[:,-1]*rate[:,-1]  # Forced segment liquidation.
    # Equal fixed notional per token; report additive P&L units, not compounded ROI.
    curve=pnl.mean(0).cumsum(0)
    peak=torch.cat([curve.new_zeros(1),curve]).cummax(0).values[1:]
    dd=float((peak-curve).max())
    net=float(curve[-1]); entries=int(((pos>0)&(prev==0)).sum())
    if top_k:
        if top_k == 1:
            hold=_stateful_top1(liq,eligible,float('-inf'),min_hold=min_hold,cooldown=cooldown)
        else:
            hold_scores=liq.masked_fill(~eligible,-torch.inf)
            winners=torch.topk(hold_scores,min(top_k,hold_scores.shape[0]),dim=0).indices
            hold=torch.zeros_like(hold_scores)
            hold.scatter_(0,winners,torch.gather(eligible.float(),0,winners))
    else:
        hold=eligible.float()
    hold_prev=torch.cat([torch.zeros_like(hold[:,:1]),hold[:,:-1]],1)
    hold_changes=(hold-hold_prev).abs()
    hold_pnl=hold*target[:,a:b]-hold_changes*rate
    hold_pnl[:,-1]-=hold[:,-1]*rate[:,-1]
    hold_net=float(hold_pnl.mean(0).sum())
    excess=net-hold_net
    # Penalize churn separately so a high turnover formula cannot win by
    # exploiting small noisy returns after costs.
    per_token=pnl.sum(1)
    median_excess=float((per_token-hold_pnl.sum(1)).median())
    active_tokens=int((pos.sum(1)>0).sum())
    positive_fraction=float((per_token>0).float().mean())
    concentration_penalty=max(0.0, 5-active_tokens)*0.05
    coverage_gate=max(5, int(0.20 * pos.shape[0] + 0.999))
    reward=0.5*excess+0.5*median_excess-dd-0.00005*float(changes.sum())-concentration_penalty if entries>=5 and active_tokens>=coverage_gate else -1.0
    return dict(net_pnl_per_initial_notional=net,hold_net_pnl=hold_net,
                excess_vs_hold=excess,drawdown_additive=dd,
                median_excess_vs_hold=median_excess,
                turnover_units=float(changes.sum()+pos[:,-1].sum()),entries=entries,
                exposure=float(pos.mean()),cost_per_initial_notional=float((changes*rate).sum()/pos.shape[0]+(pos[:,-1]*rate[:,-1]).mean()),
                active_tokens=active_tokens,median_token_pnl=float(per_token.median()),
                positive_token_fraction=positive_fraction,top_k=top_k,
                reward=reward)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=30); p.add_argument('--batch',type=int,default=64); p.add_argument('--out',required=True); args=p.parse_args()
    torch.manual_seed(42); torch.set_num_threads(4)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    loader=CryptoDataLoader(); loader.load_data(); raw=loader.raw_data_cache; feat=causal_features(raw); target=loader.target_ret
    torch.save({'raw':raw,'target':target,'addresses':loader.addresses},out/'data_snapshot.pt')
    T=target.shape[1]; cut=int(T*.6); val=int(T*.8)
    if cut<100 or T-val<10: raise ValueError('Insufficient history')
    model=AlphaGPT().to(feat.device); model.eval()  # Disable dropout, retain gradients.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4); vm=StackVM(); candidates={}; cache={}
    for step in range(args.steps):
        inp=torch.zeros(args.batch,1,dtype=torch.long,device=feat.device); depth=torch.zeros(args.batch,dtype=torch.long,device=feat.device); lp=[]; ent=[]
        for t in range(12):
            logits,_,_=model(inp); dist=Categorical(logits=logits.masked_fill(~legal_mask(depth,11-t),-torch.inf)); action=dist.sample(); lp.append(dist.log_prob(action)); ent.append(dist.entropy())
            arities=torch.tensor([0]*V.feature_count+[x[2] for x in OPS_CONFIG],device=feat.device); depth+=1-arities[action]; inp=torch.cat([inp,action[:,None]],1)
        rewards=[]
        for seq in inp[:,1:].tolist():
            key=tuple(load_formula(seq))
            if key not in cache:
                f=vm.execute(seq,feat[:,:,:cut]); cache[key]=metrics(f,{k:v[:,:cut] for k,v in raw.items()},target[:,:cut],20,cut)
            r=cache[key]['reward']; rewards.append(r); candidates[key]=r
        reward=torch.tensor(rewards,device=feat.device); adv=(reward-reward.mean())/(reward.std()+1e-5)
        loss=-(torch.stack(lp).sum(0)*adv.detach()).mean()-.01*torch.stack(ent).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        row=dict(step=step+1,mean_reward=float(reward.mean()),best_reward=max(candidates.values()),legal_rate=1.0,unique_formulas=len(candidates))
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    # Only training-ranked finalists enter validation; test is evaluated once after selection.
    finalists=sorted(candidates,key=candidates.get,reverse=True)[:10]; checked=[]
    for seq in finalists:
        f=vm.execute(seq,feat[:,:,:val]); checked.append((metrics(f,raw,target,cut,val)['reward'],seq))
    _,best=max(checked); f=vm.execute(best,feat)
    report={'config':vars(args),'seed':42,'split_indices':[20,cut,val,T],'formula':list(best),'vocab_version':FORMULA_VOCAB_VERSION,'token_names':list(V.token_names),'train':metrics(f,raw,target,20,cut),'validation':metrics(f,raw,target,cut,val),'test':metrics(f,raw,target,val,T),'limitations':['Research only: historical universe selection may introduce survivorship bias.','Previously inspected historical test period; fresh forward data required.','Additive fixed-notional approximation; gaps and execution fills not fully simulated.','Feature preprocessing differs from production runner; do not deploy this formula directly.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)); torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},out/'checkpoint.pt'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
