"""Short-cycle event baselines: closed 3m/5m bars, one-minute execution.

This module is research-only. It never changes the live strategy or timeframe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy

from .holdout_execution import simulate
from .resampling import resample_ohlcv

FIELDS = ('open', 'high', 'low', 'close', 'volume', 'liquidity', 'fdv')
SCORES = ('impulse', 'reversal', 'volume_breakout')


def snapshot_frame(snapshot=None):
    if snapshot:
        frame = pd.read_csv(snapshot, parse_dates=['time'])
    else:
        from .config import ModelConfig
        engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        try:
            with engine.connect() as connection:
                frame = pd.read_sql(sqlalchemy.text(
                    'SELECT time,address,open,high,low,close,volume,liquidity,fdv '
                    'FROM ohlcv ORDER BY address,time'), connection)
        finally:
            engine.dispose()
    if frame.empty:
        raise ValueError('No minute candles')
    frame['time'] = pd.to_datetime(frame['time'])
    if frame.duplicated(['address', 'time']).any():
        raise ValueError('Duplicate source candle')
    if not frame['time'].equals(frame['time'].dt.floor('1min')):
        raise ValueError('Source must be aligned minute candles')
    spacing = frame.groupby('address')['time'].diff().dropna()
    if spacing.empty or spacing.mode().iloc[0] != pd.Timedelta(minutes=1):
        raise ValueError('Source cadence is not predominantly one minute')
    return frame.sort_values(['address', 'time']).reset_index(drop=True)


def panel(frame, addresses, times):
    output = {}
    for name in FIELDS:
        table = frame.pivot(index='address', columns='time', values=name)
        output[name] = table.reindex(index=addresses, columns=times).to_numpy(dtype=float)
    observed = np.ones(output['open'].shape, dtype=bool)
    for name in FIELDS:
        values = output[name]
        observed &= np.isfinite(values) & (values > 0 if name in FIELDS[:4] else values >= 0)
    if 'observed' in frame:
        table = frame.pivot(index='address', columns='time', values='observed')
        observed &= table.reindex(index=addresses, columns=times).astype('boolean').fillna(False).to_numpy(dtype=bool)
    output['observed'] = observed
    return output


def shift(values, amount=1, fill=np.nan):
    out = np.full(values.shape, fill, dtype=values.dtype)
    if amount < values.shape[1]:
        out[:, amount:] = values[:, :-amount]
    return out


def event_features(raw):
    """No long MA or whole-series fitted normalization; each feature is causal."""
    ok = raw['observed']
    c, o, h, low, v = [raw[name] for name in ('close', 'open', 'high', 'low', 'volume')]
    prev_ok = shift(ok, fill=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        body_return = c / o - 1
        ret1 = c / shift(c) - 1
        pressure = np.where(h > low, (c - o) / (h - low), 0.0)
        # Volume burst relative to the preceding completed bar; no forced
        # twenty-bar warm-up for a newly listed token.
        volume_burst = np.log1p(v) - np.log1p(shift(v))
        liquidity_change = raw['liquidity'] / shift(raw['liquidity']) - 1
    ret1 = np.where(ok & prev_ok, ret1, np.nan)
    volume_burst = np.where(ok & prev_ok, volume_burst, np.nan)
    liquidity_change = np.where(ok & prev_ok, liquidity_change, np.nan)
    high3 = h.copy()
    for lag in (1, 2):
        previous = shift(h, lag)
        high3 = np.fmax(high3, previous)
    # A gap resets short rolling highs rather than spanning a missing period.
    for t in range(h.shape[1]):
        for lag in (1, 2):
            if t >= lag:
                continuity = ok[:, t-lag:t+1].all(axis=1)
                if lag == 1:
                    high3[:, t] = h[:, t]
                high3[:, t] = np.where(continuity, np.fmax(high3[:, t], h[:, t-lag]), high3[:, t])
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown3 = c / high3 - 1
    result = dict(body_return=body_return, ret1=ret1, pressure=pressure,
                  volume_burst=volume_burst, liquidity_change=liquidity_change,
                  drawdown3=drawdown3)
    return {name: np.where(ok & np.isfinite(value), value, np.nan)
            for name, value in result.items()}


def event_scores(features):
    impulse = np.clip(features['body_return'] / .01, -10, 10) + features['pressure']
    reversal = -np.clip(features['ret1'] / .01, -10, 10)
    volume = impulse * np.clip(features['volume_burst'], 0, 5)
    return dict(impulse=impulse, reversal=reversal, volume_breakout=volume)


def forward_labels(raw, horizon):
    """Signal t: enter open[t+1], scheduled exit open[t+1+h].

    Labels are diagnostics/supervised targets only, never a trading mask or a
    per-minute P&L. A multi-bar label must not be summed on each held minute.
    """
    op, valid = raw['open'], raw['observed']
    result = np.full(op.shape, np.nan)
    mask = np.zeros(op.shape, dtype=bool)
    count = op.shape[1] - horizon - 1
    if count > 0:
        good = valid[:, :count].copy()
        for k in range(1, horizon + 2):
            good &= valid[:, k:k+count]
        value = op[:, horizon+1:horizon+1+count] / op[:, 1:1+count] - 1
        mask[:, :count] = good & np.isfinite(value)
        result[:, :count] = np.where(mask[:, :count], value, np.nan)
    return result, mask


def to_minute_signals(scores, bucket_times, minute_times, width):
    """Publish only at the final constituent minute's close, then carry forward."""
    out = np.full((scores.shape[0], len(minute_times)), np.nan)
    decision_times = bucket_times + pd.Timedelta(minutes=width-1)
    indexes = minute_times.get_indexer(decision_times)
    current = np.full(scores.shape[0], np.nan)
    updates = {int(t): i for i, t in enumerate(indexes) if t >= 0}
    for t in range(len(minute_times)):
        if t in updates:
            current = scores[:, updates[t]].copy()
        out[:, t] = current
    return out


def split_addresses(addresses):
    train, holdout = [], []
    for i, address in enumerate(addresses):
        bucket = int(hashlib.sha256(address.encode()).hexdigest()[:8], 16) % 5
        (train if bucket < 3 else holdout).append(i)
    if not train or not holdout:
        raise ValueError('Address split must contain both groups')
    return train, holdout


def fit_threshold(scores, eligible, ids, stop):
    sample = scores[ids, :stop][eligible[ids, :stop] & np.isfinite(scores[ids, :stop])]
    if len(sample) < 20:
        return None
    # Frozen, preregistered choices: sparse entry and lower exit threshold.
    return {'entry': float(np.quantile(sample, .95)), 'exit': float(np.quantile(sample, .50)),
            'samples': int(len(sample))}


def compact(result):
    compacted = {k: v for k, v in result.items() if k not in ('equity_path', 'entry_events', 'events')}
    pnl = np.asarray(result.get('per_token_pnl_fraction', []), dtype=float)
    active = int(result.get('active_tokens', 0))
    compacted['coverage_fraction'] = float(active / pnl.size) if pnl.size else 0.0
    abs_pnl = np.abs(pnl[pnl != 0])
    compacted['max_active_token_abs_share'] = float(abs_pnl.max() / abs_pnl.sum()) if abs_pnl.size and abs_pnl.sum() else 0.0
    return compacted


def metric_key(result):
    if not result['available'] or result.get('bankrupt') or result['entries'] < 5:
        return -np.inf
    pnl = np.asarray(result.get('per_token_pnl_fraction', []), dtype=float)
    active = int(result.get('active_tokens', 0))
    if pnl.size == 0 or active < max(5, int(np.ceil(.20 * pnl.size))):
        return -np.inf
    active_pnl = np.abs(pnl[pnl != 0])
    if active_pnl.size and float(active_pnl.max() / active_pnl.sum()) > .25:
        return -np.inf
    return result['net_pnl_fraction'] - result['max_drawdown_fraction']


def execute(scores, raw, eligibility, ids, a, b, threshold, width, horizon):
    sliced = {name: values[ids] for name, values in raw.items()}
    return simulate(scores[ids], sliced, a, b, threshold['entry'], threshold['exit'],
                    hold_bars=width*horizon, cooldown=width, min_hold_bars=width,
                    eligible=eligibility[ids], min_liquidity=500000., stop_loss=.05,
                    liquidity_drop_fraction=.20, max_impact=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--snapshot', help='Frozen source CSV(.gz); otherwise read DB only')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    frame = snapshot_frame(args.snapshot)
    minute_times = pd.date_range(frame.time.min(), frame.time.max(), freq='1min')
    train_end, val_end = int(len(minute_times)*.6), int(len(minute_times)*.8)
    if train_end < 100:
        raise ValueError('Insufficient minute history')
    # Selection uses observations before training cutoff; no top-by-full-history
    # truncation and no current-token-list join. Surviving DB universe bias remains.
    addresses = sorted(frame.loc[frame.time < minute_times[train_end], 'address'].unique())
    frame = frame[frame.address.isin(addresses)].copy()
    train_ids, holdout_ids = split_addresses(addresses)
    frame.to_csv(out/'minutes.csv.gz', index=False, compression='gzip')
    digest = hashlib.sha256((out/'minutes.csv.gz').read_bytes()).hexdigest()
    protocol = dict(version=1, source_start=str(minute_times[0]), source_end=str(minute_times[-1]),
                    source_sha256=digest, source_rows=len(frame), source_cadence='1min',
                    timezone='provider local-naive; not relabelled as UTC',
                    timeframes=[3,5], horizons=[3,5], candidates=list(SCORES),
                    entry_quantile=.95, exit_quantile=.50, execution_clock='1min',
                    fee_one_way=.006, notional=1000., min_liquidity=500000.,
                    stop_loss=.05, liquidity_drop_fraction=.20,
                    split_indices=[0,train_end,val_end,len(minute_times)],
                    train_addresses=[addresses[i] for i in train_ids],
                    heldout_addresses=[addresses[i] for i in holdout_ids],
                    written_at_utc=datetime.now(timezone.utc).isoformat())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    raw = panel(frame, addresses, minute_times)
    raw['open_available'] = np.isfinite(raw['open']) & (raw['open'] > 0)
    # The minute-close liquidity at t is usable for an open[t+1] estimate.
    # Provider may backfill snapshot liquidity: document, never call this true
    # historical pool depth. These remain scenario costs, not measured fills.
    raw['execution_liquidity'] = shift(raw['liquidity'])
    eligibility = raw['observed'] & (raw['liquidity'] > 500000.)
    rows = []; candidate_signals = {}; thresholds = {}; labels_info = {}
    for width in (3, 5):
        aggregated = resample_ohlcv(frame, f'{width}min')
        aggregated.to_csv(out/f'bars_{width}m.csv.gz', index=False, compression='gzip')
        bucket_times = pd.date_range(aggregated.time.min(), aggregated.time.max(), freq=f'{width}min')
        grouped_raw = panel(aggregated, addresses, bucket_times)
        features = event_features(grouped_raw)
        base_scores = event_scores(features)
        label_train = bucket_times + pd.Timedelta(minutes=width) <= minute_times[train_end]
        for h in (3,5):
            y, mask = forward_labels(grouped_raw, h)
            # Purge any label whose exit is outside the train cutoff.
            end_times = bucket_times + pd.Timedelta(minutes=width*(h+1))
            train_mask = mask[train_ids] & label_train & (end_times < minute_times[train_end])
            labels_info[f'{width}m_h{h}'] = {'train_valid_labels':int(train_mask.sum()),
                'mean_train_return':float(y[train_ids][train_mask].mean()) if train_mask.any() else None}
            np.savez_compressed(out/f'labels_{width}m_h{h}.npz', values=y, valid=mask)
        for name, score in base_scores.items():
            expanded = to_minute_signals(score, bucket_times, minute_times, width)
            ready = eligibility & np.isfinite(expanded)
            # One sample per completed bar; don't weight training by minutes.
            sample_mask = np.zeros_like(ready)
            sample_times = minute_times.get_indexer(bucket_times + pd.Timedelta(minutes=width-1))
            sample_mask[:,sample_times[sample_times>=0]] = True
            threshold = fit_threshold(expanded, ready & sample_mask, train_ids, train_end)
            if threshold is None:
                continue
            for horizon in (3,5):
                key=f'{width}m_{name}_h{horizon}'
                candidate_signals[key]=(expanded,ready,width,horizon)
                thresholds[key]=threshold
                row={'id':key,'threshold':threshold,'timeframe_minutes':width,'hold_bars':horizon,
                     'max_hold_minutes':width*horizon}
                for group,ids,a,b in [('train',train_ids,0,train_end),
                                      ('validation',train_ids,train_end,val_end)]:
                    row[group] = compact(execute(expanded,raw,ready,ids,a,b,threshold,width,horizon))
                rows.append(row)
                print(json.dumps({'candidate':key,'train_net':row['train']['net_pnl_fraction'],
                                  'validation_net':row['validation']['net_pnl_fraction'],
                                  'valid':row['validation']['available']}),flush=True)
    selectable = [r for r in rows if np.isfinite(metric_key(r['validation']))]
    selected = max(selectable,key=lambda r:metric_key(r['validation'])) if selectable else None
    # Freeze selection before reading held-out-address and later-time outcomes.
    freeze = {'selected':None if selected is None else selected['id'],
              'threshold':None if selected is None else selected['threshold'],
              'forward_after':str(minute_times[-1]+pd.Timedelta(minutes=1)),
              'status':'research candidate only; no deployment', 'source_sha256':digest}
    (out/'selection.json').write_text(json.dumps(freeze,indent=2))
    comparisons = {}
    if selected:
        key=selected['id']; score,ready,width,h= candidate_signals[key]
        for name,ids in [('time_holdout',train_ids),('address_and_time_holdout',holdout_ids)]:
            full=execute(score,raw,ready,ids,val_end,len(minute_times),thresholds[key],width,h)
            (out/f'{name}_trades.json').write_text(json.dumps(full,indent=2,allow_nan=False))
            selected[name]=compact(full)
            # Matched execution costs; buy once in highest-liquidity currently
            # eligible token and hold unless a safety exit is necessary.
            liquidity=raw['liquidity']
            hold=execute(liquidity,raw,eligibility,ids,val_end,len(minute_times),
                         {'entry':500000.,'exit':0.},1,len(minute_times))
            comparisons[name]={'cash_net':0.,'liquidity_hold':compact(hold)}
    limitations=[
        'Historical stability diagnostic, not untouched blind test: this database has been inspected.',
        'Prior collection used trending/current-token filters; dead/uncollected tokens remain absent.',
        'Historical liquidity/FDV can be backfilled current snapshots. Liquidity-change factors are diagnostic only.',
        '0.6% single-side base fee plus size/liquidity impact is an assumption, not verified realized execution cost.',
        'Minute-open fills and minute-close safety checks cannot represent second-level rugs, chain delay or rejected swaps.',
        '3m/5m signal bars with one-minute risk checks; 3/5 bars mean maximum 9/15/15/25 minute horizons, not mandatory holds.',
        'The fixed preregistered baseline comparison is not full expanding-window retraining.',
    ]
    report={'protocol':protocol,'labels':labels_info,'candidates':rows,'selected':freeze['selected'],
            'baselines':comparisons,'limitations':limitations,'deployed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# 短周期事件基线评估','',
           f"来源：{minute_times[0]} 至 {minute_times[-1]}，{len(addresses)} 个币；训练地址 {len(train_ids)}，留出地址 {len(holdout_ids)}。",'',
           '信号使用完整 3m/5m OHLCV；执行和风控每分钟检查；无长期均线。以下为模拟权益净收益（初始单本金），不与旧累加指标直接比较。','',
           '| 候选 | 训练净收益 | 验证净收益 | 验证可用 |','|---|---:|---:|---|']
    def fmt(x): return '不可估值' if x is None else f'{x:.2%}'
    for r in rows:
        lines.append(f"| {r['id']} | {fmt(r['train']['net_pnl_fraction'])} | {fmt(r['validation']['net_pnl_fraction'])} | {r['validation']['available']} |")
    lines += ['',f"冻结候选：{freeze['selected'] or '无可用候选'}。"]
    if selected:
        for name in comparisons:
            m=selected[name]; base=comparisons[name]['liquidity_hold']
            lines += [f"- {name}：净收益 {fmt(m['net_pnl_fraction'])}，数据可用 {m['available']}，入场 {m['entries']} 次；持有基线 {fmt(base['net_pnl_fraction'])}。"]
    lines += ['','## 边界','']+['- '+x for x in limitations]
    lines += ['','产物只供研究；未部署或覆盖现有策略。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(out/'report.json'),'selected':freeze['selected'],'deployed':False}),flush=True)


if __name__ == '__main__':
    main()
