"""Report whether point-in-time liquidity coverage is sufficient for research."""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import sqlalchemy

from .config import Config


def main():
    p=argparse.ArgumentParser(); p.add_argument('--min-snapshots',type=int,default=20); p.add_argument('--max-age-minutes',type=float,default=15); p.add_argument('--min-liquidity',type=float,default=500000); p.add_argument('--min-volume-5m',type=float,default=10000); p.add_argument('--min-txns-5m',type=int,default=10); p.add_argument('--source',default='dexscreener'); args=p.parse_args()
    engine=sqlalchemy.create_engine(Config.DB_DSN)
    query=sqlalchemy.text('''
      SELECT address, time, liquidity, fdv, volume_5m, txns_5m, source
      FROM liquidity_snapshots
      WHERE source IN ('dexscreener','birdeye_trending')
      ORDER BY address,time
    ''')
    with engine.connect() as c: frame=pd.read_sql(query,c)
    now=pd.Timestamp.now(tz=None)
    if frame.empty:
        report={'available':False,'reason':'no liquidity snapshots','addresses':0,'eligible_addresses':[]}
    else:
        frame['time']=pd.to_datetime(frame['time'])
        rows=[]
        for address, g in frame.groupby('address'):
            g=g.sort_values('time')
            source_g=g[g.source == args.source]
            last=source_g.iloc[-1] if not source_g.empty else g.iloc[-1]
            age=(now-last['time']).total_seconds()/60
            rows.append({'address':address,'snapshots':int(len(source_g)),'last_time':str(last['time']),'age_minutes':age,'latest_liquidity':float(last['liquidity']) if pd.notna(last['liquidity']) else None,'latest_volume_5m':float(last['volume_5m']) if pd.notna(last['volume_5m']) else None,'latest_txns_5m':int(last['txns_5m']) if pd.notna(last['txns_5m']) else None,'sources':sorted(g.source.unique().tolist())})
        eligible=[r for r in rows if r['snapshots']>=args.min_snapshots and r['age_minutes']<=args.max_age_minutes and r['latest_liquidity'] is not None and r['latest_liquidity']>=args.min_liquidity and (args.source != 'dexscreener' or (r['latest_volume_5m'] is not None and r['latest_volume_5m']>=args.min_volume_5m and r['latest_txns_5m'] is not None and r['latest_txns_5m']>=args.min_txns_5m))]
        report={'available':bool(eligible),'now':str(now),'config':vars(args),'addresses':len(rows),'eligible_addresses':eligible,'coverage':rows,'reason':None if eligible else 'insufficient fresh snapshot coverage'}
    out=json.dumps(report,indent=2); print(out)

if __name__=='__main__': main()
