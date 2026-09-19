import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.addresses = []
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT t.address
        FROM tokens AS t
        JOIN ohlcv AS o ON o.address = t.address
        GROUP BY t.address
        ORDER BY COUNT(*) DESC, t.address ASC
        LIMIT {limit_tokens}
        """
        self.addresses = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not self.addresses: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(self.addresses) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        if df.empty:
            raise ValueError("No candles found.")
        # A step must mean one minute even when a token has missing candles.
        from data_pipeline.config import Config
        frequency = {'1m': '1min', '15m': '15min'}[Config.TIMEFRAME]
        timeline = pd.date_range(df['time'].min(), df['time'].max(), freq=frequency)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.reindex(index=timeline, columns=self.addresses)
            pivot = pivot.fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        def present_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.reindex(index=timeline, columns=self.addresses)
            return torch.tensor(pivot.notna().values.T, dtype=torch.bool, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        # A zero-filled gap is not an observed candle.  Keep the original
        # presence mask so research features cannot manufacture signals at
        # listing boundaries or during missing-data intervals.
        valid = torch.ones_like(self.raw_data_cache['open'], dtype=torch.bool)
        for key, values in self.raw_data_cache.items():
            valid &= present_tensor(key)
            valid &= torch.isfinite(values)
            valid &= values > 0 if key in {'open', 'high', 'low', 'close'} else values >= 0
        self.raw_data_cache['observed'] = valid
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        target_valid = valid & torch.roll(valid, -1, dims=1) & torch.roll(valid, -2, dims=1)
        target_valid[:, -2:] = False
        self.raw_data_cache['tradable'] = target_valid
        # Compute logs only of positive values; masked gaps are not tradable.
        tiny = torch.finfo(op.dtype).tiny
        self.target_ret = torch.where(
            target_valid, torch.log(t2.clamp_min(tiny)) - torch.log(t1.clamp_min(tiny)), 0.0
        )
        self.target_ret[:, -2:] = 0.0
        if not torch.isfinite(self.feat_tensor).all() or not torch.isfinite(self.target_ret).all():
            raise ValueError("Non-finite training data after preparation")
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")
