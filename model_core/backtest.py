import torch

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060
        self.buy_threshold = 0.85
        self.min_score_spread = 0.01

    def check_candidate_quality(self, factors, raw_data):
        """Check ranking quality at the supplied tensor's final candle.

        Raw-logit variance over the whole history is insufficient: a formula
        can vary in old candles yet saturate sigmoid to 1 for every token at
        that candle. Use the same sigmoid and buy threshold as the live entry
        scan, restricted to the backtest's tradable-liquidity universe.
        """
        liquidity = raw_data["liquidity"]
        if factors.ndim != 2 or factors.shape != liquidity.shape:
            return False, "factor and liquidity shapes differ"
        if not torch.isfinite(factors).all().item():
            return False, "non-finite formula output"

        tradable = torch.isfinite(liquidity[:, -1]) & (liquidity[:, -1] > self.min_liq)
        if tradable.sum().item() < 2:
            return False, "fewer than two tradable tokens at the latest candle"
        scores = torch.sigmoid(factors[tradable, -1])
        if (scores.max() - scores.min()).item() < self.min_score_spread:
            return False, "latest tradable scores cannot separate tokens"

        buys = scores[scores > self.buy_threshold]
        if buys.numel() > scores.numel() / 2:
            return False, "latest signal buys more than half the tradable universe"
        if buys.numel() > 1 and (buys.max() - buys.min()).item() < self.min_score_spread:
            return False, "latest buy candidates have indistinguishable scores"
        return True, ""

    @staticmethod
    def temporal_windows(num_candles):
        """Reserve the last 20% for a final test, with a two-candle label gap."""
        if num_candles < 30:
            raise ValueError("At least 30 candles are required for temporal validation")
        label_horizon = 2  # target_ret[t] uses opens at t + 1 and t + 2.
        last_labeled = num_candles - label_horizon
        holdout_start = int(last_labeled * 0.8)
        training_end = holdout_start - label_horizon
        if training_end < 5 or last_labeled - holdout_start < 5:
            raise ValueError("At least 30 candles are required for temporal validation")
        return slice(0, training_end), slice(holdout_start, last_labeled)

    def evaluate(self, factors, raw_data, target_ret, period=None):
        if period is None:
            period = slice(None)
        liquidity = raw_data['liquidity'][:, period]
        factors = factors[:, period]
        target_ret = target_ret[:, period]
        if not (factors.shape == liquidity.shape == target_ret.shape):
            raise ValueError("factor, liquidity, and target shapes must match")

        signal = torch.sigmoid(factors)
        valid_target = torch.isfinite(target_ret)
        is_safe = torch.isfinite(liquidity) & (liquidity > self.min_liq)
        position = ((signal > self.buy_threshold) & is_safe & valid_target).float()
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.where(
            is_safe, torch.clamp(impact_slippage, 0.0, 0.05), 0.05
        )
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        # 0 * -inf is NaN. Missing pre-listing opens do not constitute trades.
        gross_pnl = torch.where(position.bool(), target_ret, 0.0)
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        score = cum_ret - (big_drawdowns * 2.0)
        activity = position.sum(dim=1)
        active = activity >= 5
        # Including inactive tokens as -10 in the median made buying most of
        # the universe the easiest way to improve fitness. Score participating
        # tokens only, but demand enough breadth to avoid one lucky token.
        minimum_active = max(2, (factors.shape[0] + 19) // 20)
        if active.sum().item() < minimum_active:
            return score.new_tensor(-10.0), 0.0
        return torch.median(score[active]), cum_ret[active].mean().item()
