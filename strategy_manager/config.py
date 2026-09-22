class StrategyConfig:
    # Keep exposure diversified while respecting the live wallet balance.
    MAX_OPEN_POSITIONS = 5
    ENTRY_AMOUNT_SOL = 1.0
    STOP_LOSS_PCT = -0.05
    TAKE_PROFIT_Target1 = 0.10
    TP_Target1_Ratio = 0.5
    TRAILING_ACTIVATION = 0.05
    TRAILING_DROP = 0.03
    STOP_LOSS_COOLDOWN_SECONDS = 900
    BUY_THRESHOLD = 0.85
    SELL_THRESHOLD = 0.45
