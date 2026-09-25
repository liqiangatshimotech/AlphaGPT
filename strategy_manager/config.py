class StrategyConfig:
    # Keep exposure diversified while respecting the live wallet balance.
    MAX_OPEN_POSITIONS = 5
    ENTRY_AMOUNT_SOL = 1.0
    MIN_ENTRY_LIQUIDITY_USD = 500000.0
    # Reject a buy when an immediate full-size round trip is quoted to lose
    # more than this many basis points (300 bps = 3%). Can be overridden with
    # MAX_ENTRY_ROUND_TRIP_COST_BPS in the environment.
    MAX_ENTRY_ROUND_TRIP_COST_BPS = 300
    STOP_LOSS_PCT = -0.05
    TAKE_PROFIT_Target1 = 0.10
    TP_Target1_Ratio = 0.5
    TRAILING_ACTIVATION = 0.05
    TRAILING_DROP = 0.03
    STOP_LOSS_COOLDOWN_SECONDS = 86400
    POSITION_BALANCE_TIMEOUT_SECONDS = 5
    POSITION_QUOTE_TIMEOUT_SECONDS = 6
    POSITION_EXIT_TIMEOUT_SECONDS = 15
    ENTRY_SCAN_TIMEOUT_SECONDS = 20
    ZERO_BALANCE_ALERT_INTERVAL_SECONDS = 3600
    FAILED_DEX_EXCLUSION_SECONDS = 120
    # A DEX label lookup must not consume the exit budget needed to quote and send.
    DEX_LABEL_LOOKUP_TIMEOUT_SECONDS = 1.0
    BUY_THRESHOLD = 0.85
    SELL_THRESHOLD = 0.45
