from monitor_exit_distance import exit_distance


def test_distances_from_current_price_not_historical_high():
    pos = {"entry_price": 1, "highest_price": 1.08, "is_moonbag": False}
    result = exit_distance(pos, 1.04)
    assert round(result["take_profit_pct"], 4) == round((1.1 - 1.04) / 1.04 * 100, 4)
    assert round(result["stop_pct"], 4) == round((1.04 - .95) / 1.04 * 100, 4)
    assert round(result["trailing_stop_pct"], 4) == round((1.04 - 1.08 * .97) / 1.04 * 100, 4)


def test_moonbag_has_no_remaining_first_take_profit():
    result = exit_distance({"entry_price": 1, "highest_price": 1.12, "is_moonbag": True}, 1.07)
    assert "take_profit_pct" not in result
    assert result["trailing_stop_pct"] < 0


def test_trailing_stop_not_active_at_threshold():
    result = exit_distance({"entry_price": 1, "highest_price": 1.049}, 1.02)
    assert "trailing_stop_pct" not in result
