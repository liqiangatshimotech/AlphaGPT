"""Read-only unrealized PnL and exit-distance reporting for the monitor."""

import asyncio
import math
import os
import time

import aiohttp

from strategy_manager.config import StrategyConfig
from execution.rate_limit import wait_for_shared_slot


SOL_MINT = "So11111111111111111111111111111111111111112"
_JUPITER_MIN_INTERVAL = float(os.getenv("JUPITER_MONITOR_MIN_INTERVAL_SECONDS", "0.25"))


def exit_distance(position, current_price):
    """Return price-relative distances to the strategy's price-based exits."""
    entry = float(position["entry_price"])
    high = float(position["highest_price"])
    if any(not math.isfinite(value) or value <= 0 for value in (entry, current_price, high)):
        raise ValueError("invalid position price")
    high = max(high, current_price)

    stop = entry * (1 + StrategyConfig.STOP_LOSS_PCT)
    result = {
        "pnl_pct": (current_price - entry) / entry * 100,
        "stop_pct": (current_price - stop) / current_price * 100,
    }
    if not position.get("is_moonbag"):
        target = entry * (1 + StrategyConfig.TAKE_PROFIT_Target1)
        result["take_profit_pct"] = (target - current_price) / current_price * 100

    if (high - entry) / entry > StrategyConfig.TRAILING_ACTIVATION:
        trailing = high * (1 - StrategyConfig.TRAILING_DROP)
        result["trailing_stop_pct"] = (current_price - trailing) / current_price * 100
    return result


async def _quote_position(session, rpc_url, quote_url, headers, position):
    address = position["token_address"]
    async with session.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [address]
    }) as response:
        response.raise_for_status()
        supply = await response.json()
    decimals = int(supply["result"]["value"]["decimals"])
    params = {
        "inputMint": address, "outputMint": SOL_MINT,
        "amount": str(10 ** decimals), "slippageBps": "100",
        "onlyDirectRoutes": "false", "asLegacyTransaction": "false",
    }
    for attempt in range(4):
        await wait_for_shared_slot(_JUPITER_MIN_INTERVAL)
        async with session.get(quote_url, params=params, headers=headers) as response:
            if response.status == 200:
                quote = await response.json()
                return int(quote["outAmount"]) / 1e9
            body = await response.text()
            if response.status == 429 and attempt < 3:
                try:
                    retry_after = min(float(response.headers.get("Retry-After", 0)), 60.0)
                except (TypeError, ValueError):
                    retry_after = 0.0
                await asyncio.sleep(max(retry_after, min(2.0 ** attempt, 30.0)))
                continue
            response.raise_for_status()
            raise RuntimeError(f"Jupiter quote failed: {response.status} {body[:200]}")


async def position_report(positions):
    if not positions:
        return ""
    rpc_url = os.getenv("QUICKNODE_RPC_URL", "https://api.mainnet-beta.solana.com")
    quote_url = os.getenv("JUPITER_BASE_URL", "https://api.jup.ag/swap/v1").rstrip("/") + "/quote"
    headers = {"accept": "application/json"}
    if os.getenv("JUPITER_API_KEY"):
        headers["x-api-key"] = os.environ["JUPITER_API_KEY"]
    lines = ["逐仓浮动盈亏（相对入场价，未计手续费及已实现盈亏）："]
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for pos in positions:
            # The preserved monitor renames portfolio-state fields in its
            # snapshot. Support both representations without altering state.
            pos = {
                **pos,
                "token_address": pos.get("token_address", pos.get("token")),
                "entry_price": pos.get("entry_price", pos.get("entry_price_sol")),
                "highest_price": pos.get("highest_price", pos.get("highest_price_sol")),
            }
            name = pos.get("symbol") or pos.get("token_address") or "未知仓位"
            try:
                price = await _quote_position(session, rpc_url, quote_url, headers, pos)
                distances = exit_distance(pos, price)
                pnl = round(distances["pnl_pct"], 2)
                if pnl > 0:
                    performance = f"赚 {pnl:.2f} 个点（{pnl:+.2f}%）"
                elif pnl < 0:
                    performance = f"亏 {abs(pnl):.2f} 个点（{pnl:+.2f}%）"
                else:
                    performance = "持平（0.00%）"
                parts = [f"固定止损 {distances['stop_pct']:+.2f}%"]
                if "take_profit_pct" in distances:
                    parts.insert(0, f"首档止盈 {distances['take_profit_pct']:+.2f}%")
                else:
                    parts.insert(0, "首档止盈已执行")
                if "trailing_stop_pct" in distances:
                    parts.append(f"移动止损 {distances['trailing_stop_pct']:+.2f}%")
                lines.append(f"- {name}：{performance}\n  " + "；".join(parts))
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, TypeError, OverflowError) as exc:
                lines.append(f"- {name}：报价暂不可用（{type(exc).__name__}）")
    lines.append("止盈止损距离相对当前报价计算，负数表示已越过触发价。")
    return "\n".join(lines)
