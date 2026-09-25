import math
import os

from .config import StrategyConfig
from execution.config import ExecutionConfig
from execution.jupiter import JupiterAggregator
from loguru import logger


def _positive_raw_amount(value):
    """Accept only a positive Solana u64 amount in atomic token units."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("amount must be a positive integer")
    if isinstance(value, str) and (
        not value or not value.isascii() or not value.isdecimal()
    ):
        raise ValueError("amount must contain decimal digits only")
    amount = int(value)
    if not 0 < amount <= 2**64 - 1:
        raise ValueError("amount must be in the positive u64 range")
    return amount


def _matches_quote(quote, input_mint, output_mint, in_amount):
    """Check the quote is for the exact route and input size being evaluated."""
    return (
        isinstance(quote, dict)
        and quote.get("inputMint") == input_mint
        and quote.get("outputMint") == output_mint
        and _positive_raw_amount(quote.get("inAmount")) == in_amount
        and quote.get("swapMode") == "ExactIn"
    )


class RiskEngine:
    def __init__(self):
        self.config = StrategyConfig()
        self.jup = JupiterAggregator()

    async def check_safety(self, token_address, liquidity_usd):
        try:
            liquidity = float(liquidity_usd)
        except (TypeError, ValueError, OverflowError):
            liquidity = float("nan")
        if (not math.isfinite(liquidity)
                or liquidity < self.config.MIN_ENTRY_LIQUIDITY_USD):
            logger.warning(f"[x] Risk: Liquidity too low (${liquidity_usd})")
            return False

        return True

    async def check_entry_round_trip(
        self, token_address, buy_quote, *, input_lamports, quote_provider=None
    ):
        """Reject entries whose quoted immediate full-position exit costs too much.

        The buy quote must be the exact quote later submitted for the purchase.
        This uses expected output amounts, so it does not include swap execution
        slippage, network fees, or price movement after the quotes are obtained.
        """
        try:
            input_lamports = _positive_raw_amount(input_lamports)
            if token_address == ExecutionConfig.SOL_MINT or not _matches_quote(
                buy_quote, ExecutionConfig.SOL_MINT, token_address, input_lamports
            ):
                raise ValueError("buy quote does not match intended trade")
            token_amount = _positive_raw_amount(buy_quote.get("outAmount"))
            cost_limit_bps = int(os.getenv(
                "MAX_ENTRY_ROUND_TRIP_COST_BPS",
                str(self.config.MAX_ENTRY_ROUND_TRIP_COST_BPS),
            ))
            if not 0 <= cost_limit_bps < 10000:
                raise ValueError("round-trip cost limit must be between 0 and 9999 bps")
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning(f"[x] Risk: Invalid entry quote or cost limit: {exc}")
            return False

        provider = quote_provider if quote_provider is not None else self.jup
        try:
            sell_quote = await provider.get_quote(
                input_mint=token_address,
                output_mint=ExecutionConfig.SOL_MINT,
                amount_integer=token_amount,
            )
            if not _matches_quote(
                sell_quote, token_address, ExecutionConfig.SOL_MINT, token_amount
            ):
                raise ValueError("sell quote does not match full expected position")
            sell_lamports = _positive_raw_amount(sell_quote.get("outAmount"))
        except Exception as exc:
            logger.warning(f"[x] Risk: Cannot verify full-size sell path: {exc}")
            return False

        # Integer comparison keeps the threshold exact at the boundary.
        loss_lamports = input_lamports - sell_lamports
        cost_bps = 10000 * loss_lamports / input_lamports
        if 10000 * loss_lamports > cost_limit_bps * input_lamports:
            logger.warning(
                f"[x] Risk: Round-trip quote loses {cost_bps:.2f} bps "
                f"(limit {cost_limit_bps} bps; {input_lamports} -> {sell_lamports} lamports)"
            )
            return False

        logger.info(
            f"Entry round-trip quote: {cost_bps:.2f} bps "
            f"({input_lamports} -> {sell_lamports} lamports)"
        )
        return True

    def calculate_position_size(self, wallet_balance_sol):
        size = self.config.ENTRY_AMOUNT_SOL
        if wallet_balance_sol is None:
            return 0.0

        if wallet_balance_sol < size + 0.1:
            return 0.0
            
        return size

    async def close(self):
        await self.jup.close()
