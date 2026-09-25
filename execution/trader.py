import asyncio
import inspect
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from loguru import logger
from solders.pubkey import Pubkey
from solana.rpc.types import TokenAccountOpts

from .config import ExecutionConfig
from .rpc_handler import (
    QuickNodeClient,
    TransactionFailed,
    TransactionPreflightRejected,
    TransactionStatusUnknown,
)
from .jupiter import JupiterAggregator

class SolanaTrader:
    def __init__(self):
        self.config = ExecutionConfig
        self.rpc = QuickNodeClient()
        self.jup = JupiterAggregator()
        self.is_running = True
        self.TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    async def buy(
        self,
        token_address: str,
        amount_sol: float,
        slippage_bps=500,
        *,
        should_cancel=None,
        quote_response=None,
        on_submitting=None,
    ):
        if should_cancel is not None and should_cancel():
            return False
        logger.info(f"Executing BUY: {amount_sol} SOL -> {token_address}")
        balance = await self.rpc.get_balance()
        if balance is None:
            logger.warning("BUY deferred because the wallet balance is unavailable from RPC.")
            return False
        if balance < amount_sol + 0.02:
            logger.warning(f"Insufficient SOL balance: {balance}. Needed: {amount_sol + 0.02}")
            return False
        amount_lamports = int(amount_sol * 1e9)
        quote = quote_response
        if quote is None:
            quote = await self.jup.get_quote(
                input_mint=ExecutionConfig.SOL_MINT,
                output_mint=token_address,
                amount_integer=amount_lamports,
                slippage_bps=slippage_bps
            )
        if not quote:
            logger.error("No quote found.")
            return False
        out_amount = int(quote['outAmount'])
        logger.info(f"Quote received. Est. Output: {out_amount} raw units.")
        pre_raw_balance = await self.rpc.get_token_balance(token_address)
        if pre_raw_balance is None or pre_raw_balance > 0:
            logger.warning("BUY deferred: existing or unavailable on-chain token balance.")
            return False
        supply = await self.rpc.client.get_token_supply(Pubkey.from_string(token_address))
        decimals = int(supply.value.decimals)
        swap = await self.jup.get_swap_tx(quote, include_metadata=True)
        if (not swap or not swap.get("swapTransaction")
                or not isinstance(swap.get("lastValidBlockHeight"), int)
                or swap["lastValidBlockHeight"] <= 0):
            return False
        if should_cancel is not None and should_cancel():
            logger.warning("Buy cancelled before signing/submission.")
            return False
        try:
            txn = self.jup.deserialize_and_sign(swap["swapTransaction"])
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return False
        try:
            async def record(metadata):
                metadata = {**metadata, "pre_raw_balance": pre_raw_balance,
                            "decimals": decimals, "expected_out": out_amount,
                            "amount_raw": amount_lamports}
                if on_submitting is not None:
                    result = on_submitting(metadata)
                    if inspect.isawaitable(result):
                        await result
            sig = await self.rpc.send_and_confirm(
                txn, on_submitting=record,
                last_valid_block_height=swap.get("lastValidBlockHeight"),
            )
        except TransactionStatusUnknown:
            # The swap was submitted.  The caller must reconcile the
            # signature; it must never submit the same quote again.
            raise
        except TransactionFailed as e:
            logger.error(str(e))
            return False
        if sig:
            logger.success(f"BUY Successful: {token_address} | Tx: {sig}")
            return True
        return False

    async def sell(
        self, token_address: str, percentage: float = 1.0, slippage_bps=500,
        *, on_submitting=None, exclude_dexes=None, expected_raw_balance=None,
    ):
        try:
            if isinstance(percentage, bool):
                raise InvalidOperation("boolean percentage")
            ratio = Decimal(str(percentage))
            if not ratio.is_finite() or not 0 < ratio <= 1:
                raise InvalidOperation("percentage outside (0, 1]")
        except (InvalidOperation, ValueError):
            logger.warning(f"Invalid sell percentage for {token_address}: {percentage!r}")
            return False
        logger.info(f"Executing SELL: {ratio * 100}% of {token_address} -> SOL")
        raw_balance = 0
        decimals = None
        try:
            wallet_pubkey = Pubkey.from_string(ExecutionConfig.get_wallet_address())
            mint_pubkey = Pubkey.from_string(token_address)
            opts = TokenAccountOpts(
                program_id=self.TOKEN_PROGRAM_ID,
                mint=mint_pubkey
            )
            resp = await self.rpc.client.get_token_accounts_by_owner_json_parsed(
                wallet_pubkey,
                opts
            )
            if resp.value:
                for account_info in resp.value:
                    amount_str = account_info.account.data.parsed['info']['tokenAmount']['amount']
                    decimals = int(account_info.account.data.parsed['info']['tokenAmount']['decimals'])
                    raw_balance += int(amount_str)
            logger.info(f"Token Balance Found: {raw_balance} raw units")
            if raw_balance == 0:
                logger.warning(f"No balance found for {token_address}, skipping sell.")
                return False
            if expected_raw_balance is not None and raw_balance != int(expected_raw_balance):
                logger.warning(
                    f"Sell balance changed for {token_address}: expected "
                    f"{expected_raw_balance}, found {raw_balance}; requote before selling."
                )
                return False
            sell_amount = (
                raw_balance if ratio == 1 else
                int((Decimal(raw_balance) * ratio).to_integral_value(rounding=ROUND_DOWN))
            )
            if sell_amount == 0:
                logger.warning("Sell amount is 0 (too small percentage?)")
                return False
        except Exception as e:
            logger.error(f"Failed to fetch token balance: {e}")
            return False
        quote = await self.jup.get_quote(
            input_mint=token_address,
            output_mint=ExecutionConfig.SOL_MINT,
            amount_integer=sell_amount,
            slippage_bps=slippage_bps,
            exclude_dexes=exclude_dexes,
        )
        if not quote:
            logger.error("Sell quote not found.")
            return False
        swap = await self.jup.get_swap_tx(quote, include_metadata=True)
        if (not swap or not swap.get("swapTransaction")
                or not isinstance(swap.get("lastValidBlockHeight"), int)
                or swap["lastValidBlockHeight"] <= 0):
            return False
        try:
            txn = self.jup.deserialize_and_sign(swap["swapTransaction"])
            try:
                async def record(metadata):
                    metadata = {**metadata, "pre_raw_balance": raw_balance,
                                "amount_raw": sell_amount, "decimals": decimals,
                                "quoted_out_lamports": int(quote["outAmount"])}
                    if on_submitting is not None:
                        result = on_submitting(metadata)
                        if inspect.isawaitable(result):
                            await result
                sig = await self.rpc.send_and_confirm(
                    txn, on_submitting=record,
                    last_valid_block_height=swap.get("lastValidBlockHeight"),
                )
            except TransactionStatusUnknown:
                # Preserve the unknown state so the runner does not remove
                # the position or issue a second sell.
                raise
            except TransactionPreflightRejected as error:
                error.route_labels = list(dict.fromkeys(
                    step.get("swapInfo", {}).get("label")
                    for step in quote.get("routePlan", [])
                    if isinstance(step, dict)
                    and isinstance(step.get("swapInfo"), dict)
                    and step["swapInfo"].get("label")
                ))
                raise
            except TransactionFailed as e:
                logger.error(str(e))
                return False
            if sig:
                logger.success(f"SELL Successful: {token_address} | Tx: {sig}")
                return True
        except TransactionStatusUnknown:
            raise
        except TransactionPreflightRejected:
            raise
        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
        return False

    async def close(self):
        await self.rpc.close()
        await self.jup.close()

if __name__ == "__main__": # test
    async def test_run():
        trader = SolanaTrader()
        BONK_ADDRESS = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
        logger.info("Testing Token Balance Fetch...")
        await trader.sell(BONK_ADDRESS, percentage=0.5)
        await trader.close()
    try:
        asyncio.run(test_run())
    except KeyboardInterrupt:
        pass
