import asyncio
import torch
import json
import os
import time
from loguru import logger

from data_pipeline.data_manager import DataManager
from model_core.vm import StackVM
from model_core.data_loader import CryptoDataLoader
from model_core.vocab import load_formula
from execution.trader import SolanaTrader
from execution.rpc_handler import TransactionStatusUnknown
from execution.utils import get_mint_decimals
from .config import StrategyConfig
from .portfolio import PortfolioManager
from .risk import RiskEngine

class StrategyRunner:
    def __init__(self):
        self.data_mgr = DataManager()
        self.portfolio = PortfolioManager()
        self.risk = RiskEngine()
        self.trader = SolanaTrader()
        self.vm = StackVM()

        self.loader = CryptoDataLoader()
        self.token_map = {} # {address: tensor_index} 用于快速查找特征
        self.last_scan_time = 0
        self.entries_paused = False
        self.entry_cooldowns = {}
        self.stop_signal_path = os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL")
        self.pending_orders_path = os.getenv("PENDING_ORDERS_PATH", "pending_orders.json")
        self.order_history_path = os.getenv("ORDER_HISTORY_PATH", "logs/order_recovery.jsonl")
        self.pending_orders = self._load_pending_orders()
        
        try:
            with open("best_meme_strategy.json", "r") as f:
                data = json.load(f)
                self.formula = load_formula(data)
                if isinstance(data, list) or "vocab_version" not in data:
                    logger.warning("Unversioned formula loaded with original VM token semantics.")
            logger.success(f"Loaded Strategy: {self.formula}")
            logger.info(
                f"Live sizing: max_positions={StrategyConfig.MAX_OPEN_POSITIONS}, "
                f"entry_amount_sol={StrategyConfig.ENTRY_AMOUNT_SOL:.4f}, "
                f"buy_threshold={StrategyConfig.BUY_THRESHOLD:.3f}"
            )
        except FileNotFoundError:
            logger.critical("Strategy file not found! Please train model first.")
            exit(1)

    async def initialize(self):
        # Reconcile and monitor persisted positions before the data pipeline
        # warms up.  This keeps exits available during a launchd handover.
        await self._reconcile_pending_orders()
        await self.monitor_positions()
        await self.data_mgr.initialize()
        bal = await self.trader.rpc.get_balance()
        if bal is None:
            logger.warning("Bot Initialized, but wallet balance is currently unavailable.")
        else:
            logger.info(f"Bot Initialized. Wallet Balance: {bal:.4f} SOL")

    async def run_loop(self):
        logger.info(">_< | Strategy Runner Started (Live Mode)")
        
        while True:
            try:
                stop_requested = self._handle_stop_signal()

                loop_start = time.time()
                
                if not stop_requested and time.time() - self.last_scan_time > 900: # 15 min
                    logger.info("o.O | Syncing Data Pipeline...")
                    await self.data_mgr.pipeline_sync_daily()
                    self.last_scan_time = time.time()

                # A database outage must not disable price-based exits.
                try:
                    self.loader.load_data(limit_tokens=300)
                    await self._build_token_mapping()
                except Exception:
                    self.token_map = {}
                    logger.exception("Data refresh failed; monitoring positions without AI signals.")
                    await self._reconcile_pending_orders()
                    await self.monitor_positions()
                    await asyncio.sleep(30)
                    continue

                await self._reconcile_pending_orders()
                await self.monitor_positions()

                if self._handle_stop_signal():
                    logger.warning("New entries paused; position monitoring remains active.")
                elif self._entry_slot_count() < StrategyConfig.MAX_OPEN_POSITIONS:
                    await self.scan_for_entries()
                else:
                    logger.info("Max positions reached. Scanning skipped.")

                elapsed = time.time() - loop_start
                sleep_time = max(10, 60 - elapsed)
                logger.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Global Loop Error: {e}")
                await asyncio.sleep(30)

    def _stop_requested(self):
        try:
            with open(self.stop_signal_path, "r") as f:
                signal = f.read().strip().upper()
        except FileNotFoundError:
            return False
        except (OSError, UnicodeError):
            return True
        return signal in {"", "STOP", "STOPPED"}

    def _handle_stop_signal(self):
        paused = self._stop_requested()
        if paused != self.entries_paused:
            logger.warning(
                "New entries paused; existing positions remain monitored."
                if paused else "Stop signal cleared; new entries enabled."
            )
            self.entries_paused = paused
        return paused

    async def _build_token_mapping(self):
        self.token_map = {addr: idx for idx, addr in enumerate(self.loader.addresses)}
        logger.info(f"Mapped {len(self.token_map)} tokens for inference.")

    def _entry_slot_count(self):
        """Reserve a slot for each submitted buy until chain reconciliation."""
        pending_buys = {
            token for token, order in self.pending_orders.items()
            if order.get("side") == "buy"
        }
        return len(set(self.portfolio.positions) | pending_buys)

    def _load_pending_orders(self):
        try:
            with open(self.pending_orders_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Pending journal must be an object")
        except FileNotFoundError:
            return {}
        # Malformed journals must not be treated as no outstanding orders.
        # This file is only read at startup; current service stays running
        # throughout deployment validation.
        recovery_path = os.getenv("LEGACY_PENDING_RECOVERY_PATH", "pending_recovery.json")
        try:
            with open(recovery_path) as f:
                recoveries = json.load(f)
        except FileNotFoundError:
            recoveries = {}
        for token, order in data.items():
            recovery = recoveries.get(token, {})
            if (recovery.get("signature") == order.get("signature")
                    and recovery.get("kind") == "audited_legacy_jupiter_v1"
                    and not order.get("recent_blockhash")
                    and not order.get("last_valid_block_height")):
                # This is an audited, conservative upper bound for this one
                # legacy ordinary swap, NOT its original blockhash expiry.
                order.update({k: recovery[k] for k in (
                    "last_valid_block_height", "pre_raw_balance", "amount_raw",
                    "decimals", "recovery_evidence",
                )})
        return data

    def _save_pending_orders(self):
        tmp_path = f"{self.pending_orders_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.pending_orders, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.pending_orders_path)

    def _archive_order(self, token_addr, order, outcome, evidence):
        path = getattr(self, "order_history_path", self.pending_orders_path + ".history.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = {**order, "token_address": token_addr, "outcome": outcome,
                  "resolved_at": time.time(), "evidence": evidence}
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _resolve_pending(self, token_addr, order, outcome, evidence):
        self._archive_order(token_addr, order, outcome, evidence)
        self.pending_orders.pop(token_addr, None)
        try:
            self._save_pending_orders()
        except Exception:
            self.pending_orders[token_addr] = order
            raise
        logger.info(f"Pending {order['side']} resolved: {token_addr} | {outcome}")

    def _record_pending_order(self, token_addr, signature, amount_sol, expected_out, score, metadata=None):
        self.pending_orders[token_addr] = {
            "side": "buy", "token_address": token_addr,
            "signature": str(signature), "amount_sol": float(amount_sol),
            "expected_out": int(expected_out), "score": float(score),
            "created_at": time.time(), **(metadata or {}),
        }
        self._save_pending_orders()
        logger.info(f"BUY journaled before submission: {token_addr} | {signature}")

    def _record_pending_sell(self, token_addr, signature, ratio, reason, metadata=None):
        self.pending_orders[token_addr] = {
            "side": "sell", "token_address": token_addr,
            "signature": str(signature), "ratio": float(ratio), "reason": reason,
            "created_at": time.time(), **(metadata or {}),
        }
        self._save_pending_orders()
        logger.info(f"SELL journaled before submission: {token_addr} | {signature}")

    async def _finalize_pending_order(self, token_addr, order):
        raw = await self.trader.rpc.get_token_balance(token_addr, commitment="confirmed")
        before = order.get("pre_raw_balance")
        decimals = order.get("decimals")
        if raw is None or before is None or decimals is None or raw <= before:
            return False
        amount = raw / (10 ** int(decimals))
        if token_addr not in self.portfolio.positions:
            cost = float(order["amount_sol"])
            self.portfolio.add_position(token_addr, f"Meme_{token_addr[:4]}",
                                        cost / amount, amount, cost)
        else:
            self.portfolio.update_holding(token_addr, amount)
        self._resolve_pending(token_addr, order, "confirmed", {"raw_balance": raw})
        return True

    async def _finalize_pending_sell(self, token_addr, order):
        # Set the actual balance, never subtract the percentage again after
        # restart.  Persist the take-profit flag in the same portfolio write.
        raw = await self.trader.rpc.get_token_balance(token_addr, commitment="confirmed")
        before, amount = order.get("pre_raw_balance"), order.get("amount_raw")
        decimals = order.get("decimals")
        if (raw is None or before is None or amount is None or decimals is None
                or raw > int(before) - int(amount)):
            return False
        pos = self.portfolio.positions.get(token_addr)
        if pos is not None:
            if raw == 0:
                self.portfolio.close_position(token_addr)
            else:
                pos.amount_held = raw / (10 ** int(decimals))
                if order.get("reason") == "Moonbag":
                    pos.is_moonbag = True
                self.portfolio.save_state()
        self._resolve_pending(token_addr, order, "confirmed", {"raw_balance": raw})
        return True

    async def _reconcile_pending_orders(self, token_addr=None):
        if token_addr is None:
            # Limit the entire recovery pass so an unhealthy RPC cannot
            # indefinitely delay monitoring of other positions.
            deadline = asyncio.get_running_loop().time() + 8.0
            for token in list(self.pending_orders):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._reconcile_pending_orders(token), remaining)
                except asyncio.TimeoutError:
                    logger.warning("Pending recovery budget exhausted; continuing position monitoring.")
                    break
            return
        for token, order in list(self.pending_orders.items()):
            if token_addr is not None and token != token_addr:
                continue
            try:
                signature = order.get("signature")
                if not signature:
                    logger.error(f"Pending order missing signature; retained: {token}")
                    continue
                state, evidence = await self.trader.rpc.get_expiry_evidence(
                    signature, recent_blockhash=order.get("recent_blockhash"),
                    last_valid_block_height=order.get("last_valid_block_height"),
                )
                if state == "confirmed":
                    if order.get("side") == "sell":
                        await self._finalize_pending_sell(token, order)
                    else:
                        await self._finalize_pending_order(token, order)
                elif state in {"failed", "expired"}:
                    raw = await self.trader.rpc.get_token_balance(
                        token, commitment="finalized",
                        min_context_slot=(evidence or {}).get("expiry_slot"),
                    )
                    # Clear only after definitive failure/expiry AND an
                    # unchanged finalized balance.  RPC errors are not zero.
                    before = order.get("pre_raw_balance")
                    if raw is not None and before is not None and raw == int(before):
                        self._resolve_pending(token, order, state,
                                              {**(evidence or {}), "raw_balance": raw})
                        logger.warning(f"Exit/entry unblocked after {state}: {token}; re-evaluate live strategy.")
                    else:
                        logger.warning(f"Pending {state} but balance needs reconciliation: {token}")
                else:
                    logger.info(f"Pending status {state}; retained: {token}")
            except Exception as exc:
                # One unavailable token must not stop exits for other tokens.
                logger.warning(f"Pending reconciliation deferred: {token} ({type(exc).__name__})")

    async def monitor_positions(self):
        if not self.portfolio.positions: return

        logger.info(f"o.O | Monitoring {len(self.portfolio.positions)} positions...")
        
        for token_addr, pos in list(self.portfolio.positions.items()):
            current_price = await self._fetch_live_price_sol(token_addr)
            if current_price <= 0:
                logger.warning(f"Could not fetch price for {pos.symbol}, skipping.")
                continue

            self.portfolio.update_price(token_addr, current_price)

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            
            if pnl_pct <= StrategyConfig.STOP_LOSS_PCT:
                logger.warning(f"!!! | STOP LOSS: {pos.symbol} PnL: {pnl_pct:.2%}")
                await self._execute_sell(token_addr, 1.0, "StopLoss")
                self.entry_cooldowns[token_addr] = (
                    time.time() + StrategyConfig.STOP_LOSS_COOLDOWN_SECONDS
                )
                continue

            if not pos.is_moonbag and pnl_pct >= StrategyConfig.TAKE_PROFIT_Target1:
                logger.success(f"😄 | MOONBAG TP: {pos.symbol} PnL: {pnl_pct:.2%}")
                sold = await self._execute_sell(token_addr, StrategyConfig.TP_Target1_Ratio, "Moonbag")
                if sold and token_addr in self.portfolio.positions:
                    pos.is_moonbag = True
                    self.portfolio.save_state()
                continue

            max_gain = (pos.highest_price - pos.entry_price) / pos.entry_price
            drawdown = (pos.highest_price - current_price) / pos.highest_price
            
            if max_gain > StrategyConfig.TRAILING_ACTIVATION and drawdown > StrategyConfig.TRAILING_DROP:
                logger.warning(f"😠 | TRAILING STOP: {pos.symbol} Max: {max_gain:.2%} DD: {drawdown:.2%}")
                await self._execute_sell(token_addr, 1.0, "TrailingStop")
                continue

            if not pos.is_moonbag:
                ai_score = await self._run_inference(token_addr)
                if ai_score != -1 and ai_score < StrategyConfig.SELL_THRESHOLD:
                    logger.info(f"🤖 | AI EXIT: {pos.symbol} Score: {ai_score:.2f}")
                    await self._execute_sell(token_addr, 1.0, "AI_Signal")

    async def scan_for_entries(self):
        if self._handle_stop_signal():
            return

        if self._entry_slot_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
            return

        raw_signals = self.vm.execute(self.formula, self.loader.feat_tensor)
        
        if raw_signals is None: return

        latest_signals = raw_signals[:, -1]
        scores = torch.sigmoid(latest_signals).cpu().numpy() # 转为概率 0~1
        
        # 翻转排序，从高分到低分处理
        sorted_indices = scores.argsort()[::-1]
        
        # 反向查表：Index -> Address
        # (效率较低，但 Top 300 没关系)
        idx_to_addr = {v: k for k, v in self.token_map.items()}
        
        for idx in sorted_indices:
            if self._entry_slot_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
                break
            score = float(scores[idx])
            
            if score < StrategyConfig.BUY_THRESHOLD:
                break # 后面的都不够分，不用看了
                
            token_addr = idx_to_addr.get(idx)
            if not token_addr: continue

            cooldown_until = self.entry_cooldowns.get(token_addr)
            if cooldown_until is not None:
                if time.time() < cooldown_until:
                    continue
                self.entry_cooldowns.pop(token_addr, None)
            
            # 过滤已持仓
            if token_addr in self.portfolio.positions or token_addr in self.pending_orders:
                continue
            
            # 从 loader 缓存获取该 Token 的最新流动性
            # raw_data_cache['liquidity']: [Tokens, Time]
            liq_usd = self.loader.raw_data_cache['liquidity'][idx, -1].item()
            
            logger.info(f"🔍 | Inspecting {token_addr} | Score: {score:.2f} | Liq: ${liq_usd:.0f}")
            
            is_safe = await self.risk.check_safety(token_addr, liq_usd)
            if is_safe:
                if self._handle_stop_signal():
                    return
                buy_success = await self._execute_buy(token_addr, score)
                
                # 检查仓位上限
                if self._entry_slot_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
                    break
                # A failed live order usually indicates a quote/API/RPC issue.
                # Stop this scan and retry from a fresh market snapshot next
                # cycle instead of hammering the execution API with every
                # remaining candidate.
                if not buy_success:
                    return

    async def _execute_buy(self, token_addr, score):
        if self._handle_stop_signal():
            logger.warning("Buy skipped because STOP signal is active.")
            return False

        if self._entry_slot_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
            logger.warning("Buy skipped because existing and pending buys fill all position slots.")
            return False

        if token_addr in self.pending_orders:
            logger.warning(f"Buy skipped; token already has a pending transaction: {token_addr}")
            return False

        balance = await self.trader.rpc.get_balance()
        if balance is None:
            logger.warning("Entry deferred because wallet balance is unavailable from RPC.")
            return False
        amount_sol = self.risk.calculate_position_size(balance)
        
        if amount_sol <= 0:
            logger.warning("Insufficient balance for new entry.")
            return False

        logger.info(f"🎉 | EXECUTING BUY: {token_addr} | Amt: {amount_sol} SOL")
        
        amount_lamports = int(amount_sol * 1e9)
        quote = await self.trader.jup.get_quote(
            input_mint=self.trader.config.SOL_MINT,
            output_mint=token_addr,
            amount_integer=amount_lamports
        )
        
        if not quote:
            logger.error("Failed to get quote for buy.")
            return False

        if self._handle_stop_signal():
            logger.warning("Buy cancelled because STOP signal became active.")
            return False

        def record(metadata):
            self._record_pending_order(
                token_addr, metadata["signature"], amount_sol,
                quote["outAmount"], score, metadata,
            )
        try:
            success = await self.trader.buy(
                token_addr, amount_sol, should_cancel=self._handle_stop_signal,
                quote_response=quote, on_submitting=record,
            )
        except TransactionStatusUnknown as exc:
            logger.warning(f"BUY still pending: {token_addr} | {exc.signature}")
            return False
        await self._reconcile_pending_orders(token_addr)
        return bool(success and token_addr not in self.pending_orders
                    and token_addr in self.portfolio.positions)

    async def _execute_sell(self, token_addr, ratio, reason):
        pos = self.portfolio.positions.get(token_addr)
        if pos is None:
            return False
        if token_addr in self.pending_orders:
            logger.warning(f"Sell skipped; token already has a pending transaction: {token_addr}")
            return False
        logger.info(f"- | EXECUTING SELL: {token_addr} | Ratio: {ratio:.0%} | Reason: {reason}")

        def record(metadata):
            self._record_pending_sell(token_addr, metadata["signature"], ratio, reason, metadata)
        try:
            success = await self.trader.sell(token_addr, percentage=ratio, on_submitting=record)
        except TransactionStatusUnknown as exc:
            logger.warning(f"SELL still pending: {token_addr} | {exc.signature}")
            return False
        await self._reconcile_pending_orders(token_addr)
        return bool(success and token_addr not in self.pending_orders)

    async def _run_inference(self, token_addr):
        idx = self.token_map.get(token_addr)
        if idx is None:
            return -1

        features = self.loader.feat_tensor[idx] # 此时是 2D Tensor
        
        features_batch = features.unsqueeze(0) # [1, F, T]
        
        res = self.vm.execute(self.formula, features_batch) # -> [1, Time]
        
        if res is None: return -1
        
        latest_logit = res[0, -1]
        score = torch.sigmoid(latest_logit).item()
        return score

    async def _fetch_live_price_sol(self, token_addr):
        try:
            # 1. 获取精度
            decimals = await get_mint_decimals(token_addr, self.trader.rpc.client)
            amount_1_unit = 10 ** decimals
            
            # 2. 询价: 1 Token -> ? SOL
            quote = await self.trader.jup.get_quote(
                input_mint=token_addr,
                output_mint=self.trader.config.SOL_MINT,
                amount_integer=amount_1_unit
            )
            
            if quote:
                out_lamports = int(quote['outAmount'])
                price_sol = out_lamports / 1e9
                return price_sol
            
        except Exception as e:
            logger.warning(f"Price fetch failed for {token_addr}: {e}")
        
        return 0.0

    async def shutdown(self):
        logger.info("O.o | Shutting down strategy runner...")
        await self.data_mgr.close()
        await self.trader.close()
        await self.risk.close()

if __name__ == "__main__":
    runner = StrategyRunner()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(runner.initialize())
        loop.run_until_complete(runner.run_loop())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.shutdown())
