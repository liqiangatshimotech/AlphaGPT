import asyncio
import aiohttp
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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
from execution.rpc_handler import TransactionPreflightRejected, TransactionStatusUnknown
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
        self._pipeline_task = None
        self.entries_paused = False
        self.zero_balance_quarantine = set()
        self.zero_balance_first_seen = {}
        self.zero_balance_last_alerted = {}
        self.failed_route_exclusions = {}
        self.failed_route_candidates = {}
        self.stop_signal_path = os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL")
        self.pending_orders_path = os.getenv("PENDING_ORDERS_PATH", "pending_orders.json")
        self.order_history_path = os.getenv("ORDER_HISTORY_PATH", "logs/order_recovery.jsonl")
        self.entry_cooldowns = self._load_entry_cooldowns()
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
                self._handle_stop_signal()

                loop_start = time.time()
                
                if (time.time() - self.last_scan_time > 900
                        and (self._pipeline_task is None or self._pipeline_task.done())):
                    logger.info("o.O | Scheduling Data Pipeline sync...")
                    self._pipeline_task = asyncio.create_task(self._sync_data_pipeline())
                    self.last_scan_time = time.time()

                # A database outage must not disable price-based exits.
                try:
                    self.loader.load_data(
                        limit_tokens=300,
                        max_candle_age_seconds=int(
                            os.getenv("MAX_SIGNAL_CANDLE_AGE_SECONDS", "1800")
                        ),
                    )
                    await self._build_token_mapping()
                except Exception:
                    self.token_map = {}
                    logger.exception("Data refresh failed; monitoring positions without AI signals.")
                    await self._reconcile_pending_orders()
                    await self.monitor_positions()
                    await asyncio.sleep(10 if self.pending_orders else 30)
                    continue

                await self._reconcile_pending_orders()
                await self.monitor_positions()

                if self._handle_stop_signal():
                    logger.warning("New entries paused; position monitoring remains active.")
                elif self._entry_slot_count() < StrategyConfig.MAX_OPEN_POSITIONS:
                    try:
                        await asyncio.wait_for(
                            self.scan_for_entries(),
                            timeout=StrategyConfig.ENTRY_SCAN_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Entry scan timed out; returning to position monitoring.")
                else:
                    logger.info("Max positions reached. Scanning skipped.")

                elapsed = time.time() - loop_start
                sleep_time = 10 if self.pending_orders else max(10, 60 - elapsed)
                logger.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Global Loop Error: {e}")
                await asyncio.sleep(30)

    async def _sync_data_pipeline(self):
        """Keep long Birdeye/DB refreshes off the position-monitoring loop."""
        try:
            await self.data_mgr.pipeline_sync_daily()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background data pipeline sync failed.")

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

    def _load_entry_cooldowns(self):
        """Restore confirmed stop-loss exclusions from the durable order journal."""
        cooldowns = {}
        now = time.time()
        try:
            with open(self.order_history_path) as history:
                for line in history:
                    try:
                        order = json.loads(line)
                        if (order.get("side") != "sell"
                                or order.get("reason") != "StopLoss"
                                or order.get("outcome") != "confirmed"):
                            continue
                        token = order.get("token_address")
                        expiry = float(order["resolved_at"]) + StrategyConfig.STOP_LOSS_COOLDOWN_SECONDS
                        if token and expiry > now:
                            cooldowns[token] = max(cooldowns.get(token, 0), expiry)
                    except (TypeError, ValueError, KeyError):
                        logger.warning("Skipping malformed stop-loss cooldown journal row.")
        except FileNotFoundError:
            pass
        logger.info(f"Restored {len(cooldowns)} active stop-loss cooldowns.")
        return cooldowns

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
        if (outcome == "confirmed" and order.get("side") == "sell"
                and order.get("reason") == "StopLoss"):
            self.entry_cooldowns[token_addr] = max(
                self.entry_cooldowns.get(token_addr, 0),
                time.time() + StrategyConfig.STOP_LOSS_COOLDOWN_SECONDS,
            )
        if outcome == "confirmed" and order.get("side") == "sell":
            getattr(self, "failed_route_exclusions", {}).pop(token_addr, None)
            getattr(self, "failed_route_candidates", {}).pop(token_addr, None)
        self.pending_orders.pop(token_addr, None)
        try:
            self._save_pending_orders()
        except Exception:
            self.pending_orders[token_addr] = order
            raise
        logger.info(f"Pending {order['side']} resolved: {token_addr} | {outcome}")

    async def _release_preflight_rejection(self, token_addr, error):
        """Release a rejected one-shot send only after signature and balance checks."""
        order = self.pending_orders.get(token_addr)
        if (order is None or order.get("signature") != error.signature
                or not error.single_attempt_proven):
            logger.warning(f"Preflight rejection retained for reconciliation: {token_addr}")
            return False
        state, detail = await self.trader.rpc.get_signature_status(error.signature)
        if state != "pending" or detail is not None:
            logger.warning(f"Preflight signature not proven absent: {token_addr} ({state})")
            return False
        raw_balance = await self.trader.rpc.get_token_balance(
            token_addr, commitment="finalized"
        )
        before = order.get("pre_raw_balance")
        if raw_balance is None or before is None or raw_balance != int(before):
            logger.warning(f"Preflight balance not proven unchanged: {token_addr}")
            return False
        self._resolve_pending(
            token_addr, order, "preflight_rejected",
            {"signature_absent": True, "raw_balance": raw_balance,
             "simulation_error": str(error.simulation_error),
             "failed_program_id": error.failed_program_id},
        )
        return True

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
                                        cost / amount, amount, cost,
                                        mint_decimals=int(decimals))
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
        evidence = {"raw_balance": raw}
        delta_reader = getattr(self.trader.rpc, "get_wallet_transaction_delta", None)
        if delta_reader is not None:
            try:
                settlement = await asyncio.wait_for(delta_reader(order["signature"]), timeout=3.0)
            except Exception as error:
                logger.warning(f"Sell settlement details unavailable for {order['signature']}: {error}")
                settlement = None
            if settlement is not None:
                evidence.update(settlement)
        if pos is not None:
            evidence["initial_cost_sol"] = pos.initial_cost_sol
            evidence["entry_price_sol"] = pos.entry_price
        if pos is not None:
            if raw == 0:
                self.portfolio.close_position(token_addr)
            else:
                pos.amount_held = raw / (10 ** int(decimals))
                if order.get("reason") == "Moonbag":
                    pos.is_moonbag = True
                self.portfolio.save_state()
        self._resolve_pending(token_addr, order, "confirmed", evidence)
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
            try:
                await self._monitor_one_position(token_addr, pos)
            except Exception:
                logger.exception(f"Position monitoring failed for {pos.symbol}; continuing other exits.")

    async def _monitor_one_position(self, token_addr, pos):
        if token_addr in self.pending_orders:
            return
        try:
            raw_balance = await asyncio.wait_for(
                self.trader.rpc.get_token_balance(token_addr, commitment="confirmed"),
                timeout=StrategyConfig.POSITION_BALANCE_TIMEOUT_SECONDS,
            )
        except Exception as error:
            logger.warning(f"Token balance read failed for {pos.symbol}: {error}")
            raw_balance = None
        used_local_balance = raw_balance is None
        if raw_balance is None:
            # Continue evaluating a possible stop with the persisted size. The
            # trader must later confirm that exact raw balance before sending.
            try:
                decimals = await asyncio.wait_for(
                    self._position_mint_decimals(token_addr),
                    timeout=StrategyConfig.POSITION_BALANCE_TIMEOUT_SECONDS,
                )
                amount = Decimal(str(pos.amount_held))
                if (not isinstance(decimals, int) or not 0 <= decimals <= 18
                        or not amount.is_finite() or amount <= 0):
                    raise ValueError("Invalid persisted token amount or decimals")
                raw_balance = int(
                    (amount * (10 ** decimals)).to_integral_value(rounding=ROUND_HALF_UP)
                )
                if raw_balance <= 0:
                    raise ValueError("Persisted token amount rounds to zero")
            except (Exception, InvalidOperation) as error:
                logger.warning(f"Cannot derive fallback balance for {pos.symbol}: {error}")
                return
            logger.warning(f"Using persisted size to evaluate stop for {pos.symbol}; chain balance unavailable.")
        if raw_balance == 0:
            final_balance = await asyncio.wait_for(
                self.trader.rpc.get_token_balance(token_addr, commitment="finalized"),
                timeout=StrategyConfig.POSITION_BALANCE_TIMEOUT_SECONDS,
            )
            if final_balance == 0:
                # Preserve the persisted position: two reads from one RPC
                # provider cannot independently prove that it was sold or
                # transferred. Quarantine it from repeated sell attempts.
                quarantined = getattr(self, "zero_balance_quarantine", set())
                first_seen = getattr(self, "zero_balance_first_seen", {})
                last_alerted = getattr(self, "zero_balance_last_alerted", {})
                now = time.time()
                first_seen.setdefault(token_addr, now)
                self.zero_balance_first_seen = first_seen
                if (token_addr not in quarantined
                        or now - last_alerted.get(token_addr, 0)
                        >= StrategyConfig.ZERO_BALANCE_ALERT_INTERVAL_SECONDS):
                    logger.error(
                        f"Zero on-chain balance for {pos.symbol} ({token_addr}) at "
                        "confirmed and finalized commitment; retaining position "
                        f"for audit (observed {now - first_seen[token_addr]:.0f}s)."
                    )
                    last_alerted[token_addr] = now
                    self.zero_balance_last_alerted = last_alerted
                quarantined.add(token_addr)
                self.zero_balance_quarantine = quarantined
            else:
                logger.warning(f"Zero balance for {pos.symbol} not yet finalized; deferring.")
            return
        quarantined = getattr(self, "zero_balance_quarantine", set())
        if token_addr in quarantined:
            quarantined.remove(token_addr)
            getattr(self, "zero_balance_first_seen", {}).pop(token_addr, None)
            getattr(self, "zero_balance_last_alerted", {}).pop(token_addr, None)
            logger.info(f"On-chain balance restored for {pos.symbol}; resuming monitoring.")

        price_kwargs = {"raw_balance": raw_balance}
        if used_local_balance:
            price_kwargs["decimals"] = decimals
        exit_price = await asyncio.wait_for(
            self._fetch_live_price_sol(token_addr, **price_kwargs),
            timeout=StrategyConfig.POSITION_QUOTE_TIMEOUT_SECONDS,
        )
        if exit_price <= 0:
            logger.warning(f"Could not fetch full exit quote for {pos.symbol}, skipping.")
            return

        exit_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        if exit_pnl_pct <= StrategyConfig.STOP_LOSS_PCT:
            logger.warning(f"!!! | STOP LOSS: {pos.symbol} Full-exit PnL: {exit_pnl_pct:.2%}")
            if used_local_balance:
                await self._execute_sell_bounded(
                    token_addr, 1.0, "StopLoss", expected_raw_balance=raw_balance
                )
            else:
                await self._execute_sell_bounded(token_addr, 1.0, "StopLoss")
            return

        if used_local_balance:
            logger.warning(f"Deferring non-stop exits for {pos.symbol} until chain balance returns.")
            return

        if not pos.is_moonbag and exit_pnl_pct >= StrategyConfig.TAKE_PROFIT_Target1:
            logger.success(f"😄 | MOONBAG TP: {pos.symbol} Full-exit PnL: {exit_pnl_pct:.2%}")
            sold = await self._execute_sell_bounded(
                token_addr, StrategyConfig.TP_Target1_Ratio, "Moonbag"
            )
            if sold and token_addr in self.portfolio.positions:
                pos.is_moonbag = True
                self.portfolio.save_state()
            return

        if pos.highest_price_basis == "full_exit":
            current_price = exit_price
            if not pos.highest_price_initialized:
                pos.highest_price = current_price
                pos.highest_price_initialized = True
                self.portfolio.save_state()
            else:
                self.portfolio.update_price(token_addr, current_price)
        elif pos.highest_price_basis == "spot":
            # Existing positions retain their historical one-token basis.
            current_price = await asyncio.wait_for(
                self._fetch_live_price_sol(token_addr),
                timeout=StrategyConfig.POSITION_QUOTE_TIMEOUT_SECONDS,
            )
            if current_price <= 0:
                logger.warning(f"Could not fetch spot quote for {pos.symbol}; trailing exit deferred.")
                return
            self.portfolio.update_price(token_addr, current_price)
        else:
            logger.error(f"Unknown high-water quote basis for {pos.symbol}; trailing exit deferred.")
            return
        max_gain = (pos.highest_price - pos.entry_price) / pos.entry_price
        drawdown = (pos.highest_price - current_price) / pos.highest_price

        if max_gain > StrategyConfig.TRAILING_ACTIVATION and drawdown > StrategyConfig.TRAILING_DROP:
            # A gap can overrun the configured drawdown between monitor passes.
            # Trailing is still a risk exit even if the current PnL fell below zero.
            logger.warning(
                f"😠 | TRAILING STOP: {pos.symbol} Max: {max_gain:.2%} DD: {drawdown:.2%} "
                f"Full-exit PnL: {exit_pnl_pct:.2%}"
            )
            await self._execute_sell_bounded(token_addr, 1.0, "TrailingStop")
            return

        if not pos.is_moonbag:
            ai_score = await self._run_inference(token_addr)
            if ai_score != -1 and ai_score < StrategyConfig.SELL_THRESHOLD:
                logger.info(f"🤖 | AI EXIT: {pos.symbol} Score: {ai_score:.2f}")
                await self._execute_sell_bounded(token_addr, 1.0, "AI_Signal")

    async def _execute_sell_bounded(self, token_addr, ratio, reason, *, expected_raw_balance=None):
        # A canceled send still has its signed identity in the pending journal;
        # the next cycle reconciles it before attempting another exit.
        kwargs = ({"expected_raw_balance": expected_raw_balance}
                  if expected_raw_balance is not None else {})
        return await asyncio.wait_for(
            self._execute_sell(token_addr, ratio, reason, **kwargs),
            timeout=StrategyConfig.POSITION_EXIT_TIMEOUT_SECONDS,
        )

    async def scan_for_entries(self):
        if self._handle_stop_signal():
            return

        if self._entry_slot_count() >= StrategyConfig.MAX_OPEN_POSITIONS:
            return

        raw_signals = self.vm.execute(self.formula, self.loader.feat_tensor)
        
        if raw_signals is None: return

        latest_signals = raw_signals[:, -1]
        if not torch.isfinite(latest_signals).all().item():
            logger.error("Entry scan blocked: model emitted non-finite scores.")
            return
        score_tensor = torch.sigmoid(latest_signals)
        liquidity = self.loader.raw_data_cache["liquidity"][:, -1]
        if liquidity.shape != score_tensor.shape:
            logger.error("Entry scan blocked: score and liquidity arrays differ.")
            return
        eligible = torch.isfinite(liquidity) & (
            liquidity >= StrategyConfig.MIN_ENTRY_LIQUIDITY_USD
        )
        eligible_scores = score_tensor[eligible]
        if eligible_scores.numel() < 2:
            logger.error("Entry scan blocked: fewer than two liquid candidates.")
            return
        if (eligible_scores.max() - eligible_scores.min()).item() < 0.01:
            logger.error(
                "Entry scan blocked: model scores cannot separate candidates "
                f"(count={eligible_scores.numel()}, min={eligible_scores.min().item():.6f}, "
                f"max={eligible_scores.max().item():.6f})."
            )
            return
        if (eligible_scores >= StrategyConfig.BUY_THRESHOLD).sum().item() > eligible_scores.numel() / 2:
            logger.error("Entry scan blocked: model buys more than half the liquid universe.")
            return
        scores = score_tensor.cpu().numpy() # 转为概率 0~1
        
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

        if not await self.risk.check_entry_round_trip(
            token_addr, quote, input_lamports=amount_lamports,
            quote_provider=self.trader.jup,
        ):
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
        except TransactionPreflightRejected as exc:
            await self._release_preflight_rejection(token_addr, exc)
            return False
        except TransactionStatusUnknown as exc:
            logger.warning(f"BUY still pending: {token_addr} | {exc.signature}")
            return False
        await self._reconcile_pending_orders(token_addr)
        return bool(success and token_addr not in self.pending_orders
                    and token_addr in self.portfolio.positions)

    async def _execute_sell(self, token_addr, ratio, reason, *, expected_raw_balance=None):
        pos = self.portfolio.positions.get(token_addr)
        if pos is None:
            return False
        if token_addr in self.pending_orders:
            logger.warning(f"Sell skipped; token already has a pending transaction: {token_addr}")
            return False
        logger.info(f"- | EXECUTING SELL: {token_addr} | Ratio: {ratio:.0%} | Reason: {reason}")

        def record(metadata):
            self._record_pending_sell(token_addr, metadata["signature"], ratio, reason, metadata)
        route_cache = getattr(self, "failed_route_exclusions", {})
        route_candidates = getattr(self, "failed_route_candidates", {})
        cached = route_cache.get(token_addr)
        excluded_dexes = None
        if cached is not None:
            label, expiry = cached
            if time.time() < expiry:
                excluded_dexes = [label]
            else:
                route_cache.pop(token_addr, None)
        if excluded_dexes is None:
            candidate = route_candidates.get(token_addr)
            if candidate is not None:
                program_id, labels, expiry = candidate
                if time.time() < expiry:
                    label = await self._failed_dex_label(program_id)
                    if label in labels:
                        excluded_dexes = [label]
                        route_cache[token_addr] = (label, expiry)
                        self.failed_route_exclusions = route_cache
                        route_candidates.pop(token_addr, None)
                else:
                    route_candidates.pop(token_addr, None)
        for attempt in range(2):
            try:
                success = await self.trader.sell(
                    token_addr, percentage=ratio, on_submitting=record,
                    exclude_dexes=excluded_dexes,
                    expected_raw_balance=expected_raw_balance,
                )
            except TransactionPreflightRejected as exc:
                labels = getattr(exc, "route_labels", [])
                if exc.single_attempt_proven and exc.failed_program_id and labels:
                    route_candidates[token_addr] = (
                        exc.failed_program_id, tuple(labels),
                        time.time() + StrategyConfig.FAILED_DEX_EXCLUSION_SECONDS,
                    )
                    self.failed_route_candidates = route_candidates
                if not await self._release_preflight_rejection(token_addr, exc):
                    return False
                if attempt == 0 and labels and exc.failed_program_id:
                    failed_label = await self._failed_dex_label(exc.failed_program_id)
                    if failed_label in labels:
                        excluded_dexes = [failed_label]
                        route_cache[token_addr] = (
                            failed_label,
                            time.time() + StrategyConfig.FAILED_DEX_EXCLUSION_SECONDS,
                        )
                        self.failed_route_exclusions = route_cache
                        route_candidates.pop(token_addr, None)
                        logger.warning(
                            f"Requoting rejected exit for {token_addr} without DEX {failed_label}"
                        )
                        continue
                return False
            except TransactionStatusUnknown as exc:
                logger.warning(f"SELL still pending: {token_addr} | {exc.signature}")
                return False
            await self._reconcile_pending_orders(token_addr)
            return bool(success and token_addr not in self.pending_orders)
        return False

    async def _failed_dex_label(self, program_id):
        # Unknown within the short budget means "don't exclude"; the route
        # candidate stays cached so a later exit can still map it.
        try:
            return await asyncio.wait_for(
                self.trader.jup.get_program_id_label(program_id),
                timeout=StrategyConfig.DEX_LABEL_LOOKUP_TIMEOUT_SECONDS,
            )
        except Exception as error:
            logger.warning(f"DEX label unavailable for rejected exit: {error!r}")
            return None

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

    async def _position_mint_decimals(self, token_addr):
        pos = self.portfolio.positions.get(token_addr)
        cached = getattr(pos, "mint_decimals", None)
        if type(cached) is int and 0 <= cached <= 18:
            return cached
        decimals = await get_mint_decimals(token_addr, self.trader.rpc.client)
        if pos is not None:
            pos.mint_decimals = decimals
            self.portfolio.save_state()
        return decimals

    async def _fetch_live_price_sol(self, token_addr, *, raw_balance=None, decimals=None):
        """Return per-token SOL value for one token or an exact full-size exit."""
        for attempt in range(2):
            try:
                if decimals is None:
                    decimals = await self._position_mint_decimals(token_addr)
                if raw_balance is None:
                    raw_balance = 10 ** decimals
                if raw_balance is None or raw_balance <= 0:
                    return 0.0
                amount_held = raw_balance / (10 ** decimals)
                quote = await self.trader.jup.get_quote(
                    input_mint=token_addr,
                    output_mint=self.trader.config.SOL_MINT,
                    amount_integer=raw_balance,
                )
                if quote and int(quote["outAmount"]) > 0:
                    return int(quote["outAmount"]) / 1e9 / amount_held
            except (aiohttp.ClientSSLError, aiohttp.ServerFingerprintMismatch):
                raise
            except Exception as error:
                logger.warning(f"Full-position price fetch failed for {token_addr}: {error}")
            if attempt == 0:
                await asyncio.sleep(0.25)
        return 0.0

    async def shutdown(self):
        logger.info("O.o | Shutting down strategy runner...")
        pipeline_task = getattr(self, "_pipeline_task", None)
        if pipeline_task is not None and not pipeline_task.done():
            pipeline_task.cancel()
            await asyncio.gather(pipeline_task, return_exceptions=True)
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
