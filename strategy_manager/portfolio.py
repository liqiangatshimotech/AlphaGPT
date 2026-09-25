import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict
from loguru import logger

@dataclass
class Position:
    token_address: str
    symbol: str
    entry_price: float     # 入场价格 (USD 或 SOL)
    entry_time: float      # 入场时间戳
    amount_held: float     # 当前持仓数量 (Token Units)
    initial_cost_sol: float # 初始投入 SOL
    highest_price: float   #以此计算回撤
    is_moonbag: bool = False # 是否已经翻倍出本，剩下的让利润奔跑
    # Older persisted positions have spot-quote high-water marks. New entries
    # track the value of selling the whole remaining position.
    highest_price_basis: str = "spot"
    highest_price_initialized: bool = True

class PortfolioManager:
    def __init__(self, state_file="portfolio_state.json"):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.load_state()

    def add_position(self, token, symbol, price, amount, cost_sol):
        self.positions[token] = Position(
            token_address=token,
            symbol=symbol,
            entry_price=price,
            entry_time=time.time(),
            amount_held=amount,
            initial_cost_sol=cost_sol,
            highest_price=price,
            highest_price_basis="full_exit",
            highest_price_initialized=False,
        )
        self.save_state()
        logger.info(f"[+] Position Added: {symbol} @ {price}")

    def update_price(self, token, current_price):
        if token in self.positions:
            pos = self.positions[token]
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            self.save_state()

    def update_holding(self, token, new_amount):
        if token in self.positions:
            self.positions[token].amount_held = new_amount
            if new_amount <= 0:
                del self.positions[token]
            self.save_state()

    def close_position(self, token):
        if token in self.positions:
            logger.info(f"[+] Position Closed: {self.positions[token].symbol}")
            del self.positions[token]
            self.save_state()

    def get_open_count(self):
        return len(self.positions)

    def save_state(self):
        data = {k: asdict(v) for k, v in self.positions.items()}
        tmp_path = f"{self.state_file}.{os.getpid()}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.state_file)

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    self.positions[k] = Position(**v)
        except FileNotFoundError:
            self.positions = {}
