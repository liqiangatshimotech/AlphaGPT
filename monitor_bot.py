"""Read-only AlphaGPT health snapshots and DingTalk notifications.

This source replaces the deployment's Python-version-specific bytecode loader.
It never submits trades or changes portfolio/strategy state.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import asyncpg
from dotenv import load_dotenv
from solders.keypair import Keypair

from monitor_exit_distance import position_report


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
FRESHNESS_THRESHOLD_SECONDS = int(os.getenv("MONITOR_FRESHNESS_THRESHOLD_SECONDS", "1200"))
QUARANTINE_ALERT_MAX_AGE_SECONDS = 2 * 3600
QUARANTINE_WARNING_PREFIX = "Zero-balance quarantine: "


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def local_timezone() -> ZoneInfo:
    name = env("MONITOR_TIMEZONE", "Asia/Shanghai")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        print(f"Invalid MONITOR_TIMEZONE={name!r}; falling back to Asia/Shanghai")
        return ZoneInfo("Asia/Shanghai")


def display_time(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(local_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")


def redact(value: str) -> str:
    return value[:6] + "..." + value[-6:] if len(value) > 16 else "<configured>"


@dataclass
class Snapshot:
    collected_at_utc: str
    runner_running: bool
    runner_pid: int | None
    entries_paused: bool
    wallet_address: str
    wallet_balance_sol: float | None
    db_rows: int | None
    db_tokens: int | None
    db_latest_candle_utc: str | None
    local_positions: list[dict[str, Any]]
    recent_transactions: list[dict[str, Any]]
    recent_log_events: list[str]
    warnings: list[str]


def runner_process() -> tuple[bool, int | None]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None
    for line in result.stdout.splitlines():
        if "strategy_manager.runner" in line and "rg " not in line:
            match = re.match(r"\s*(\d+)\s+(.*)", line)
            return True, int(match.group(1)) if match else None
    return False, None


def local_positions() -> list[dict[str, Any]]:
    try:
        data = json.loads((ROOT / "portfolio_state.json").read_text())
        if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
            raise ValueError("Invalid portfolio state")
        return [{
            "symbol": v.get("symbol"), "token": v.get("token_address"),
            "amount": v.get("amount_held"), "cost_sol": v.get("initial_cost_sol"),
            "entry_price_sol": v.get("entry_price"), "highest_price_sol": v.get("highest_price"),
            "is_moonbag": v.get("is_moonbag", False),
        } for v in data.values()]
    except FileNotFoundError:
        return []
    except (OSError, TypeError, ValueError):
        return [{"error": "portfolio_state.json could not be parsed"}]


def _sanitize_log_line(line: str) -> str:
    # Exception URLs can contain credentials: remove configured secrets before
    # including the active strategy log in external summaries/notifications.
    for name in (
        "SOLANA_PRIVATE_KEY", "QUICKNODE_RPC_URL", "JUPITER_API_KEY", "BIRDEYE_API_KEY",
        "DEEPSEEK_API_KEY", "DINGTALK_WEBHOOK_URL", "DINGTALK_SECRET", "DB_PASSWORD",
    ):
        value = env(name)
        if value:
            line = line.replace(value, "<redacted>")
    return line[-500:]


def _active_log_lines() -> list[str]:
    # loguru writes to launchd's stderr file; preserve the old paths as fallbacks.
    candidates = [ROOT / "logs/runner.error.log", ROOT / "logs/runner.log", ROOT / "strategy.log"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    try:
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def recent_log_events(limit: int = 20, *, lines: list[str] | None = None) -> list[str]:
    if limit <= 0:
        return []
    if lines is None:
        lines = _active_log_lines()
    interesting = re.compile(
        r"(BUY Successful|SELL Successful|Transaction (Sent|Confirmed|Failed)|Jupiter .*Error|"
        r"Birdeye .*Error|Global Loop Error|STOP|AI EXIT|STOP LOSS|TRAILING STOP|MOONBAG TP)", re.I,
    )
    return [_sanitize_log_line(line) for line in [line for line in lines if interesting.search(line)][-limit:]]


def quarantine_warnings(
    positions: list[dict[str, Any]], *, lines: list[str] | None = None,
    now: float | None = None,
) -> list[str]:
    """Report recent zero-balance quarantines still present in saved positions."""
    if lines is None:
        lines = _active_log_lines()
    if now is None:
        now = time.time()
    current = {
        position.get("token"): position.get("symbol")
        for position in positions if position.get("token")
    }
    if not current:
        return []
    by_symbol: dict[str, list[str]] = {}
    for token, symbol in current.items():
        by_symbol.setdefault(symbol, []).append(token)
    quarantined: dict[str, float] = {}
    zero_event = re.compile(
        r"Zero on-chain balance for .+? \(([1-9A-HJ-NP-Za-km-z]{32,44})\) "
        r"at confirmed and finalized commitment; retaining position for audit"
    )
    restored_event = re.compile(r"On-chain balance restored for (.+?); resuming monitoring\.")
    for line in lines:
        timestamp = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if timestamp is None:
            continue
        try:
            observed_at = datetime.fromisoformat(timestamp.group(1)).timestamp()
        except ValueError:
            continue
        if observed_at > now + 300 or now - observed_at > QUARANTINE_ALERT_MAX_AGE_SECONDS:
            continue
        if "Loaded Strategy:" in line:
            # A new runner does not inherit the previous process's quarantine.
            # Its startup balance check will report any still-zero position.
            quarantined.clear()
            continue
        zero = zero_event.search(line)
        if zero and zero.group(1) in current:
            quarantined[zero.group(1)] = observed_at
        restored = restored_event.search(line)
        if restored:
            tokens = by_symbol.get(restored.group(1), [])
            # A symbol can repeat; only clear an unambiguous restoration.
            if len(tokens) == 1:
                quarantined.pop(tokens[0], None)
    return [
        f"{QUARANTINE_WARNING_PREFIX}{current[token]} ({token}); "
        f"last observed {max(0, int((now - observed_at) // 60))} min ago; manual chain audit required"
        for token, observed_at in quarantined.items()
    ]


def signature_from_logs(events: list[str]) -> list[str]:
    return list(dict.fromkeys(re.findall(
        r"Transaction Sent: ([1-9A-HJ-NP-Za-km-z]{32,90})", "\n".join(events),
    )))


async def rpc_call(session: aiohttp.ClientSession, method: str, params: list[Any]) -> Any:
    async with session.post(
        env("QUICKNODE_RPC_URL"),
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        allow_redirects=False,
    ) as response:
        body = await response.json(content_type=None)
        if response.status != 200 or body.get("error"):
            raise RuntimeError(f"RPC {method} failed with HTTP {response.status}")
        return body.get("result")


async def collect_snapshot() -> Snapshot:
    warnings: list[str] = []
    running, pid = runner_process()
    stop_path = Path(env("STOP_SIGNAL_PATH", "STOP_SIGNAL"))
    if not stop_path.is_absolute():
        stop_path = ROOT / stop_path
    try:
        paused = stop_path.exists() and stop_path.read_text().strip().upper() in {"", "STOP", "STOPPED"}
    except (OSError, UnicodeError):
        paused = True
        warnings.append("STOP_SIGNAL could not be read")

    wallet_address, balance = "unknown", None
    try:
        wallet_address = str(Keypair.from_base58_string(env("SOLANA_PRIVATE_KEY")).pubkey())
    except Exception:
        warnings.append("wallet private key could not be parsed")

    db_rows = db_tokens = db_latest = None
    connection = None
    try:
        connection = await asyncpg.connect(
            user=env("DB_USER"), password=env("DB_PASSWORD"), host=env("DB_HOST"),
            port=int(env("DB_PORT", "5432")), database=env("DB_NAME"), timeout=5,
        )
        db_rows = await connection.fetchval("SELECT COUNT(*) FROM ohlcv")
        db_tokens = await connection.fetchval("SELECT COUNT(DISTINCT address) FROM ohlcv")
        latest = await connection.fetchval("SELECT MAX(time) FROM ohlcv")
        db_latest = latest.isoformat() if latest else None
    except Exception as exc:
        warnings.append(f"PostgreSQL unavailable: {type(exc).__name__}")
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception as exc:
                warnings.append(f"PostgreSQL close failed: {type(exc).__name__}")

    positions = local_positions()
    log_lines = _active_log_lines()
    events = recent_log_events(lines=log_lines)
    warnings.extend(quarantine_warnings(positions, lines=log_lines))
    transactions = [
        {"signature": sig, "solscan": f"https://solscan.io/tx/{sig}"}
        for sig in signature_from_logs(events)[-5:]
    ]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            if wallet_address != "unknown":
                result = await rpc_call(session, "getBalance", [wallet_address])
                balance = float(result["value"]) / 1e9
            for tx in transactions:
                try:
                    result = await rpc_call(session, "getTransaction", [tx["signature"], {
                        "encoding": "jsonParsed", "commitment": "confirmed", "maxSupportedTransactionVersion": 0,
                    }])
                    tx["chain_error"] = result.get("meta", {}).get("err") if result else "unavailable"
                    tx["fee_sol"] = result.get("meta", {}).get("fee", 0) / 1e9 if result else None
                except Exception:
                    tx["chain_error"] = "unavailable"
    except Exception as exc:
        warnings.append(f"Solana RPC unavailable: {type(exc).__name__}")

    if not running:
        warnings.append("strategy runner process is not running")
    if paused:
        warnings.append("new entries are paused by STOP_SIGNAL")
    if db_latest:
        latest = datetime.fromisoformat(db_latest)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age = time.time() - latest.timestamp()
        if age > FRESHNESS_THRESHOLD_SECONDS:
            warnings.append(f"market data is {age / 60:.1f} minutes old")
    return Snapshot(
        datetime.now(timezone.utc).isoformat(), running, pid, paused, redact(wallet_address),
        balance, db_rows, db_tokens, db_latest, positions, transactions, events, warnings,
    )


async def deepseek_summary(snapshot: Snapshot) -> str:
    prompt = {
        "instruction": "用中文输出一段简短的量化机器人运行摘要。只基于给定 JSON，不要建议自动下单，不要暴露密钥。包含：运行状态、钱包余额、持仓、最近交易、数据新鲜度、警告、下一步建议。最多 500 字。",
        "snapshot": asdict(snapshot),
    }
    payload = {
        "model": env("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": "你是一个谨慎的交易系统运维监控助手。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "thinking": {"type": "disabled"}, "temperature": 0.1, "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {env('DEEPSEEK_API_KEY')}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
        async with session.post(
            env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
            headers=headers, json=payload,
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200:
                raise RuntimeError(f"DeepSeek returned HTTP {response.status}")
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise RuntimeError("DeepSeek returned empty content")
            return content


def fallback_summary(snapshot: Snapshot) -> str:
    return (
        f"AlphaGPT 监控：Runner={'运行中' if snapshot.runner_running else '未运行'}"
        f"；余额={snapshot.wallet_balance_sol if snapshot.wallet_balance_sol is not None else '未知'} SOL"
        f"；持仓={len(snapshot.local_positions)}；行情K线={snapshot.db_rows or '未知'}"
        f"；暂停买入={'是' if snapshot.entries_paused else '否'}"
        f"；警告={'；'.join(snapshot.warnings) if snapshot.warnings else '无'}。"
    )


async def send_dingtalk(content: str) -> None:
    url, secret = env("DINGTALK_WEBHOOK_URL"), env("DINGTALK_SECRET")
    if secret:
        timestamp = str(int(time.time() * 1000))
        digest = hmac.new(secret.encode(), f"{timestamp}\n{secret}".encode(), hashlib.sha256).digest()
        sign = quote_plus(base64.b64encode(digest))
        url += ("&" if "?" in url else "?") + f"timestamp={timestamp}&sign={sign}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        async with session.post(url, json={"msgtype": "text", "text": {"content": content}}) as response:
            data = await response.json(content_type=None)
            if response.status != 200 or data.get("errcode") not in (0, None):
                raise RuntimeError(f"DingTalk returned HTTP {response.status}: {data.get('errmsg', 'unknown error')}")


async def run_once(dry_run: bool = False) -> None:
    snapshot = await collect_snapshot()
    report = await position_report(snapshot.local_positions)
    try:
        summary = await deepseek_summary(snapshot)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        warning = f"DeepSeek unavailable: {detail}"
        snapshot.warnings.append(warning)
        print(f"[{display_time(snapshot.collected_at_utc)}] {warning}", flush=True)
        summary = fallback_summary(snapshot)
    if report:
        summary += "\n\n" + report
    quarantine_alerts = [
        warning for warning in snapshot.warnings
        if warning.startswith(QUARANTINE_WARNING_PREFIX)
    ]
    if quarantine_alerts:
        # Keep the alert in the outgoing text even if the generated summary omits it.
        summary += "\n\n零余额隔离告警（需人工核查）：\n" + "\n".join(quarantine_alerts)
    content = f"{summary}\n\n采集时间：{display_time(snapshot.collected_at_utc)}"
    if dry_run:
        print(content)
        return
    await send_dingtalk(content)
    print(f"[{display_time(snapshot.collected_at_utc)}] DingTalk notification sent", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=int(env("MONITOR_INTERVAL_SECONDS", "300")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.once or args.dry_run:
        await run_once(args.dry_run)
        return
    while True:
        try:
            await run_once(False)
        except Exception as exc:
            print(
                f"[{display_time(datetime.now(timezone.utc).isoformat())}] "
                f"monitor iteration failed: {type(exc).__name__}: {exc}", flush=True,
            )
        await asyncio.sleep(max(args.interval, 30))


if __name__ == "__main__":
    asyncio.run(main())
