"""Collect prospective Jupiter round-trip quotes without a wallet or trades.

Pass token mints explicitly or through a text file. Each mint gets one ExactIn
1 SOL -> token quote followed by a quote to sell the *entire quoted token
amount* back to SOL. Results, including failures, are appended to JSONL.

This module intentionally does not import the trading client or wallet config.
It only makes GET requests to Jupiter's official quote endpoint. Set
JUPITER_API_KEY in the process environment; this collector never loads .env.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from solders.pubkey import Pubkey

from execution.rate_limit import wait_for_shared_slot


SOL_MINT = "So11111111111111111111111111111111111111112"
ONE_SOL_LAMPORTS = 1_000_000_000
DEFAULT_JUPITER_BASE_URL = "https://api.jup.ag/swap/v1"
MAX_MINTS_PER_RUN = 5
HTTP_TIMEOUT_SECONDS = 6.0
TOKEN_TIMEOUT_SECONDS = 15.0
MIN_REQUEST_INTERVAL_SECONDS = 0.25
SLIPPAGE_BPS = 200  # Match the live quote request; not an execution-cost estimate.


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_mint(value: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("mint must be a canonical Solana address")
    try:
        canonical = str(Pubkey.from_string(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("mint must be a canonical Solana address") from exc
    if canonical != value or canonical == SOL_MINT:
        raise ValueError("mint must be a token address other than SOL")
    return canonical


def _positive_u64(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("amount must be a positive integer")
    if isinstance(value, str) and (not value or not value.isascii() or not value.isdecimal()):
        raise ValueError("amount must contain decimal digits only")
    amount = int(value)
    if not 0 < amount < 2**64:
        raise ValueError("amount must fit in a positive u64")
    return amount


class QuoteError(Exception):
    def __init__(self, category: str, http_status: int | None = None):
        super().__init__(category)
        self.category = category
        self.http_status = http_status


def _checked_quote(quote: object, input_mint: str, output_mint: str, amount: int) -> int:
    if quote is None or quote == {}:
        raise QuoteError("no_quote")
    if not isinstance(quote, dict):
        raise QuoteError("invalid_quote")
    try:
        valid = (
            quote.get("inputMint") == input_mint
            and quote.get("outputMint") == output_mint
            and quote.get("swapMode") == "ExactIn"
            and _positive_u64(quote.get("inAmount")) == amount
        )
        out_amount = _positive_u64(quote.get("outAmount"))
    except (TypeError, ValueError) as exc:
        raise QuoteError("invalid_quote") from exc
    if not valid:
        raise QuoteError("invalid_quote")
    return out_amount


def _quote_metadata(quote: dict) -> tuple[int | None, float | None, list[str]]:
    slot = quote.get("contextSlot")
    if type(slot) is not int or slot < 0:
        slot = None
    elapsed = quote.get("timeTaken")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        elapsed = None
    elif not math.isfinite(elapsed) or elapsed < 0:
        elapsed = None
    routes = []
    plan = quote.get("routePlan")
    if isinstance(plan, list):
        for leg in plan:
            info = leg.get("swapInfo") if isinstance(leg, dict) else None
            label = info.get("label") if isinstance(info, dict) else None
            if isinstance(label, str) and label.strip():
                routes.append(label.strip())
    return slot, elapsed, routes


class JupiterQuoteOnlyClient:
    """One GET quote at a time, sharing the live process's Jupiter rate gate."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_JUPITER_BASE_URL,
        session: aiohttp.ClientSession | None = None,
        rate_gate=wait_for_shared_slot,
    ):
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("JUPITER_API_KEY is required in the process environment")
        try:
            url = urlsplit(base_url)
            official_endpoint = (
                url.scheme == "https" and url.hostname == "api.jup.ag"
                and url.port in (None, 443) and url.path.rstrip("/") == "/swap/v1"
                and not (url.query or url.fragment or url.username or url.password)
            )
        except (TypeError, ValueError):
            official_endpoint = False
        if not official_endpoint:
            raise ValueError("Jupiter base URL must be https://api.jup.ag/swap/v1")
        self.url = DEFAULT_JUPITER_BASE_URL + "/quote"
        self.headers = {"accept": "application/json", "x-api-key": api_key}
        self.session = session
        self._owns_session = session is None
        self.rate_gate = rate_gate

    async def __aenter__(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
            )
        return self

    async def __aexit__(self, *_):
        if self._owns_session and self.session is not None:
            await self.session.close()
            self.session = None

    async def get_quote(self, input_mint: str, output_mint: str, amount: int):
        if self.session is None:
            raise RuntimeError("quote client is not open")
        await self.rate_gate(MIN_REQUEST_INTERVAL_SECONDS)
        requested_at = _utc_now()
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(SLIPPAGE_BPS),
            "swapMode": "ExactIn",
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false",
        }
        try:
            async with self.session.get(
                self.url, params=params, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    if 300 <= response.status < 400:
                        category = "redirect_blocked"
                    elif response.status == 429:
                        category = "rate_limited"
                    elif 400 <= response.status < 500:
                        category = "http_4xx"
                    elif 500 <= response.status < 600:
                        category = "http_5xx"
                    else:
                        category = "http_unexpected"
                    raise QuoteError(category, response.status)
                try:
                    quote = await response.json()
                except (aiohttp.ContentTypeError, ValueError) as exc:
                    raise QuoteError("invalid_json") from exc
        except asyncio.TimeoutError as exc:
            raise QuoteError("request_timeout") from exc
        except (aiohttp.ClientSSLError, aiohttp.ServerFingerprintMismatch) as exc:
            raise QuoteError("tls_error") from exc
        except aiohttp.ClientError as exc:
            raise QuoteError("transport_error") from exc
        return quote, requested_at, _utc_now()


async def collect_mint(mint: str, client, *, token_timeout: float = TOKEN_TIMEOUT_SECONDS) -> dict:
    """Return one auditable row; failures never yield a numeric cost."""
    mint = validate_mint(mint)
    if not math.isfinite(token_timeout) or token_timeout <= 0 or token_timeout > 30:
        raise ValueError("token timeout must be between 0 and 30 seconds")
    row = {
        "observed_at": _utc_now(),
        "token_mint": mint,
        "input_lamports": ONE_SOL_LAMPORTS,
        "slippage_bps_requested": SLIPPAGE_BPS,
        "status": "unavailable",
        "error_category": None,
        "http_status": None,
        "buy_requested_at": None,
        "buy_received_at": None,
        "sell_requested_at": None,
        "sell_received_at": None,
        "buy_context_slot": None,
        "sell_context_slot": None,
        "buy_time_taken_seconds": None,
        "sell_time_taken_seconds": None,
        "buy_route_labels": [],
        "sell_route_labels": [],
        "buy_out_amount_raw": None,
        "sell_out_lamports": None,
        "round_trip_cost_bps": None,
    }
    stage = "buy"

    async def two_quotes():
        nonlocal stage
        buy, row["buy_requested_at"], row["buy_received_at"] = await client.get_quote(
            SOL_MINT, mint, ONE_SOL_LAMPORTS
        )
        amount = _checked_quote(buy, SOL_MINT, mint, ONE_SOL_LAMPORTS)
        row["buy_out_amount_raw"] = amount
        slot, elapsed, labels = _quote_metadata(buy)
        row["buy_context_slot"] = slot
        row["buy_time_taken_seconds"] = elapsed
        row["buy_route_labels"] = labels

        stage = "sell"
        sell, row["sell_requested_at"], row["sell_received_at"] = await client.get_quote(
            mint, SOL_MINT, amount
        )
        out_lamports = _checked_quote(sell, mint, SOL_MINT, amount)
        row["sell_out_lamports"] = out_lamports
        slot, elapsed, labels = _quote_metadata(sell)
        row["sell_context_slot"] = slot
        row["sell_time_taken_seconds"] = elapsed
        row["sell_route_labels"] = labels
        row["round_trip_cost_bps"] = 10_000 * (
            ONE_SOL_LAMPORTS - out_lamports
        ) / ONE_SOL_LAMPORTS
        row["status"] = "available"

    try:
        await asyncio.wait_for(two_quotes(), timeout=token_timeout)
    except QuoteError as exc:
        row["error_category"] = f"{stage}_{exc.category}"
        row["http_status"] = exc.http_status
    except asyncio.TimeoutError:
        row["error_category"] = f"{stage}_token_timeout"
    except Exception:
        # Do not print exception text: transport exceptions may include URLs.
        row["error_category"] = f"{stage}_internal_error"
    return row


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("output must have a .jsonl suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("short JSONL append")
        os.fsync(fd)
    finally:
        os.close(fd)


async def collect_many(mints: list[str], output: str | Path, client) -> list[dict]:
    """Collect sequentially: at most one token and one API request in flight."""
    if not mints or len(mints) > MAX_MINTS_PER_RUN:
        raise ValueError(f"pass 1 to {MAX_MINTS_PER_RUN} unique token mints")
    canonical = list(dict.fromkeys(validate_mint(mint) for mint in mints))
    if len(canonical) != len(mints):
        raise ValueError("duplicate token mint")
    rows = []
    for mint in canonical:
        row = await collect_mint(mint, client)
        append_jsonl(output, row)
        rows.append(row)
    return rows


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mint", action="append", default=[], help="Token mint; repeat as needed")
    parser.add_argument("--mints-file", type=Path, help="One token mint per line; # comments allowed")
    parser.add_argument("--output", type=Path, default=Path("logs/quote_snapshots.jsonl"))
    args = parser.parse_args(argv)
    if args.mints_file:
        args.mint.extend(
            line.strip() for line in args.mints_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not args.mint:
        parser.error("provide --mint or --mints-file")
    return args


async def _main(argv=None):
    args = _parse_args(argv)
    key = os.environ.get("JUPITER_API_KEY", "")
    base_url = os.environ.get("JUPITER_BASE_URL", DEFAULT_JUPITER_BASE_URL)
    async with JupiterQuoteOnlyClient(key, base_url=base_url) as client:
        rows = await collect_many(args.mint, args.output, client)
    available = sum(row["status"] == "available" for row in rows)
    print(f"Appended {len(rows)} quote snapshots ({available} available) to {args.output}")


if __name__ == "__main__":
    asyncio.run(_main())
