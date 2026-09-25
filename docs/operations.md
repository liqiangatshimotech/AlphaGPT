# Live runner operations

## Install and validate

Use Python 3.10 or newer. The execution layer is validated with solana-py
0.38 and solders 0.27; keep these dependencies within the ranges in
`requirements.txt`.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The PostgreSQL integration test creates temporary tables on a dedicated
connection and does not update application tables. It skips if the configured
database is unavailable. Execution tests use fake clients and do not send trades.

Configure credentials locally in `.env`; do not commit the file. The main
services require `BIRDEYE_API_KEY`, `JUPITER_API_KEY`, `QUICKNODE_RPC_URL`,
`SOLANA_PRIVATE_KEY`, and the `DB_*` settings in `data_pipeline/config.py`.
`QUICKNODE_RPC_URL` is a historical variable name and may point to any compatible
Solana RPC provider, including Helius. Keys must correspond to the configured
provider endpoints.

The runner requires a locally trained `best_meme_strategy.json` compatible with
`model_core/vocab.py`. This live artifact is not deployed from Git.
Review `strategy_manager/config.py` before enabling live entries. Current source
defaults are five positions and 1 SOL per entry.

Training now writes `candidate_meme_strategy.json`; promote a candidate to the
live filename only after reviewing its validation and live score distribution.
The entry scanner refuses a non-finite or nearly constant score batch. A valid
strategy file alone does not imply that live entries are enabled.

For a model-recovery investigation, `TOKEN_QUOTE_SAMPLE_SIZE` controls an
optional read-only sample of at most five Birdeye-selected mints per 15-minute
pipeline sync (default `0`, disabled). It appends 1 SOL buy/full-sell quote
results and failures to `TOKEN_QUOTE_SNAPSHOT_PATH` (default
`logs/quote_snapshots.jsonl`); the same sync records the observed candidate
universe in `TOKEN_UNIVERSE_SNAPSHOT_PATH` (default
`logs/token_universe_snapshots.jsonl`). Birdeye returns at most 50 trending
mints here, not the full market. Quote sampling shares the Jupiter API budget
with live exits, so monitor rate limits and exit latency after enabling it.
Quotes are collected after the OHLCV refresh; use each quote row's own request
time and check its delay from the discovery snapshot before analysis.
These logs contain no wallet key and do
not build, sign, or send transactions. They supply prospective cost and
universe evidence; [the recovery plan](model-recovery-2026-09-25.md) explains
why they cannot justify live promotion on their own.

## Entry and exit safeguards

`MAX_SIGNAL_CANDLE_AGE_SECONDS` defaults to 1800. A live candidate must have an
actual OHLCV candle within that window; forward-filled values from an older
token cannot make it eligible. If none are fresh, entries are skipped while
position monitoring continues. The entry liquidity minimum is $500,000,
matching the data discovery and backtest thresholds.

`MAX_ENTRY_ROUND_TRIP_COST_BPS` defaults to 300. Before buying, the runner
quotes a sale of the full token amount expected from the exact 1 SOL buy quote.
It rejects missing routes or expected round-trip losses above 3%. This is a
quote-based screen, not a cap on execution slippage, fees, or later price moves.
Confirmed stop-loss sales also block re-entry into that token for 24 hours;
the cooldown is restored from `logs/order_recovery.jsonl` after a restart.

For held tokens, stop-loss and first take-profit checks use a quote for the
full on-chain balance. Existing positions keep their historical one-token
high-water basis; new positions initialize and update the high-water mark with
full-position exit quotes. When the chain balance is temporarily unavailable,
the runner can still evaluate a stop from the saved size, but the trader must
verify that exact balance before sending a sell. A zero
balance at both confirmed and finalized commitment is quarantined in the
portfolio and reported again every hour; it is not deleted without independent
exit evidence. A quarantined position continues to occupy one of the five
slots. To clear one, verify the zero balance with a second RPC or explorer and
review wallet transaction history, confirm there is no pending order, stop the
runner, back up `portfolio_state.json`, remove only that verified position from
the saved state, then restart and recheck balances. Do not clear a position on
one provider's zero response alone.

The optional monitor bot scans recent runner logs for zero-balance quarantine
events tied to currently saved positions and includes a manual-audit warning
in its outgoing summary. This addition takes effect after the monitor bot's
next restart; it does not itself resolve or remove a position.

Mint precision is read from Solana `getTokenSupply` and cached in each position.
An unavailable mint response is never assumed to have six decimals; if no
verified precision is cached, that position's quote-based exit check is
deferred rather than risking a false stop-loss sale.

## State and pending transactions

Keep these files with the existing wallet deployment, outside version control:

- `portfolio_state.json`: position amounts, entry prices, high-water marks and
  first take-profit flags.
- `pending_orders.json`: signed transaction identities, exact expiration block
  heights and balances recorded **before** submission.
- `STOP_SIGNAL`: creating this file with `STOP` pauses new entries. Existing
  position monitoring and exits continue.
- `pending_recovery.json`: optional, audited recovery evidence for legacy orders
  that did not record their expiry metadata. Never reuse another wallet's file.
- `logs/order_recovery.jsonl`: append-only records of reconciled transactions.

Do not delete a pending order just because a request timed out. Submission may
have succeeded even when the response was lost. The runner checks transaction
history, then accepts expiry only after finalized block height exceeds the
recorded bound, the signature/transaction are still absent, and finalized token
balance matches the pre-submission balance. Missing or contradictory evidence
keeps the order pending for investigation.

An explicitly rejected preflight simulation from a single-attempt send can be
released sooner, but only after the historical signature query is absent and
the finalized token balance equals the pre-submission balance. A transport
timeout remains pending. For a DEX program failure, the runner maps the program
ID through Jupiter and retries once with that specific DEX excluded. The
failure program ID is kept in memory before reconciliation awaits, so a
timeout after a preflight rejection can inform the next sell attempt in this
process. Pending-order safety checks still take precedence over route changes.
The RPC node's default rebroadcasting remains enabled until finalization or
blockhash expiry. While any order is pending, the runner checks recovery every
10 seconds; it does not issue a new quote merely because confirmation is slow.

Confirmed partial sales use actual on-chain balances and persist the first
take-profit flag, so a restart does not sell another half of the position by
reapplying the same transaction. After an expired order is cleared, exits are
evaluated using current prices and the existing strategy.

For audited legacy recovery, `pending_recovery.json` entries must match the exact
token and signature and contain `kind: audited_legacy_jupiter_v1`, a conservative
`last_valid_block_height`, `pre_raw_balance`, `amount_raw`, `decimals`, and
`recovery_evidence`. Such a bound is not the original transaction's expiry;
establish its validity independently before supplying it. There is no automatic
wall-clock timeout for discarding unknown legacy transactions.

## API pacing

Quote and swap-building requests, including the position-report monitor, share
`JUPITER_RATE_LIMIT_FILE` (default `/tmp/alphagpt-jupiter-rate.lock`) on the same
macOS/Linux host. All local consumers must use the same file. This does not
coordinate external tools or other machines using the same API key.

The runner defaults to a 0.20-second request interval; the monitor uses 0.25
seconds. Keep these at or above their defaults for headroom under a 10 RPS plan.
429 responses have bounded retries with exponential backoff. No HTTP retry in
the quote/swap-building layer broadcasts a signed transaction.
Jupiter HTTP requests default to a six-second timeout. Position balance reads,
quotes and exit attempts have separate short budgets; a failure for one token
does not stop checks for other positions. The slower Birdeye/DB pipeline sync
runs as a background task so it cannot hold up exit monitoring.

`SOLANA_RPC_TIMEOUT_SECONDS` defaults to 20 seconds;
`SOLANA_CONFIRMATION_WAIT_SECONDS` defaults to 3 seconds before handing an
unresolved transaction back to pending reconciliation. The overall recovery
pass has an eight-second budget to avoid indefinitely delaying other positions.

## Deployment

Preserve the wallet state and pending journal during every update. Validate the
new source before restarting a live process. Run only one strategy runner per
wallet: two runners can submit duplicate exits even with entries paused.
Check that the supervisor reports one healthy process and that logs show
position monitoring and pending reconciliation after restart.

The optional `monitor_bot.py` produces reports and notifications only. Configure
its `DEEPSEEK_*` and `DINGTALK_*` values locally; it does not place trades. Use
`python monitor_bot.py --help` for the available reporting options. A live
notification run sends messages to the configured webhook.
