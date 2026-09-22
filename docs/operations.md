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
`model_core/vocab.py`. Training writes this artifact; it is not deployed from Git.
Review `strategy_manager/config.py` before enabling live entries. Current source
defaults are five positions and 1 SOL per entry.

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
