import asyncio
import inspect
import os
from datetime import datetime, timezone

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.models import TxOpts
from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.commitment_config import CommitmentLevel
from solders.account_decoder import UiAccountEncoding
from solders.rpc.config import RpcAccountInfoConfig, RpcTokenAccountsFilterMint
from solders.rpc.requests import GetTokenAccountsByOwner
from solders.rpc.responses import GetTokenAccountsByOwnerJsonParsedResp
from loguru import logger
from .config import ExecutionConfig


class TransactionStatusUnknown(RuntimeError):
    """The transaction was submitted, but its final status could not be read."""

    def __init__(self, signature, reason=None, *, metadata=None):
        self.signature = str(signature)
        self.reason = reason
        self.metadata = dict(metadata or {})
        message = f"Transaction status unknown: {self.signature}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class TransactionFailed(RuntimeError):
    """The RPC explicitly reported a failed transaction."""

    def __init__(self, signature, error):
        self.signature = str(signature)
        self.error = error
        super().__init__(f"Transaction failed: {self.signature}: {error}")


class QuickNodeClient:
    def __init__(self):
        timeout = float(os.getenv("SOLANA_RPC_TIMEOUT_SECONDS", "20"))
        self.client = AsyncClient(
            ExecutionConfig.RPC_URL,
            commitment=Confirmed,
            timeout=timeout,
        )
        self._status_attempts = int(os.getenv("SOLANA_STATUS_ATTEMPTS", "8"))
        self._confirmation_timeout_seconds = float(
            os.getenv("SOLANA_CONFIRMATION_WAIT_SECONDS", "3")
        )

    async def get_balance(self):
        try:
            resp = await self.client.get_balance(ExecutionConfig.get_payer_keypair().pubkey())
            return resp.value / 1e9
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            # An unavailable balance is not an empty wallet.  Returning zero
            # here used to make an RPC outage look like insufficient funds.
            return None

    async def get_token_balance(
        self, mint_address: str, *, commitment="confirmed", min_context_slot=None
    ):
        try:
            owner = Pubkey.from_string(ExecutionConfig.get_wallet_address())
            mint = Pubkey.from_string(mint_address)
            opts = TokenAccountOpts(
                program_id=Pubkey.from_string(
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                ),
                mint=mint,
            )
            if min_context_slot is None:
                response = await self.client.get_token_accounts_by_owner_json_parsed(
                    owner, opts, commitment=commitment
                )
            else:
                # The installed solana-py wrapper omits minContextSlot.
                levels = {
                    "processed": CommitmentLevel.Processed,
                    "confirmed": CommitmentLevel.Confirmed,
                    "finalized": CommitmentLevel.Finalized,
                }
                request = GetTokenAccountsByOwner(
                    owner,
                    RpcTokenAccountsFilterMint(mint),
                    RpcAccountInfoConfig(
                        encoding=UiAccountEncoding.JsonParsed,
                        commitment=levels[str(commitment)],
                        min_context_slot=int(min_context_slot),
                    ),
                )
                response = await self.client._provider.make_request(
                    request, GetTokenAccountsByOwnerJsonParsedResp
                )
            return sum(
                int(account.account.data.parsed["info"]["tokenAmount"]["amount"])
                for account in response.value
            )
        except Exception as e:
            logger.error(f"Failed to get token balance for {mint_address}: {e}")
            return None

    @staticmethod
    def _status_value(value):
        if value is None:
            return None
        status = getattr(value, "confirmation_status", None)
        status = getattr(status, "value", status)
        if status is not None:
            status = str(status).lower().split(".")[-1]
        # A processed result may disappear on a fork. Do not book either a
        # success or a definitive failure before confirmed commitment.
        if status not in {"confirmed", "finalized"}:
            return "pending", status or "unconfirmed"
        if getattr(value, "err", None) is not None:
            return "failed", value.err
        return "confirmed", status

    async def get_signature_status(self, signature, *, search_transaction_history=True):
        """Read one signature without turning an RPC outage into failure."""
        try:
            if isinstance(signature, str):
                signature = Signature.from_string(signature)
            response = await self.client.get_signature_statuses(
                [signature], search_transaction_history=search_transaction_history
            )
            if response.value is None or len(response.value) != 1:
                raise ValueError("Malformed signature-status response")
            value = response.value[0]
            if value is None:
                return "pending", None
            return self._status_value(value)
        except Exception as e:
            logger.warning(f"Signature status unavailable for {signature}: {e}")
            return "unknown", e

    async def get_recovery_fence(self):
        """Read a fresh blockhash horizon for a legacy-record safety fence."""
        response = await self.client.get_latest_blockhash(commitment="confirmed")
        return {
            "recent_blockhash": str(response.value.blockhash),
            "last_valid_block_height": int(response.value.last_valid_block_height),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "observed_slot": int(response.context.slot),
        }

    async def get_expiry_evidence(
        self, signature, *, recent_blockhash=None, last_valid_block_height=None
    ):
        """Prove an absent transaction can no longer land, without submitting.

        Age alone is never evidence of expiration. A finalized block-height
        horizon is required, followed by
        another historical signature query AND an absent transaction lookup.
        RPC errors, processed statuses and incomplete evidence stay pending.
        """
        evidence = {"signature": str(signature)}
        state, detail = await self.get_signature_status(signature)
        if state in {"confirmed", "failed", "unknown"}:
            evidence["status_detail"] = str(detail) if detail is not None else None
            return state, evidence
        if last_valid_block_height is None:
            # A fresh blockhash may not yet exist in the finalized bank.
            # isBlockhashValid(finalized)=false alone cannot prove expiry.
            evidence["reason"] = "missing_expiry_metadata"
            return "unknown", evidence
        try:
            horizon = int(last_valid_block_height)
            if isinstance(last_valid_block_height, bool) or horizon <= 0:
                raise ValueError("Invalid transaction-expiry block height")
            height = int((await self.client.get_block_height(commitment="finalized")).value)
            evidence.update(
                last_valid_block_height=horizon,
                finalized_block_height=height,
                expiry_basis="finalized_block_height",
            )
            if height <= horizon:
                return "pending", evidence
            # This anchor lets the caller request a wallet read at least
            # as recent as the finalized expiry evidence.
            evidence["expiry_slot"] = int(
                (await self.client.get_slot(commitment="finalized")).value
            )

            # Re-read AFTER expiry is established, closing the race with the
            # first query. A processed record also blocks an expired verdict.
            state, detail = await self.get_signature_status(signature)
            if state != "pending" or detail is not None:
                evidence["status_detail"] = str(detail) if detail is not None else None
                return state, evidence
            sig = Signature.from_string(signature) if isinstance(signature, str) else signature
            tx = await self.client.get_transaction(
                sig, commitment="confirmed", max_supported_transaction_version=0
            )
            if tx.value is not None:
                meta = getattr(getattr(tx.value, "transaction", None), "meta", None)
                if meta is None:
                    evidence["reason"] = "transaction_found_without_result"
                    return "unknown", evidence
                error = getattr(meta, "err", None)
                evidence["transaction_found"] = True
                evidence["status_detail"] = str(error) if error is not None else None
                return ("failed" if error is not None else "confirmed"), evidence
            evidence.update(history_status_absent=True, transaction_absent=True)
            return "expired", evidence
        except Exception as error:
            logger.warning(f"Expiry evidence unavailable for {signature}: {error}")
            evidence["reason"] = str(error)
            return "unknown", evidence

    async def send_and_confirm(
        self, txn, max_retries=3, *, on_submitting=None, last_valid_block_height=None
    ):
        # A signed Solana transaction already contains its stable identity.
        # Save it BEFORE network I/O: a lost send response does not mean that
        # the RPC failed to broadcast the transaction.
        signatures = txn.signatures
        if not signatures or signatures[0] == Signature.default():
            raise ValueError("Refusing to submit a transaction without a local signature")
        sig_str = str(signatures[0])
        metadata = {
            "signature": sig_str,
            "recent_blockhash": str(txn.message.recent_blockhash),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        if last_valid_block_height is not None:
            metadata["last_valid_block_height"] = int(last_valid_block_height)
        if on_submitting is not None:
            result = on_submitting(dict(metadata))
            if inspect.isawaitable(result):
                await result
        try:
            response = await self.client.send_transaction(
                txn,
                opts=TxOpts(
                    skip_confirmation=True,
                    preflight_commitment="processed",
                    max_retries=max_retries,
                ),
            )
        except Exception as e:
            logger.warning(f"Transaction submission outcome unknown for {sig_str}: {e}")
            raise TransactionStatusUnknown(sig_str, e, metadata=metadata) from e

        if str(response.value) != sig_str:
            raise TransactionStatusUnknown(
                sig_str, "RPC response signature differs from signed transaction", metadata=metadata
            )
        logger.info(f"Transaction Sent: {sig_str}")
        delay = 0.75
        last_error = None
        deadline = asyncio.get_running_loop().time() + max(
            0.01, getattr(self, "_confirmation_timeout_seconds", 3.0)
        )
        for attempt in range(self._status_attempts):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                state, detail = await asyncio.wait_for(
                    self.get_signature_status(signatures[0]), timeout=remaining
                )
            except asyncio.TimeoutError as error:
                last_error = error
                break
            if state == "confirmed":
                logger.success(f"Transaction Confirmed: https://solscan.io/tx/{sig_str}")
                return sig_str
            if state == "failed":
                raise TransactionFailed(sig_str, detail)
            if state == "unknown":
                last_error = detail
            if attempt + 1 < self._status_attempts:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, 5.0)

        # A signature exists, so retrying the swap would be unsafe.  The
        # caller must persist this as pending and reconcile it later.
        raise TransactionStatusUnknown(sig_str, last_error, metadata=metadata)

    async def close(self):
        await self.client.close()
