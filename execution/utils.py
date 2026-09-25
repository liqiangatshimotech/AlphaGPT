from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from .config import ExecutionConfig

async def get_mint_decimals(mint_str: str, client: AsyncClient) -> int:
    """Return verified mint precision; an RPC outage must never guess six."""
    if mint_str == ExecutionConfig.SOL_MINT:
        return 9
    pubkey = Pubkey.from_string(mint_str)
    response = await client.get_token_supply(pubkey)
    if response.value is None:
        raise ValueError(f"Mint precision unavailable for {mint_str}")
    decimals = int(response.value.decimals)
    if not 0 <= decimals <= 18:
        raise ValueError(f"Invalid mint precision for {mint_str}: {decimals}")
    return decimals
