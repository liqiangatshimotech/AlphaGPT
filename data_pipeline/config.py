import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0  
    MIN_FDV = 10000000.0            
    MAX_FDV = float('inf') 
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_BASE_URL = os.getenv("BIRDEYE_BASE_URL", "https://public-api.birdeye.so").rstrip("/")
    BIRDEYE_OFFICIAL_HOST = "public-api.birdeye.so"
    BIRDEYE_ALLOW_CUSTOM_BASE_URL = os.getenv(
        "BIRDEYE_ALLOW_CUSTOM_BASE_URL", "false"
    ).strip().lower() in {"1", "true", "yes"}
    BASE_URL = BIRDEYE_BASE_URL
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = 20
    HISTORY_DAYS = 7

    @classmethod
    def birdeye_headers(cls):
        """Build Birdeye headers without sending the API key to an unapproved host."""
        parsed = urlparse(cls.BIRDEYE_BASE_URL)
        if (parsed.scheme != "https" or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("BIRDEYE_BASE_URL must be HTTPS without credentials, query or fragment")

        is_official = (
            parsed.scheme == "https"
            and parsed.hostname.lower() == cls.BIRDEYE_OFFICIAL_HOST
            and parsed.port in {None, 443}
        )
        if not is_official and not cls.BIRDEYE_ALLOW_CUSTOM_BASE_URL:
            raise ValueError(
                "Custom BIRDEYE_BASE_URL is disabled. Set "
                "BIRDEYE_ALLOW_CUSTOM_BASE_URL=true only for a trusted endpoint."
            )

        headers = {"accept": "application/json"}
        if cls.BIRDEYE_API_KEY:
            headers["X-API-KEY"] = cls.BIRDEYE_API_KEY
        return headers
