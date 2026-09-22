"""Small cross-process request gate for shared API keys."""

import asyncio
import os
import time

try:  # pragma: no cover - available on the macOS/Linux deployment host
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


DEFAULT_RATE_LIMIT_FILE = "/tmp/alphagpt-jupiter-rate.lock"


def _reserve_slot(path, minimum_interval):
    """Reserve one slot while holding an advisory POSIX lock."""
    if fcntl is None:
        time.sleep(minimum_interval)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                last = float(raw)
            except (TypeError, ValueError):
                last = 0.0
            now = time.time()
            wait = minimum_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            handle.seek(0)
            handle.truncate()
            handle.write(f"{now:.6f}\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def wait_for_shared_slot(minimum_interval, path=None):
    """Throttle requests across all local Jupiter-consuming processes."""
    rate_path = path or os.getenv("JUPITER_RATE_LIMIT_FILE", DEFAULT_RATE_LIMIT_FILE)
    await asyncio.to_thread(_reserve_slot, rate_path, float(minimum_interval))
