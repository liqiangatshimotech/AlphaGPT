import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
from solders.keypair import Keypair

from execution.jupiter import JupiterAggregator


class _QuoteResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def json(self):
        return {"outAmount": "123"}


class _MappingResponse(_QuoteResponse):
    def __init__(self, mapping, status=200):
        self.mapping = mapping
        self.status = status

    async def json(self):
        return self.mapping


class _Session:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, *, params=None, headers, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _CertificateError(aiohttp.ClientSSLError):
    def __init__(self):
        Exception.__init__(self, "certificate mismatch")

    def __str__(self):
        return "certificate mismatch"


class JupiterQuoteTests(unittest.IsolatedAsyncioTestCase):
    def _aggregator(self, session):
        jup = JupiterAggregator()
        jup._get_session = AsyncMock(return_value=session)
        jup._wait_for_global_slot = AsyncMock()
        jup._max_retries = 2
        return jup

    async def test_real_session_has_short_request_timeout(self):
        jup = JupiterAggregator()
        try:
            session = await jup._get_session()
            self.assertLessEqual(session.timeout.total, 6.0)
        finally:
            await jup.close()

    async def test_excluded_dex_labels_are_comma_separated(self):
        session = _Session(_QuoteResponse())
        quote = await self._aggregator(session).get_quote(
            "input", "output", 100, exclude_dexes=["GoonFi", "Orca V2"]
        )
        self.assertEqual(quote["outAmount"], "123")
        self.assertEqual(session.calls[0][1]["excludeDexes"], "GoonFi,Orca V2")

    async def test_timeout_gets_bounded_retry_without_route_change(self):
        session = _Session(asyncio.TimeoutError("timed out"), _QuoteResponse())
        with patch("execution.jupiter.asyncio.sleep", new_callable=AsyncMock) as sleep:
            quote = await self._aggregator(session).get_quote(
                "input", "output", 100, exclude_dexes="GoonFi"
            )
        self.assertEqual(quote["outAmount"], "123")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0][1], session.calls[1][1])
        sleep.assert_awaited_once_with(0.5)

    async def test_transport_retry_is_bounded(self):
        session = _Session(*[asyncio.TimeoutError("timed out") for _ in range(3)])
        with patch("execution.jupiter.asyncio.sleep", new_callable=AsyncMock) as sleep:
            quote = await self._aggregator(session).get_quote("input", "output", 100)
        self.assertIsNone(quote)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_certificate_mismatch_is_not_retried(self):
        session = _Session(_CertificateError())
        with self.assertRaises(_CertificateError):
            await self._aggregator(session).get_quote("input", "output", 100)
        self.assertEqual(len(session.calls), 1)

    async def test_program_label_lookup_validates_and_caches_mapping(self):
        program_id = str(Keypair.from_seed(bytes(range(32))).pubkey())
        unknown_id = str(Keypair.from_seed(bytes(range(1, 33))).pubkey())
        session = _Session(_MappingResponse({program_id: "GoonFi"}))
        jup = self._aggregator(session)

        self.assertEqual(await jup.get_program_id_label(program_id), "GoonFi")
        self.assertIsNone(await jup.get_program_id_label(unknown_id))
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0][0].endswith("/program-id-to-label"))
        self.assertLessEqual(session.calls[0][3].total, 3.0)

    async def test_invalid_program_mapping_fails_closed_without_cache(self):
        program_id = str(Keypair.from_seed(bytes(range(32))).pubkey())
        session = _Session(
            _MappingResponse({program_id: "GoonFi", "not a pubkey": "Other"}),
            _MappingResponse({program_id: "GoonFi"}),
        )
        jup = self._aggregator(session)

        self.assertIsNone(await jup.get_program_id_label(program_id))
        self.assertEqual(await jup.get_program_id_label(program_id), "GoonFi")
        self.assertEqual(len(session.calls), 2)

    async def test_program_mapping_429_retries_once_then_fails_closed(self):
        program_id = str(Keypair.from_seed(bytes(range(32))).pubkey())
        session = _Session(
            _MappingResponse({}, status=429),
            _MappingResponse({}, status=429),
        )
        with patch("execution.jupiter.asyncio.sleep", new_callable=AsyncMock) as sleep:
            label = await self._aggregator(session).get_program_id_label(program_id)
        self.assertIsNone(label)
        self.assertEqual(len(session.calls), 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_program_mapping_tls_error_does_not_retry(self):
        program_id = str(Keypair.from_seed(bytes(range(32))).pubkey())
        session = _Session(_CertificateError())
        self.assertIsNone(await self._aggregator(session).get_program_id_label(program_id))
        self.assertEqual(len(session.calls), 1)

    async def test_expired_program_mapping_is_not_used_after_refresh_failure(self):
        program_id = str(Keypair.from_seed(bytes(range(32))).pubkey())
        session = _Session(
            _MappingResponse({program_id: "GoonFi"}),
            _MappingResponse({}, status=503),
            _MappingResponse({}, status=503),
        )
        jup = self._aggregator(session)
        now = [100.0]
        with (
            patch("execution.jupiter.time.monotonic", side_effect=lambda: now[0]),
            patch("execution.jupiter.asyncio.sleep", new_callable=AsyncMock),
        ):
            self.assertEqual(await jup.get_program_id_label(program_id), "GoonFi")
            now[0] += 3601.0
            self.assertIsNone(await jup.get_program_id_label(program_id))
        self.assertEqual(len(session.calls), 3)


if __name__ == "__main__":
    unittest.main()
