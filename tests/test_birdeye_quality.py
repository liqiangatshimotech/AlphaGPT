import unittest
from unittest.mock import AsyncMock
from data_pipeline.providers.birdeye import BirdeyeProvider


class BirdeyeQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_historical_pool_fields_are_not_backfilled(self):
        provider=BirdeyeProvider()
        provider._get_json=AsyncMock(return_value={'data': {'items': [
            {'unixTime': 120000, 'o': '1', 'h': '1.1', 'l': '.9', 'c': '1', 'v': '10'}
        ]}})
        rows=await provider.get_token_history(None,'token',liquidity=999999,fdv=123456,
                                              start_time=120000,end_time=120060)
        self.assertEqual(len(rows),1)
        self.assertIsNone(rows[0][7])
        self.assertIsNone(rows[0][8])

    async def test_reported_pool_fields_are_preserved(self):
        provider=BirdeyeProvider()
        provider._get_json=AsyncMock(return_value={'data': {'items': [
            {'unixTime': 120000, 'o': '1', 'h': '1.1', 'l': '.9', 'c': '1', 'v': '10',
             'liquidity': '1234', 'fdv': '5678'}
        ]}})
        rows=await provider.get_token_history(None,'token',start_time=120000,end_time=120060)
        self.assertEqual(rows[0][7],1234.0)
        self.assertEqual(rows[0][8],5678.0)


if __name__=='__main__': unittest.main()
