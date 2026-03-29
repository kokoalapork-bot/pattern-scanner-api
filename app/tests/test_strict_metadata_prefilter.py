
from datetime import datetime, timedelta, timezone

import pytest

from app.models import ScanRequest
from app.services import scan_pattern


def _dated_series(start: datetime, values: list[float]):
    return [[int((start + timedelta(days=i)).timestamp() * 1000), p] for i, p in enumerate(values)]


class DummyClient:
    async def fetch_markets(self, vs_currency: str = "usd", page: int = 1, per_page: int = 50):
        return [
            {
                "id": "old-coin",
                "symbol": "OLD",
                "name": "Old Coin",
                "market_cap": 5_000_000,
                "total_volume": 500_000,
            },
            {
                "id": "new-coin",
                "symbol": "NEW",
                "name": "New Coin",
                "market_cap": 5_000_000,
                "total_volume": 500_000,
            },
        ]

    async def fetch_coin(self, coin_id: str):
        if coin_id == "old-coin":
            return {
                "id": coin_id,
                "symbol": "old",
                "name": "Old Coin",
                "genesis_date": "2022-06-01",
                "market_data": {
                    "market_cap": {"usd": 5_000_000},
                    "total_volume": {"usd": 500_000},
                    "ath_date": {"usd": "2022-06-02T00:00:00.000Z"},
                    "atl_date": {"usd": "2022-06-03T00:00:00.000Z"},
                    "market_cap_rank": 9999,
                },
            }
        return {
            "id": coin_id,
            "symbol": "new",
            "name": "New Coin",
            "genesis_date": "2025-04-01",
            "market_data": {
                "market_cap": {"usd": 5_000_000},
                "total_volume": {"usd": 500_000},
                "ath_date": {"usd": "2025-04-20T00:00:00.000Z"},
                "atl_date": {"usd": "2025-04-10T00:00:00.000Z"},
                "market_cap_rank": 9999,
            },
        }

    async def fetch_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 365):
        start = datetime(2025, 4, 1, tzinfo=timezone.utc)
        prices = [10 + i * 0.05 for i in range(80)]
        prices[20:30] = [14.0, 14.1, 14.3, 14.5, 14.4, 14.35, 14.3, 14.25, 14.2, 14.15]
        prices[30:45] = [13.9, 13.5, 13.1, 12.7, 12.3, 12.0, 11.8, 11.6, 11.45, 11.3, 11.2, 11.1, 11.0, 10.95, 10.9]
        prices[45:65] = [10.95, 10.9, 10.92, 10.91, 10.93, 10.95, 10.94, 10.96, 10.98, 11.0, 11.02, 11.01, 11.0, 10.99, 11.0, 11.02, 11.03, 11.04, 11.05, 11.06]
        prices[65:70] = [11.2, 11.4, 11.8, 11.3, 11.0]
        return {"prices": _dated_series(start, prices)}


@pytest.mark.asyncio
async def test_old_coins_are_rejected_by_metadata_prefilter():
    req = ScanRequest(top_k=10, max_coins_to_evaluate=10, market_batch_size=50)
    resp = await scan_pattern(req, client=DummyClient())
    ids = {r.coingecko_id for r in resp.results}
    assert "old-coin" not in ids
