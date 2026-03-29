
from datetime import datetime, timedelta, timezone

import pytest

from app.models import ScanRequest
from app.services import SYMBOL_ID_OVERRIDES, scan_pattern


def _make_series():
    prices = []
    prices += [10 + i * 0.1 for i in range(20)]
    prices += [13.5, 14.0, 14.3, 14.5, 14.4, 14.6, 14.5, 14.55, 14.35, 14.4, 14.2, 14.25]
    prices += [14.1, 14.0, 13.8, 13.7, 13.5, 13.2, 12.8, 12.1, 11.5, 10.9, 10.2, 9.8]
    prices += [9.7, 9.6, 9.65, 9.7, 9.6, 9.7, 9.8, 9.7, 9.75, 9.7]
    prices += [10.1, 10.8, 11.5, 10.4, 9.9, 9.8, 9.7]
    base = datetime(2025, 9, 1, tzinfo=timezone.utc)
    return [[int((base + timedelta(days=i)).timestamp() * 1000), p] for i, p in enumerate(prices)]


class DummyClient:
    async def fetch_coin(self, coin_id: str):
        return {
            "id": coin_id,
            "symbol": "river" if coin_id == "river" else "siren",
            "name": "River" if coin_id == "river" else "Siren",
            "market_data": {
                "market_cap": {"usd": 2_000_000},
                "total_volume": {"usd": 200_000},
                "ath_date": {"usd": "2025-10-25T00:00:00.000Z"},
                "market_cap_rank": 9999,
            },
            "genesis_date": None,
        }

    async def fetch_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 450):
        return {"prices": _make_series()}

    async def search_symbol(self, query: str):
        return [{"id": SYMBOL_ID_OVERRIDES[query.upper()], "symbol": query.upper(), "market_cap_rank": 100}]


@pytest.mark.asyncio
async def test_explicit_reference_symbols_are_evaluated():
    req = ScanRequest(symbols=["RIVER", "SIREN"], top_k=5, max_coins_to_evaluate=10, include_notes=False)
    resp = await scan_pattern(req, client=DummyClient())
    assert resp.evaluated_count == 2
    assert {r.coingecko_id for r in resp.results} == {"river", "siren-2"}
    assert resp.debug_by_symbol["RIVER"].status == "evaluated"
    assert resp.debug_by_symbol["SIREN"].status == "evaluated"
