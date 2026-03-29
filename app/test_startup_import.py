
from datetime import datetime, timedelta, timezone

import pytest

from app.models import ScanRequest
from app.services import scan_pattern


def _dated_series(start: datetime, values: list[float]):
    return [[int((start + timedelta(days=i)).timestamp() * 1000), p] for i, p in enumerate(values)]


def _river_prices():
    prices = [10 + (i * 0.03) for i in range(100)]
    prices[25:34] = [14.0, 14.2, 14.5, 14.6, 14.55, 14.52, 14.48, 14.45, 14.4]
    prices[34:46] = [14.3, 14.2, 14.0, 13.8, 13.5, 13.2, 12.8, 12.2, 11.7, 11.2, 10.8, 10.5]
    prices[46:70] = [10.4, 10.35, 10.3, 10.32, 10.28, 10.3, 10.31, 10.35, 10.33, 10.31, 10.3, 10.32, 10.31, 10.33, 10.34, 10.36, 10.39, 10.41, 10.43, 10.45, 10.47, 10.5, 10.52, 10.55]
    prices[70:75] = [10.7, 11.0, 11.5, 10.8, 10.55]
    return prices


def _siren_prices():
    n = 323
    prices = [5 + i * 0.01 for i in range(n)]
    prices[55:67] = [11.0, 11.4, 11.8, 12.0, 12.1, 12.05, 12.0, 11.95, 11.9, 11.85, 11.8, 11.75]
    prices[67:88] = [11.7, 11.6, 11.55, 11.5, 11.45, 11.4, 11.2, 11.0, 10.8, 10.5, 10.2, 9.9, 9.6, 9.4, 9.2, 9.0, 8.9, 8.8, 8.75, 8.7, 8.65]
    prices[88:160] = [8.7 + ((i % 5) * 0.03) for i in range(72)]
    prices[160:168] = [8.8, 9.0, 9.3, 9.7, 10.0, 9.4, 9.0, 8.85]
    return prices


class DummyClient:
    async def fetch_coin(self, coin_id: str):
        sym = "river" if coin_id == "river" else "siren"
        return {
            "id": coin_id,
            "symbol": sym,
            "name": sym.title(),
            "market_data": {
                "market_cap": {"usd": 2_000_000},
                "total_volume": {"usd": 200_000},
                "ath_date": {"usd": "2025-10-25T00:00:00.000Z"},
                "market_cap_rank": 9999,
            },
            "genesis_date": None,
        }

    async def fetch_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 450):
        if coin_id == "river":
            return {"prices": _dated_series(datetime(2025, 9, 22, tzinfo=timezone.utc), _river_prices())}
        return {"prices": _dated_series(datetime(2025, 3, 20, tzinfo=timezone.utc), _siren_prices())}

    async def search_symbol(self, query: str):
        q = query.upper()
        return [{"id": "river" if q == "RIVER" else "siren-2", "symbol": q, "market_cap_rank": 100}]


@pytest.mark.asyncio
async def test_explicit_reference_symbols_are_evaluated():
    req = ScanRequest(symbols=["RIVER", "SIREN"], top_k=5, max_coins_to_evaluate=10, include_notes=False)
    resp = await scan_pattern(req, client=DummyClient())
    assert resp.evaluated_count == 2
    ids = {r.coingecko_id for r in resp.results}
    assert ids == {"river", "siren-2"}
