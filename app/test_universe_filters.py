from types import SimpleNamespace
import pytest

from app.data_sources import MarketDataFetchResult
from app.services import scan_pattern
from app.models import ScanRequest


class DummyClient:
    def __init__(self, markets, fetch_map):
        self._markets = markets
        self._fetch_map = fetch_map
        self.auth = SimpleNamespace(mode="demo", base_url="https://api.coingecko.com/api/v3", api_key_present=True, header_name="x-cg-demo-api-key")

    async def close(self):
        return None

    async def get_markets(self, vs_currency="usd", pages=1, per_page=250):
        return self._markets

    async def fetch_market_chart_safe(self, coin_id: str, vs_currency="usd", days=450):
        return self._fetch_map[coin_id]

    async def fetch_market_history(self, coingecko_id: str, vs_currency="usd", days=450, interval="daily"):
        return self._fetch_map[coingecko_id]

    def normalize_history_days(self, requested_days: int):
        return requested_days, False


@pytest.mark.asyncio
async def test_rate_limited_reason(monkeypatch):
    from app import services

    markets = [
        {"id": "siren", "symbol": "siren", "name": "SIREN", "market_cap": 10_000_000, "total_volume": 1_000_000},
        {"id": "river", "symbol": "river", "name": "RIVER", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "siren": MarketDataFetchResult(ok=False, reason="rate_limited"),
        "river": MarketDataFetchResult(ok=False, reason="rate_limited"),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(
        symbols=["SIREN", "RIVER"],
        min_age_days=14,
        max_age_days=450,
        top_k=10,
    )
    resp = await scan_pattern(req)

    assert resp.resolved_symbols == ["SIREN", "RIVER"]
