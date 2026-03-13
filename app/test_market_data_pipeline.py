import pytest

from app.data_sources import MarketDataFetchResult
from app.services import scan_pattern
from app.models import ScanRequest


class DummyClient:
    def __init__(self, markets, fetch_map):
        self._markets = markets
        self._fetch_map = fetch_map

    async def close(self):
        return None

    async def get_markets(self, vs_currency="usd", pages=1, per_page=250):
        return self._markets

    async def fetch_market_chart_safe(self, coin_id: str, vs_currency="usd", days=450):
        return self._fetch_map[coin_id]


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
    assert resp.evaluated_count == 0
    assert "SIREN" in resp.skipped_symbols
    assert "RIVER" in resp.skipped_symbols
    assert resp.skip_reasons["SIREN"] == "rate_limited"
    assert resp.skip_reasons["RIVER"] == "rate_limited"
    assert resp.debug_by_symbol["SIREN"].stage == "fetch_market_data"


@pytest.mark.asyncio
async def test_empty_history_reason(monkeypatch):
    from app import services

    markets = [
        {"id": "btc", "symbol": "btc", "name": "Bitcoin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "btc": MarketDataFetchResult(ok=True, chart={"prices": []}),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["BTC"], min_age_days=14, max_age_days=450, top_k=10)
    resp = await scan_pattern(req)

    assert resp.skip_reasons["BTC"] == "empty_history"


@pytest.mark.asyncio
async def test_unresolved_symbol(monkeypatch):
    from app import services

    markets = []
    fetch_map = {}

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["NOPE"], min_age_days=14, max_age_days=450, top_k=10)
    resp = await scan_pattern(req)

    assert resp.unresolved_symbols == ["NOPE"]
    assert resp.skip_reasons["NOPE"] == "unresolved_symbol"


@pytest.mark.asyncio
async def test_debug_invariants(monkeypatch):
    from app import services

    prices = [[1_700_000_000_000 + i * 86_400_000, 1.0 + (i % 10) * 0.01] for i in range(80)]
    markets = [
        {"id": "btc", "symbol": "btc", "name": "Bitcoin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "btc": MarketDataFetchResult(ok=True, chart={"prices": prices}),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["BTC"], min_age_days=14, max_age_days=60, top_k=10)
    resp = await scan_pattern(req)

    if resp.evaluated_count > 0:
        assert len(resp.evaluated_symbols) > 0

    for sym in resp.skipped_symbols:
        assert sym in resp.skip_reasons

    assert "BTC" in set(resp.resolved_symbols) | set(resp.unresolved_symbols) | set(resp.evaluated_symbols) | set(resp.skipped_symbols)
