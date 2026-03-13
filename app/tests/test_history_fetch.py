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
async def test_history_fetch_rate_limited(monkeypatch):
    from app import services

    markets = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "bitcoin": MarketDataFetchResult(
            ok=False,
            reason="rate_limited",
            endpoint="/coins/{id}/market_chart",
            http_status=429,
            request_params={"vs_currency": "usd", "days": "450", "interval": "daily"},
        )
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["BTC"], min_age_days=14, max_age_days=450, top_k=10)
    resp = await scan_pattern(req)

    assert resp.skip_reasons["BTC"] == "rate_limited"
    assert resp.debug_by_symbol["BTC"].http_status == 429
    assert resp.debug_by_symbol["BTC"].endpoint == "/coins/{id}/market_chart"


@pytest.mark.asyncio
async def test_history_fetch_empty(monkeypatch):
    from app import services

    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "river": MarketDataFetchResult(
            ok=True,
            chart={"prices": []},
            endpoint="/coins/{id}/market_chart",
            request_params={"vs_currency": "usd", "days": "450", "interval": "daily"},
        )
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["RIVER"], min_age_days=14, max_age_days=450, top_k=10)
    resp = await scan_pattern(req)

    assert resp.skip_reasons["RIVER"] == "history_empty"


@pytest.mark.asyncio
async def test_history_fetch_uses_resolved_coingecko_id(monkeypatch):
    from app import services

    markets = [
        {"id": "siren-2", "symbol": "siren", "name": "SIREN", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]

    called_ids = []

    class DummyClient2(DummyClient):
        async def fetch_market_chart_safe(self, coin_id: str, vs_currency="usd", days=450):
            called_ids.append(coin_id)
            return MarketDataFetchResult(
                ok=False,
                reason="history_not_found",
                endpoint="/coins/{id}/market_chart",
                http_status=404,
                request_params={"vs_currency": "usd", "days": "450", "interval": "daily"},
            )

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient2(markets, {}))

    req = ScanRequest(symbols=["SIREN"], min_age_days=14, max_age_days=450, top_k=10)
    resp = await scan_pattern(req)

    assert called_ids == ["siren-2"]
    assert resp.debug_by_symbol["SIREN"].coingecko_id == "siren-2"


@pytest.mark.asyncio
async def test_valid_history_reaches_scoring(monkeypatch):
    from app import services

    prices = [[1_700_000_000_000 + i * 86_400_000, 1.0 + (i % 10) * 0.01] for i in range(90)]

    markets = [
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "bitcoin": MarketDataFetchResult(
            ok=True,
            chart={"prices": prices},
            endpoint="/coins/{id}/market_chart",
            request_params={"vs_currency": "usd", "days": "450", "interval": "daily"},
        )
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["BTC"], min_age_days=14, max_age_days=60, top_k=10)
    resp = await scan_pattern(req)

    assert resp.evaluated_count >= 1
    assert "BTC" in resp.evaluated_symbols
    assert resp.debug_by_symbol["BTC"].stage == "score_windows"
