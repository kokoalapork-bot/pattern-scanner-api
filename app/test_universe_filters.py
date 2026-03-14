from types import SimpleNamespace
import asyncio
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


def test_rate_limited_reason(monkeypatch):
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
    resp = asyncio.run(scan_pattern(req))

    assert resp.resolved_symbols == ["SIREN", "RIVER"]
    assert resp.evaluated_count == 0
    assert "SIREN" in resp.skipped_symbols
    assert "RIVER" in resp.skipped_symbols
    assert resp.skip_reasons["id:siren"] == "rate_limited"
    assert resp.skip_reasons["id:river"] == "rate_limited"
    assert resp.debug_by_symbol["id:siren"].stage == "fetch_market_data"


def test_empty_history_reason(monkeypatch):
    from app import services

    markets = [
        {"id": "xcoin", "symbol": "xcn", "name": "X Coin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "xcoin": MarketDataFetchResult(ok=True, chart={"prices": []}),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))
    monkeypatch.setattr(services, "classify_behavioral_universe_filter", lambda closes: ("included_for_scoring", "included_for_scoring"))

    req = ScanRequest(symbols=["XCN"], min_age_days=14, max_age_days=450, top_k=10)
    resp = asyncio.run(scan_pattern(req))

    assert resp.skip_reasons["id:xcoin"] == "history_empty"


def test_unresolved_symbol(monkeypatch):
    from app import services

    markets = []
    fetch_map = {}

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["NOPE"], min_age_days=14, max_age_days=450, top_k=10)
    resp = asyncio.run(scan_pattern(req))

    assert resp.unresolved_symbols == ["NOPE"]
    assert resp.skip_reasons["symbol:NOPE"] == "unresolved_symbol"


def test_debug_invariants(monkeypatch):
    from app import services

    prices = [[1_700_000_000_000 + i * 86_400_000, 1.0 + (i % 10) * 0.01] for i in range(80)]
    markets = [
        {"id": "xcoin", "symbol": "xcn", "name": "X Coin", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "xcoin": MarketDataFetchResult(ok=True, chart={"prices": prices}),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))

    req = ScanRequest(symbols=["XCN"], min_age_days=14, max_age_days=60, top_k=10)
    resp = asyncio.run(scan_pattern(req))

    if resp.evaluated_count > 0:
        assert len(resp.evaluated_symbols) > 0

    for sym in resp.skipped_symbols:
        symbol_key = f"symbol:{sym}"
        has_matching_key = symbol_key in resp.skip_reasons or any(
            dbg.input_symbol == sym and key in resp.skip_reasons
            for key, dbg in resp.debug_by_symbol.items()
        )
        assert has_matching_key

    assert "XCN" in set(resp.resolved_symbols) | set(resp.unresolved_symbols) | set(resp.evaluated_symbols) | set(resp.skipped_symbols)
