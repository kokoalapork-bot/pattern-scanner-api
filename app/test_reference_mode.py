import httpx
import pytest

from app.data_sources import CoinGeckoClient


@pytest.mark.asyncio
async def test_fetch_market_chart_demo_caps_days_and_uses_demo_auth(monkeypatch):
    class DummyResponse:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"prices": []}

    calls = []

    class DummyAsyncClient:
        def __init__(self, *args, headers=None, **kwargs):
            self.headers = headers or {}
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def get(self, url, params=None):
            calls.append((url, dict(params or {}), dict(self.headers)))
            return DummyResponse()

    monkeypatch.setenv("COINGECKO_API_KEY", "demo-key")
    monkeypatch.setenv("COINGECKO_API_PLAN", "demo")
    from app.config import get_settings
    get_settings.cache_clear()

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
    client = CoinGeckoClient()
    await client.fetch_market_chart("river", days=450)

    assert calls
    url, params, headers = calls[0]
    assert "api.coingecko.com/api/v3/coins/river/market_chart" in url
    assert params["days"] == 365
    assert params["x_cg_demo_api_key"] == "demo-key"
    assert headers["x-cg-demo-api-key"] == "demo-key"

    get_settings.cache_clear()
