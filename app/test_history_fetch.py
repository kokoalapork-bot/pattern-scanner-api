import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

import pytest
import httpx

from app.data_sources import CoinGeckoClient, DEMO_HISTORY_MAX_DAYS


class DummyClient(CoinGeckoClient):
    def __init__(self, transport: httpx.AsyncBaseTransport):
        super().__init__()
        self.client = httpx.AsyncClient(
            base_url=self.auth.base_url,
            headers={
                "accept": "application/json",
                "user-agent": "crypto-pattern-scanner/1.0",
                self.auth.header_name: self.auth.api_key,
            },
            timeout=10.0,
            transport=transport,
        )


def prices_payload(n: int = 40) -> dict:
    return {"prices": [[1700000000000 + i * 86400000, float(i + 1)] for i in range(n)]}


@pytest.mark.asyncio
async def test_demo_days_are_capped_to_365():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["days"] == str(DEMO_HISTORY_MAX_DAYS)
        return httpx.Response(200, json=prices_payload(50), request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is True
    assert result.request_params["requested_days"] == "450"
    assert result.request_params["days"] == "365"
    assert result.request_params["days_capped_by_plan"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "history_http_401"),
        (403, "history_http_403"),
        (404, "history_http_404"),
        (429, "rate_limited"),
    ],
)
async def test_history_http_mapping(status, reason):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "x"}, request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 365, "daily")
    await client.close()

    assert result.ok is False
    assert result.reason == reason
