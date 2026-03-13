import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

import pytest
import httpx

from app.data_sources import CoinGeckoClient


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
async def test_history_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=prices_payload(50), request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is True
    assert result.http_status == 200
    assert result.auth_header_name == "x-cg-demo-api-key"


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
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is False
    assert result.reason == reason


@pytest.mark.asyncio
async def test_history_bad_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": "bar"}, request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is False
    assert result.reason == "history_bad_response_schema"


@pytest.mark.asyncio
async def test_history_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"prices": []}, request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is False
    assert result.reason == "history_empty"


@pytest.mark.asyncio
async def test_history_timeout():
    def handler(request: httpx.Request):
        raise httpx.ReadTimeout("timeout", request=request)

    client = DummyClient(httpx.MockTransport(handler))
    result = await client.fetch_market_history("bitcoin", "usd", 450, "daily")
    await client.close()

    assert result.ok is False
    assert result.reason == "timeout"
