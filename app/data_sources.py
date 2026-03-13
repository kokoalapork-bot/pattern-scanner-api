from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import settings

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

STABLE_SYMBOLS = {
    "usdt", "usdc", "dai", "fdusd", "tusd", "gusd", "frax", "frxusd", "usdh", "dusd", "usdon", "fidd"
}

TOKENIZED_STOCK_KEYWORDS = [
    "tokenized stock",
    "tokenized etf",
    "xstock",
    "ondo",
    "tesla",
    "alphabet",
    "nvidia",
    "meta",
    "micron",
    "sp500",
    "nasdaq",
    "ishares",
    "silver trust",
    "gold",
]


@dataclass
class MarketDataFetchResult:
    ok: bool
    reason: str | None = None
    chart: dict[str, Any] | None = None
    error_message: str | None = None


class CoinGeckoClient:
    def __init__(self) -> None:
        headers = {"accept": "application/json"}

        # Для demo key нужен x-cg-demo-api-key
        if settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = settings.coingecko_api_key

        self.client = httpx.AsyncClient(
            base_url=COINGECKO_BASE,
            headers=headers,
            timeout=30.0,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def get_markets(self, vs_currency: str = "usd", pages: int = 3, per_page: int = 250) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        for page in range(1, pages + 1):
            resp = await self.client.get(
                "/coins/markets",
                params={
                    "vs_currency": vs_currency,
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                    "sparkline": "false",
                    "price_change_percentage": "24h,7d",
                },
            )
            resp.raise_for_status()
            items.extend(resp.json())

        return items

    async def get_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 450) -> dict[str, Any]:
        resp = await self.client.get(
            f"/coins/{coin_id}/market_chart",
            params={
                "vs_currency": vs_currency,
                "days": str(days),
                "interval": "daily",
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch_market_chart_safe(self, coin_id: str, vs_currency: str = "usd", days: int = 450) -> MarketDataFetchResult:
        if not coin_id:
            return MarketDataFetchResult(ok=False, reason="coingecko_id_missing")

        try:
            chart = await self.get_market_chart(coin_id=coin_id, vs_currency=vs_currency, days=days)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429:
                return MarketDataFetchResult(ok=False, reason="rate_limited", error_message=str(e))
            if code == 404:
                return MarketDataFetchResult(ok=False, reason="history_fetch_failed", error_message=str(e))
            return MarketDataFetchResult(ok=False, reason="history_fetch_failed", error_message=str(e))
        except httpx.RequestError as e:
            return MarketDataFetchResult(ok=False, reason="network_error", error_message=str(e))
        except Exception as e:
            return MarketDataFetchResult(ok=False, reason="market_data_fetch_failed", error_message=f"{type(e).__name__}: {e}")

        if not isinstance(chart, dict):
            return MarketDataFetchResult(ok=False, reason="bad_response_schema", error_message="chart is not dict")

        prices = chart.get("prices")
        if prices is None:
            return MarketDataFetchResult(ok=False, reason="bad_response_schema", error_message="missing prices")
        if not isinstance(prices, list):
            return MarketDataFetchResult(ok=False, reason="bad_response_schema", error_message="prices is not list")
        if len(prices) == 0:
            return MarketDataFetchResult(ok=False, reason="empty_history", error_message="empty prices")

        return MarketDataFetchResult(ok=True, chart=chart)


def looks_like_stable(symbol: str, name: str) -> bool:
    s = symbol.lower()
    n = name.lower()
    return s in STABLE_SYMBOLS or " usd" in n or n.endswith(" usd") or "stable" in n


def looks_like_tokenized_stock(name: str) -> bool:
    n = name.lower()
    return any(keyword in n for keyword in TOKENIZED_STOCK_KEYWORDS)


def coingecko_daily_closes(chart: dict[str, Any]) -> list[float]:
    prices = chart.get("prices", [])
    closes: list[float] = []
    seen_days: set[str] = set()

    for item in prices:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        ts_ms, price = item[0], item[1]
        try:
            day_key = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            price_f = float(price)
        except Exception:
            continue

        if day_key in seen_days:
            closes[-1] = price_f
        else:
            seen_days.add(day_key)
            closes.append(price_f)

    return closes
