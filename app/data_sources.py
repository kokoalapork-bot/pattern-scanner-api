from __future__ import annotations

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


class CoinGeckoClient:
    def __init__(self) -> None:
        headers = {"accept": "application/json"}
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

    async def get_coin(self, coin_id: str) -> dict[str, Any]:
        resp = await self.client.get(
            f"/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false",
            },
        )
        resp.raise_for_status()
        return resp.json()

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


def parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def age_in_days(genesis_date, fallback_from_history_ts_ms=None):
    if genesis_date is None and fallback_from_history_ts_ms is None:
        return None

    if genesis_date is None:
        genesis_date = datetime.fromtimestamp(fallback_from_history_ts_ms / 1000, tz=timezone.utc)

    now = datetime.now(timezone.utc)
    return max(0, (now - genesis_date).days)


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

    for ts_ms, price in prices:
        day_key = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_key in seen_days:
            closes[-1] = float(price)
        else:
            seen_days.add(day_key)
            closes.append(float(price))

    return closes
