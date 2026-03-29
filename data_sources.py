
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .config import get_settings


COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"


class CoinGeckoClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.settings.user_agent}
        if self.settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key
        return headers

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, headers=self._headers()) as client:
            response = await client.get(f"{COINGECKO_API_BASE}{path}", params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_markets(
        self,
        vs_currency: str = "usd",
        page: int = 1,
        per_page: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/coins/markets",
            {
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
        )

    async def search_symbol(self, query: str) -> list[dict[str, Any]]:
        data = await self._get("/search", {"query": query})
        return data.get("coins", [])

    async def fetch_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 450) -> dict[str, Any]:
        return await self._get(
            f"/coins/{coin_id}/market_chart",
            {"vs_currency": vs_currency, "days": days, "interval": "daily"},
        )

    async def fetch_coin(self, coin_id: str) -> dict[str, Any]:
        return await self._get(f"/coins/{coin_id}", {"localization": "false", "tickers": "false",
                                                      "community_data": "false", "developer_data": "false"})

    @staticmethod
    def age_days_from_iso(date_str: str | None) -> int | None:
        if not date_str:
            return None
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
