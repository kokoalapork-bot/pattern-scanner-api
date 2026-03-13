# data_sources.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict
import asyncio

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
    endpoint: str | None = None
    http_status: int | None = None
    request_params: Dict[str, Any] | None = None
    auth_mode: str | None = None
    base_url: str | None = None
    api_key_present: bool | None = None


class CoinGeckoClient:
    def __init__(self) -> None:
        headers = {"accept": "application/json"}

        # У тебя demo key: для Demo API нужен именно x-cg-demo-api-key
        api_key = str(getattr(settings, "coingecko_api_key", "") or "").strip()
        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        self.auth_mode = "demo"
        self.base_url = COINGECKO_BASE
        self.api_key_present = bool(api_key)

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def auth_debug(self) -> dict[str, Any]:
        return {
            "auth_mode": self.auth_mode,
            "base_url": self.base_url,
            "api_key_present": self.api_key_present,
        }

    async def get_markets(
        self,
        vs_currency: str = "usd",
        pages: int = 3,
        per_page: int = 250,
    ) -> list[dict[str, Any]]:
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

    async def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int | str = 450,
    ) -> dict[str, Any]:
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
            "interval": "daily",
        }
        resp = await self.client.get(endpoint, params=params)
        resp.raise_for_status()
        return resp.json()

    async def fetch_market_chart_safe(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int | str = 450,
    ) -> MarketDataFetchResult:
        """
        Fetch history by coingecko_id only.
        Demo auth only:
          - base_url: https://api.coingecko.com/api/v3
          - header: x-cg-demo-api-key
        Adds retry/backoff for 429 and returns structured debug info.
        """
        if not coin_id:
            return MarketDataFetchResult(
                ok=False,
                reason="coingecko_id_missing",
                endpoint="/coins/{id}/market_chart",
                request_params={"vs_currency": vs_currency, "days": str(days), "interval": "daily"},
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
            "interval": "daily",
        }

        retries = 3
        backoffs = [0.8, 1.6, 3.2]
        chart: dict[str, Any] | None = None

        for attempt in range(retries):
            try:
                resp = await self.client.get(endpoint, params=params)
                resp.raise_for_status()
                chart = resp.json()
                break

            except httpx.HTTPStatusError as e:
                code = e.response.status_code

                if code == 429:
                    if attempt < retries - 1:
                        await asyncio.sleep(backoffs[attempt])
                        continue
                    return MarketDataFetchResult(
                        ok=False,
                        reason="rate_limited",
                        error_message=str(e),
                        endpoint="/coins/{id}/market_chart",
                        http_status=code,
                        request_params=params,
                        auth_mode=self.auth_mode,
                        base_url=self.base_url,
                        api_key_present=self.api_key_present,
                    )

                if code == 401:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_401",
                        error_message=str(e),
                        endpoint="/coins/{id}/market_chart",
                        http_status=code,
                        request_params=params,
                        auth_mode=self.auth_mode,
                        base_url=self.base_url,
                        api_key_present=self.api_key_present,
                    )

                if code == 403:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_403",
                        error_message=str(e),
                        endpoint="/coins/{id}/market_chart",
                        http_status=code,
                        request_params=params,
                        auth_mode=self.auth_mode,
                        base_url=self.base_url,
                        api_key_present=self.api_key_present,
                    )

                if code == 404:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_404",
                        error_message=str(e),
                        endpoint="/coins/{id}/market_chart",
                        http_status=code,
                        request_params=params,
                        auth_mode=self.auth_mode,
                        base_url=self.base_url,
                        api_key_present=self.api_key_present,
                    )

                return MarketDataFetchResult(
                    ok=False,
                    reason="history_http_error",
                    error_message=str(e),
                    endpoint="/coins/{id}/market_chart",
                    http_status=code,
                    request_params=params,
                    auth_mode=self.auth_mode,
                    base_url=self.base_url,
                    api_key_present=self.api_key_present,
                )

            except httpx.RequestError as e:
                return MarketDataFetchResult(
                    ok=False,
                    reason="network_error",
                    error_message=str(e),
                    endpoint="/coins/{id}/market_chart",
                    request_params=params,
                    auth_mode=self.auth_mode,
                    base_url=self.base_url,
                    api_key_present=self.api_key_present,
                )
            except Exception as e:
                return MarketDataFetchResult(
                    ok=False,
                    reason="history_fetch_failed",
                    error_message=f"{type(e).__name__}: {e}",
                    endpoint="/coins/{id}/market_chart",
                    request_params=params,
                    auth_mode=self.auth_mode,
                    base_url=self.base_url,
                    api_key_present=self.api_key_present,
                )
        else:
            return MarketDataFetchResult(
                ok=False,
                reason="history_fetch_failed",
                error_message="retry loop exhausted unexpectedly",
                endpoint="/coins/{id}/market_chart",
                request_params=params,
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        if not isinstance(chart, dict):
            return MarketDataFetchResult(
                ok=False,
                reason="history_bad_response_schema",
                error_message="chart is not a dict",
                endpoint="/coins/{id}/market_chart",
                request_params=params,
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        prices = chart.get("prices")
        if prices is None:
            return MarketDataFetchResult(
                ok=False,
                reason="history_bad_response_schema",
                error_message="missing 'prices' field",
                endpoint="/coins/{id}/market_chart",
                request_params=params,
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        if not isinstance(prices, list):
            return MarketDataFetchResult(
                ok=False,
                reason="history_bad_response_schema",
                error_message="'prices' is not a list",
                endpoint="/coins/{id}/market_chart",
                request_params=params,
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        if len(prices) == 0:
            return MarketDataFetchResult(
                ok=False,
                reason="history_empty",
                error_message="prices list is empty",
                endpoint="/coins/{id}/market_chart",
                request_params=params,
                auth_mode=self.auth_mode,
                base_url=self.base_url,
                api_key_present=self.api_key_present,
            )

        return MarketDataFetchResult(
            ok=True,
            chart=chart,
            endpoint="/coins/{id}/market_chart",
            request_params=params,
            auth_mode=self.auth_mode,
            base_url=self.base_url,
            api_key_present=self.api_key_present,
        )


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
            seen_days.add(day_key)
            closes.append(price_f)

    return closes
