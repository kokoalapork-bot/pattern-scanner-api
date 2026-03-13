from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import asyncio

import httpx

from .config import settings

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

DEMO_HISTORY_MAX_DAYS = 365


class CoinGeckoConfigError(RuntimeError):
    pass


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
    auth_header_name: str | None = None


@dataclass(frozen=True)
class CoinGeckoAuth:
    mode: str
    base_url: str
    header_name: str
    api_key: str
    api_key_present: bool


def build_coingecko_auth() -> CoinGeckoAuth:
    auth_mode = settings.coingecko_auth_mode
    api_key = settings.coingecko_api_key.strip()
    base_url = settings.coingecko_effective_base_url.rstrip("/")
    header_name = settings.coingecko_header_name

    if not api_key:
        raise CoinGeckoConfigError("COINGECKO_API_KEY is missing or blank")

    if auth_mode == "demo":
        if "pro-api.coingecko.com" in base_url:
            raise CoinGeckoConfigError("demo mode cannot use pro-api base URL")
        if header_name != "x-cg-demo-api-key":
            raise CoinGeckoConfigError("demo mode must use x-cg-demo-api-key")
    elif auth_mode == "pro":
        if "pro-api.coingecko.com" not in base_url:
            raise CoinGeckoConfigError("pro mode must use pro-api base URL")
        if header_name != "x-cg-pro-api-key":
            raise CoinGeckoConfigError("pro mode must use x-cg-pro-api-key")
    else:
        raise CoinGeckoConfigError(f"Unsupported auth mode: {auth_mode}")

    return CoinGeckoAuth(
        mode=auth_mode,
        base_url=base_url,
        header_name=header_name,
        api_key=api_key,
        api_key_present=True,
    )


def get_history_plan_limit_days(auth_mode: str) -> int | None:
    if auth_mode == "demo":
        return DEMO_HISTORY_MAX_DAYS
    return None


class CoinGeckoClient:
    def __init__(self) -> None:
        self.auth = build_coingecko_auth()
        self.client = httpx.AsyncClient(
            base_url=self.auth.base_url,
            headers={
                "accept": "application/json",
                "user-agent": "crypto-pattern-scanner/1.0",
                self.auth.header_name: self.auth.api_key,
            },
            timeout=settings.request_timeout_seconds,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def auth_debug(self) -> dict[str, Any]:
        return {
            "auth_mode": self.auth.mode,
            "base_url": self.auth.base_url,
            "api_key_present": self.auth.api_key_present,
            "auth_header_name": self.auth.header_name,
        }

    def normalize_history_days(self, requested_days: int | str) -> tuple[int, bool]:
        try:
            days_int = int(requested_days)
        except Exception:
            days_int = 365

        plan_limit = get_history_plan_limit_days(self.auth.mode)
        if plan_limit is not None and days_int > plan_limit:
            return plan_limit, True
        return days_int, False

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
            payload = resp.json()
            if not isinstance(payload, list):
                raise ValueError("CoinGecko /coins/markets returned non-list payload")
            items.extend(payload)

        return items

    async def fetch_market_history(
        self,
        coingecko_id: str,
        vs_currency: str = "usd",
        days: int | str = 450,
        interval: str = "daily",
    ) -> MarketDataFetchResult:
        if not coingecko_id:
            normalized_days, capped = self.normalize_history_days(days)
            return MarketDataFetchResult(
                ok=False,
                reason="coingecko_id_missing",
                endpoint="/coins/{id}/market_chart",
                request_params={
                    "vs_currency": vs_currency,
                    "days": str(normalized_days),
                    "interval": interval,
                    "requested_days": str(days),
                    "plan_limit_days": get_history_plan_limit_days(self.auth.mode),
                    "days_capped_by_plan": capped,
                },
                **self.auth_debug(),
            )

        normalized_days, capped = self.normalize_history_days(days)
        endpoint = f"/coins/{coingecko_id}/market_chart"
        request_params = {
            "vs_currency": vs_currency,
            "days": str(normalized_days),
            "interval": interval,
            "requested_days": str(days),
            "plan_limit_days": get_history_plan_limit_days(self.auth.mode),
            "days_capped_by_plan": capped,
        }

        retries = max(1, settings.history_retry_count)
        backoff_base = max(0.1, settings.history_backoff_base_seconds)
        chart: Optional[dict[str, Any]] = None

        for attempt in range(retries):
            try:
                resp = await self.client.get(endpoint, params={
                    "vs_currency": vs_currency,
                    "days": str(normalized_days),
                    "interval": interval,
                })
                resp.raise_for_status()
                payload = resp.json()

                if not isinstance(payload, dict):
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_bad_response_schema",
                        error_message=f"chart is not a dict: {type(payload).__name__}",
                        endpoint="/coins/{id}/market_chart",
                        http_status=200,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                prices = payload.get("prices")
                if prices is None:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_bad_response_schema",
                        error_message="missing 'prices' field",
                        endpoint="/coins/{id}/market_chart",
                        http_status=200,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if not isinstance(prices, list):
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_bad_response_schema",
                        error_message="'prices' is not a list",
                        endpoint="/coins/{id}/market_chart",
                        http_status=200,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if len(prices) == 0:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_empty",
                        error_message="prices list is empty",
                        endpoint="/coins/{id}/market_chart",
                        http_status=200,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                chart = payload
                break

            except httpx.TimeoutException as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                    continue
                return MarketDataFetchResult(
                    ok=False,
                    reason="timeout",
                    error_message=str(e),
                    endpoint="/coins/{id}/market_chart",
                    request_params=request_params,
                    **self.auth_debug(),
                )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body_text = ""
                try:
                    body_text = e.response.text[:500]
                except Exception:
                    body_text = ""
                error_message = str(e) if not body_text else f"{str(e)} | body={body_text}"

                if status == 429:
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff_base * (2 ** attempt))
                        continue
                    return MarketDataFetchResult(
                        ok=False,
                        reason="rate_limited",
                        error_message=error_message,
                        endpoint="/coins/{id}/market_chart",
                        http_status=status,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if status == 401:
                    if self.auth.mode == "demo" and int(request_params["requested_days"]) > DEMO_HISTORY_MAX_DAYS:
                        return MarketDataFetchResult(
                            ok=False,
                            reason="history_range_exceeds_plan_limit",
                            error_message=(
                                f"Demo plan requested {request_params['requested_days']} days, "
                                f"plan limit is {DEMO_HISTORY_MAX_DAYS}. Original upstream error: {error_message}"
                            ),
                            endpoint="/coins/{id}/market_chart",
                            http_status=status,
                            request_params=request_params,
                            **self.auth_debug(),
                        )
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_401",
                        error_message=error_message,
                        endpoint="/coins/{id}/market_chart",
                        http_status=status,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if status == 403:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_403",
                        error_message=error_message,
                        endpoint="/coins/{id}/market_chart",
                        http_status=status,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if status == 404:
                    return MarketDataFetchResult(
                        ok=False,
                        reason="history_http_404",
                        error_message=error_message,
                        endpoint="/coins/{id}/market_chart",
                        http_status=status,
                        request_params=request_params,
                        **self.auth_debug(),
                    )

                if 500 <= status <= 599 and attempt < retries - 1:
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                    continue

                return MarketDataFetchResult(
                    ok=False,
                    reason="history_http_error",
                    error_message=error_message,
                    endpoint="/coins/{id}/market_chart",
                    http_status=status,
                    request_params=request_params,
                    **self.auth_debug(),
                )

            except httpx.RequestError as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                    continue
                return MarketDataFetchResult(
                    ok=False,
                    reason="network_error",
                    error_message=str(e),
                    endpoint="/coins/{id}/market_chart",
                    request_params=request_params,
                    **self.auth_debug(),
                )

            except Exception as e:
                return MarketDataFetchResult(
                    ok=False,
                    reason="history_fetch_failed",
                    error_message=f"{type(e).__name__}: {e}",
                    endpoint="/coins/{id}/market_chart",
                    request_params=request_params,
                    **self.auth_debug(),
                )

        if chart is None:
            return MarketDataFetchResult(
                ok=False,
                reason="history_fetch_failed",
                error_message="retry loop exhausted unexpectedly",
                endpoint="/coins/{id}/market_chart",
                request_params=request_params,
                **self.auth_debug(),
            )

        return MarketDataFetchResult(
            ok=True,
            chart=chart,
            endpoint="/coins/{id}/market_chart",
            http_status=200,
            request_params=request_params,
            **self.auth_debug(),
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
