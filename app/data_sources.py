from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import httpx

from .config import settings

DEMO_HISTORY_MAX_DAYS = 365


@dataclass
class AssetUniverseItem:
    symbol: str
    name: str
    provider: str
    coingecko_id: str | None = None
    coinmarketcap_id: int | None = None
    market_cap_usd: float | None = None
    volume_24h_usd: float | None = None
    listing_date: date | None = None
    stable_like: bool = False
    tokenized_stock_like: bool = False


@dataclass
class MarketDataFetchResult:
    ok: bool
    endpoint: str
    reason: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    request_params: dict[str, Any] | None = None
    chart: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


class AsyncRateLimiter:
    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, int(rpm))
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            while True:
                now = loop.time()
                while self._hits and now - self._hits[0] >= 60.0:
                    self._hits.popleft()
                if len(self._hits) < self.rpm:
                    self._hits.append(now)
                    return
                sleep_for = 60.0 - (now - self._hits[0]) + 0.05
                await asyncio.sleep(max(0.05, sleep_for))


def parse_date_safe(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def looks_like_stable(symbol: str | None, name: str | None) -> bool:
    text = f"{symbol or ''} {name or ''}".lower()
    triggers = ["usd", "usdt", "usdc", "busd", "fdusd", "dai", "tusd", "susd", "stable", "eurc", "eurt", "usde", "usds", "lusd"]
    if any(t in text.split() for t in triggers):
        return True
    return "stablecoin" in text or "bridged usdc" in text or "bridged usdt" in text


def looks_like_tokenized_stock(symbol: str | None, name: str | None) -> bool:
    text = f"{symbol or ''} {name or ''}".lower()
    triggers = ["tesla", "apple", "microsoft", "nvidia", "google", "alphabet", "amazon", "meta", "coinbase", "stock", "equity", "mirror"]
    return any(t in text for t in triggers)


def classify_behavioral_universe_filter(
    item: AssetUniverseItem,
    *,
    max_market_cap_usd: float | None,
    min_market_cap_usd: float | None,
    min_24h_volume_usd: float | None,
    min_listing_date_after: date | None,
    exclude_stables: bool,
    exclude_tokenized_stocks: bool,
    min_age_days: int,
    max_age_days: int,
    now_date: date,
) -> tuple[bool, str]:
    if exclude_stables and item.stable_like:
        return False, "stablecoin"
    if exclude_tokenized_stocks and item.tokenized_stock_like:
        return False, "tokenized_stock"
    if item.market_cap_usd is not None:
        if max_market_cap_usd is not None and item.market_cap_usd > max_market_cap_usd:
            return False, "market_cap_too_large"
        if min_market_cap_usd is not None and item.market_cap_usd < min_market_cap_usd:
            return False, "market_cap_too_small"
    if item.volume_24h_usd is not None and min_24h_volume_usd is not None and item.volume_24h_usd < min_24h_volume_usd:
        return False, "volume_too_small"
    if item.listing_date is not None:
        if min_listing_date_after and item.listing_date < min_listing_date_after:
            return False, "listed_too_early"
        age_days = max(0, (now_date - item.listing_date).days)
        if age_days < min_age_days:
            return False, "too_new"
        if age_days > max_age_days:
            return False, "too_old"
    return True, "included_for_scoring"


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
        if day_key in seen_days and closes:
            closes[-1] = price_f
        else:
            seen_days.add(day_key)
            closes.append(price_f)
    return closes


class CoinGeckoClient:
    def __init__(self) -> None:
        self.base_url = settings.coingecko_effective_base_url
        self.timeout = settings.request_timeout_seconds
        self.limiter = AsyncRateLimiter(settings.coingecko_soft_rpm_limit)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if settings.coingecko_api_key:
            headers[settings.coingecko_header_name] = settings.coingecko_api_key
        return headers

    async def fetch_market_universe_page(self, *, page: int, per_page: int, vs_currency: str) -> list[AssetUniverseItem]:
        params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": per_page, "page": page, "sparkline": "false", "price_change_percentage": "24h"}
        retries = settings.universe_retry_count
        backoff = settings.universe_backoff_base_seconds
        for attempt in range(retries):
            await self.limiter.acquire()
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
                    resp = await client.get("/coins/markets", params=params)
                if resp.status_code == 429 and attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                out: list[AssetUniverseItem] = []
                for row in data:
                    symbol = str(row.get("symbol") or "").upper()
                    name = str(row.get("name") or "")
                    out.append(
                        AssetUniverseItem(
                            symbol=symbol,
                            name=name,
                            provider="coingecko",
                            coingecko_id=row.get("id"),
                            market_cap_usd=row.get("market_cap"),
                            volume_24h_usd=row.get("total_volume"),
                            listing_date=None,
                            stable_like=looks_like_stable(symbol, name),
                            tokenized_stock_like=looks_like_tokenized_stock(symbol, name),
                        )
                    )
                return out
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(backoff * (2 ** attempt))
        return []

    async def fetch_history_by_coingecko_id(self, *, coingecko_id: str, days: int, vs_currency: str) -> MarketDataFetchResult:
        params = {"vs_currency": vs_currency, "days": min(days, DEMO_HISTORY_MAX_DAYS), "interval": "daily"}
        retries = settings.history_retry_count
        backoff = settings.history_backoff_base_seconds
        for attempt in range(retries):
            await self.limiter.acquire()
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
                    resp = await client.get(f"/coins/{coingecko_id}/market_chart", params=params)
                if resp.status_code == 429 and attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                return MarketDataFetchResult(ok=True, endpoint="/coins/{id}/market_chart", http_status=200, request_params=params, chart=resp.json())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                return MarketDataFetchResult(
                    ok=False,
                    endpoint="/coins/{id}/market_chart",
                    reason=f"http_{exc.response.status_code}",
                    error_message=exc.response.text[:500],
                    http_status=exc.response.status_code,
                    request_params=params,
                )
            except httpx.RequestError as exc:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                return MarketDataFetchResult(ok=False, endpoint="/coins/{id}/market_chart", reason="network_error", error_message=str(exc), request_params=params)


class CoinMarketCapClient:
    def __init__(self) -> None:
        self.base_url = settings.cmc_base_url
        self.timeout = settings.request_timeout_seconds
        self.limiter = AsyncRateLimiter(settings.cmc_soft_rpm_limit)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if settings.cmc_api_key:
            headers["X-CMC_PRO_API_KEY"] = settings.cmc_api_key
        return headers

    async def fetch_market_universe_page(self, *, start: int, limit: int, convert: str) -> list[AssetUniverseItem]:
        params = {"start": start, "limit": limit, "convert": convert.upper(), "sort": "market_cap", "sort_dir": "desc", "cryptocurrency_type": "all"}
        retries = settings.universe_retry_count
        backoff = settings.universe_backoff_base_seconds
        for attempt in range(retries):
            await self.limiter.acquire()
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
                    resp = await client.get("/v1/cryptocurrency/listings/latest", params=params)
                if resp.status_code == 429 and attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data", [])
                out: list[AssetUniverseItem] = []
                for row in data:
                    quote = ((row.get("quote") or {}).get(convert.upper()) or {})
                    symbol = str(row.get("symbol") or "").upper()
                    name = str(row.get("name") or "")
                    out.append(
                        AssetUniverseItem(
                            symbol=symbol,
                            name=name,
                            provider="coinmarketcap",
                            coinmarketcap_id=row.get("id"),
                            coingecko_id=None,
                            market_cap_usd=quote.get("market_cap"),
                            volume_24h_usd=quote.get("volume_24h"),
                            listing_date=parse_date_safe(row.get("date_added")),
                            stable_like=looks_like_stable(symbol, name) or bool(row.get("is_fiat")),
                            tokenized_stock_like=looks_like_tokenized_stock(symbol, name),
                        )
                    )
                return out
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(backoff * (2 ** attempt))
        return []

    async def fetch_history_by_coinmarketcap_id(self, *, coinmarketcap_id: int, days: int, vs_currency: str) -> MarketDataFetchResult:
        return MarketDataFetchResult(
            ok=False,
            endpoint="/v1/cryptocurrency/ohlcv/historical",
            reason="plan_unsupported",
            error_message="CoinMarketCap free plan does not provide historical OHLCV for this scanner path.",
            request_params={"id": coinmarketcap_id, "days": days, "convert": vs_currency.upper()},
            notes=["fallback_to_coingecko_recommended"],
        )
