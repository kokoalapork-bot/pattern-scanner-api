from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .config import get_settings


PUBLIC_API_BASE = "https://api.coingecko.com/api/v3"
PRO_API_BASE = "https://pro-api.coingecko.com/api/v3"


class CoinGeckoClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _api_base(self) -> str:
        base = (self.settings.coingecko_api_base or "").strip() or PUBLIC_API_BASE
        plan = (self.settings.coingecko_api_plan or "").strip().lower()
        if not base:
            return PRO_API_BASE if plan == "pro" else PUBLIC_API_BASE
        return base.rstrip("/")

    def _headers(self, pro: bool = False) -> dict[str, str]:
        headers = {
            "User-Agent": self.settings.user_agent,
            "accept": "application/json",
        }
        api_key = (self.settings.coingecko_api_key or "").strip()
        if api_key:
            # Demo/public plans use x-cg-demo-api-key; Pro uses x-cg-pro-api-key.
            if pro:
                headers["x-cg-pro-api-key"] = api_key
            else:
                headers["x-cg-demo-api-key"] = api_key
        return headers

    def _auth_params(self, pro: bool = False) -> dict[str, str]:
        api_key = (self.settings.coingecko_api_key or "").strip()
        if not api_key:
            return {}
        # CoinGecko documents both header and query auth styles.
        return {"x_cg_pro_api_key" if pro else "x_cg_demo_api_key": api_key}

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        require_history_auth: bool = False,
    ) -> Any:
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        params = dict(params or {})
        configured_base = self._api_base()
        plan = (self.settings.coingecko_api_plan or "").strip().lower()
        api_key_present = bool((self.settings.coingecko_api_key or "").strip())

        # Try the most likely auth/base combination first, then fall back.
        candidates: list[tuple[str, bool]] = []
        if "pro-api.coingecko.com" in configured_base or plan == "pro":
            candidates.append((configured_base, True))
            if configured_base != PUBLIC_API_BASE:
                candidates.append((PUBLIC_API_BASE, False))
        else:
            candidates.append((configured_base, False))
            if configured_base != PRO_API_BASE and api_key_present and plan == "pro":
                candidates.append((PRO_API_BASE, True))

        last_exc: Exception | None = None
        for base_url, pro in candidates:
            trial_params = dict(params)
            trial_params.update(self._auth_params(pro=pro))
            async with httpx.AsyncClient(timeout=timeout, headers=self._headers(pro=pro)) as client:
                try:
                    response = await client.get(f"{base_url}{path}", params=trial_params)
                    if response.status_code == 401 and not api_key_present and require_history_auth:
                        response.raise_for_status()
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    # If the first attempt failed with auth-related issues, try the next candidate.
                    if exc.response.status_code in {401, 403}:
                        continue
                    raise
                except Exception as exc:
                    last_exc = exc
                    raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("CoinGecko request failed without an exception")

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

    async def fetch_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 365) -> dict[str, Any]:
        api_key_present = bool((self.settings.coingecko_api_key or "").strip())
        plan = (self.settings.coingecko_api_plan or "").strip().lower()
        effective_days = int(days)
        # CoinGecko Demo/Public historical access is limited to the past 365 days.
        if plan != "pro" or not api_key_present:
            effective_days = min(effective_days, 365)
        return await self._get(
            f"/coins/{coin_id}/market_chart",
            {"vs_currency": vs_currency, "days": effective_days, "interval": "daily"},
            require_history_auth=True,
        )

    async def fetch_coin(self, coin_id: str) -> dict[str, Any]:
        return await self._get(
            f"/coins/{coin_id}",
            {
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
        )

    @staticmethod
    def age_days_from_iso(date_str: str | None) -> int | None:
        if not date_str:
            return None
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
