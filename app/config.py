from __future__ import annotations

from datetime import date
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEMO_BASE_DEFAULT = "https://api.coingecko.com/api/v3"
PRO_BASE_DEFAULT = "https://pro-api.coingecko.com/api/v3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_min_age_days: int = 14
    default_max_age_days: int = 90

    min_market_cap_usd: float = 0.0
    min_24h_volume_usd: float = 0.0

    exclude_stables: bool = True
    exclude_tokenized_stocks: bool = True
    max_market_cap_usd_for_pattern: float = 1_000_000_000.0

    min_listing_date_after: str = "2025-10-10"

    # How deep to fetch the CoinGecko market universe for automatic scans.
    # This is intentionally independent from max_coins_to_evaluate, so smaller /scan
    # requests still search inside a broader market universe.
    market_universe_pages: int = 20
    market_universe_per_page: int = 250

    coingecko_auth_mode: str = Field(default="demo")
    coingecko_api_key: str = Field(default="")
    coingecko_base_url: str = Field(default="")

    request_timeout_seconds: float = 20.0
    history_retry_count: int = 3
    history_backoff_base_seconds: float = 0.8

    stable_price_peg_center: float = 1.0
    stable_price_peg_tolerance: float = 0.15
    stable_max_cv: float = 0.03
    stable_max_range_ratio: float = 0.12
    low_volatility_max_cv: float = 0.02
    low_volatility_max_range_ratio: float = 0.08

    structural_score_weight: float = 0.75
    exemplar_consistency_weight: float = 0.25

    @field_validator("coingecko_auth_mode", mode="before")
    @classmethod
    def _normalize_auth_mode(cls, v: object) -> str:
        value = str(v or "demo").strip().lower()
        if value not in {"demo", "pro"}:
            raise ValueError("COINGECKO_AUTH_MODE must be 'demo' or 'pro'")
        return value

    @field_validator("coingecko_api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, v: object) -> str:
        return str(v or "").strip()

    @field_validator("coingecko_base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, v: object) -> str:
        return str(v or "").strip().rstrip("/")

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        if not self.coingecko_api_key:
            raise ValueError("COINGECKO_API_KEY is required and cannot be empty or whitespace")

        effective_base = self.coingecko_base_url or (
            DEMO_BASE_DEFAULT if self.coingecko_auth_mode == "demo" else PRO_BASE_DEFAULT
        )

        parsed = urlparse(effective_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("COINGECKO_BASE_URL must be a valid absolute URL")

        netloc = parsed.netloc.lower()

        if self.coingecko_auth_mode == "demo" and "pro-api.coingecko.com" in netloc:
            raise ValueError("demo mode cannot use pro-api.coingecko.com")

        if self.coingecko_auth_mode == "pro" and "pro-api.coingecko.com" not in netloc:
            raise ValueError("pro mode must use pro-api.coingecko.com")

        total_weight = self.structural_score_weight + self.exemplar_consistency_weight
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("structural_score_weight + exemplar_consistency_weight must equal 1.0")

        if self.max_market_cap_usd_for_pattern <= 0:
            raise ValueError("max_market_cap_usd_for_pattern must be positive")

        if self.market_universe_pages < 1:
            raise ValueError("market_universe_pages must be >= 1")

        if self.market_universe_per_page < 1 or self.market_universe_per_page > 250:
            raise ValueError("market_universe_per_page must be between 1 and 250")

        return self

    @property
    def coingecko_effective_base_url(self) -> str:
        return self.coingecko_base_url or (
            DEMO_BASE_DEFAULT if self.coingecko_auth_mode == "demo" else PRO_BASE_DEFAULT
        )

    @property
    def coingecko_header_name(self) -> str:
        return "x-cg-demo-api-key" if self.coingecko_auth_mode == "demo" else "x-cg-pro-api-key"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
