from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEMO_BASE_DEFAULT = "https://api.coingecko.com/api/v3"
PRO_BASE_DEFAULT = "https://pro-api.coingecko.com/api/v3"
CMC_BASE_DEFAULT = "https://pro-api.coinmarketcap.com"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_min_age_days: int = 14
    default_max_age_days: int = 394
    default_top_k: int = 15
    default_market_batch_size: int = 20
    default_universe_provider: str = "coingecko"
    default_history_provider: str = "auto"

    max_market_cap_usd_for_pattern: float = 1_000_000_000.0
    min_listing_date_after: str = "2025-02-14"
    min_market_cap_usd: float = 0.0
    min_24h_volume_usd: float = 0.0
    exclude_stables: bool = True
    exclude_tokenized_stocks: bool = True

    market_universe_pages: int = 20
    market_universe_per_page: int = 250

    coingecko_auth_mode: str = Field(default="demo")
    coingecko_api_key: str = Field(default="")
    coingecko_base_url: str = Field(default="")
    coingecko_soft_rpm_limit: int = 24

    cmc_api_key: str = Field(default="")
    cmc_base_url: str = Field(default=CMC_BASE_DEFAULT)
    cmc_soft_rpm_limit: int = 24

    request_timeout_seconds: float = 20.0
    history_retry_count: int = 4
    history_backoff_base_seconds: float = 1.2
    universe_retry_count: int = 4
    universe_backoff_base_seconds: float = 1.2

    structural_score_weight: float = 0.80
    phase_score_weight: float = 0.20

    @field_validator("coingecko_auth_mode", mode="before")
    @classmethod
    def _normalize_auth_mode(cls, v: object) -> str:
        value = str(v or "demo").strip().lower()
        if value not in {"demo", "pro"}:
            raise ValueError("COINGECKO_AUTH_MODE must be 'demo' or 'pro'")
        return value

    @field_validator("coingecko_api_key", mode="before")
    @classmethod
    def _normalize_cg_api_key(cls, v: object) -> str:
        return str(v or "").strip()

    @field_validator("cmc_api_key", mode="before")
    @classmethod
    def _normalize_cmc_api_key(cls, v: object) -> str:
        return str(v or "").strip()

    @field_validator("coingecko_base_url", mode="before")
    @classmethod
    def _normalize_cg_base_url(cls, v: object) -> str:
        return str(v or "").strip().rstrip("/")

    @field_validator("cmc_base_url", mode="before")
    @classmethod
    def _normalize_cmc_base_url(cls, v: object) -> str:
        return str(v or "").strip().rstrip("/")

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        effective_cg_base = self.coingecko_base_url or (
            DEMO_BASE_DEFAULT if self.coingecko_auth_mode == "demo" else PRO_BASE_DEFAULT
        )
        cg = urlparse(effective_cg_base)
        if cg.scheme not in {"http", "https"} or not cg.netloc:
            raise ValueError("COINGECKO_BASE_URL must be a valid absolute URL")
        netloc = cg.netloc.lower()
        if self.coingecko_auth_mode == "demo" and "pro-api.coingecko.com" in netloc:
            raise ValueError("demo mode cannot use pro-api.coingecko.com")
        if self.coingecko_auth_mode == "pro" and "pro-api.coingecko.com" not in netloc:
            raise ValueError("pro mode must use pro-api.coingecko.com")

        cmc = urlparse(self.cmc_base_url or CMC_BASE_DEFAULT)
        if cmc.scheme not in {"http", "https"} or not cmc.netloc:
            raise ValueError("CMC_BASE_URL must be a valid absolute URL")

        if self.default_universe_provider not in {"coingecko", "coinmarketcap"}:
            raise ValueError("default_universe_provider must be coingecko or coinmarketcap")
        if self.default_history_provider not in {"auto", "coingecko", "coinmarketcap"}:
            raise ValueError("default_history_provider must be auto, coingecko, or coinmarketcap")
        if self.market_universe_per_page < 1 or self.market_universe_per_page > 250:
            raise ValueError("market_universe_per_page must be between 1 and 250")
        if self.default_market_batch_size < 1 or self.default_market_batch_size > 250:
            raise ValueError("default_market_batch_size must be between 1 and 250")
        if self.coingecko_soft_rpm_limit < 1 or self.cmc_soft_rpm_limit < 1:
            raise ValueError("soft rpm limits must be positive")

        total_weight = self.structural_score_weight + self.phase_score_weight
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("structural_score_weight + phase_score_weight must equal 1.0")

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
