from __future__ import annotations

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
    default_max_age_days: int = 450

    min_market_cap_usd: float = 0.0
    min_24h_volume_usd: float = 0.0

    exclude_stables: bool = True
    exclude_tokenized_stocks: bool = True

    coingecko_auth_mode: str = Field(default="demo")
    coingecko_api_key: str = Field(default="")
    coingecko_base_url: str = Field(default="")

    request_timeout_seconds: float = 20.0
    history_retry_count: int = 3
    history_backoff_base_seconds: float = 0.8

    @field_validator("coingecko_auth_mode", mode="before")
    @classmethod
    def _normalize_auth_mode(cls, v: object) -> str:
        value = str(v or "demo").strip().lower()
        if value not in {"demo", "pro"}:
            raise ValueError("coingecko_auth_mode must be 'demo' or 'pro'")
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
    def _validate_coingecko_config(self) -> "Settings":
        if not self.coingecko_api_key:
            raise ValueError("COINGECKO_API_KEY is required and cannot be empty")

        expected_base = DEMO_BASE_DEFAULT if self.coingecko_auth_mode == "demo" else PRO_BASE_DEFAULT
        actual_base = self.coingecko_base_url or expected_base

        parsed = urlparse(actual_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("COINGECKO_BASE_URL must be a valid absolute URL")

        if self.coingecko_auth_mode == "demo" and "pro-api.coingecko.com" in parsed.netloc:
            raise ValueError("demo mode cannot use pro-api.coingecko.com")

        if self.coingecko_auth_mode == "pro" and "api.coingecko.com" in parsed.netloc and "pro-api.coingecko.com" not in parsed.netloc:
            raise ValueError("pro mode cannot use public/demo api.coingecko.com base URL")

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
