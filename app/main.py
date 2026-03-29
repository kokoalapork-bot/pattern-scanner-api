from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000

    coingecko_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "COINGECKO_API_KEY",
            "COINGECKO_DEMO_API_KEY",
            "X_CG_DEMO_API_KEY",
            "COINGECKO_PRO_API_KEY",
            "X_CG_PRO_API_KEY",
        ),
    )
    coingecko_api_base: str = Field(
        default="https://api.coingecko.com/api/v3",
        validation_alias=AliasChoices("COINGECKO_API_BASE", "X_CG_API_BASE"),
    )
    coingecko_api_plan: str = Field(
        default="demo",
        validation_alias=AliasChoices("COINGECKO_API_PLAN",),
    )
    request_timeout_seconds: int = 30
    user_agent: str = "pattern-scanner-api/1.3.1"

    default_top_k: int = 20
    default_min_age_days: int = 14
    default_max_age_days: int = 365
    default_max_coins_to_evaluate: int = 250

    min_market_cap_usd: float = 1_000_000
    min_24h_volume_usd: float = 100_000

    exclude_stables: bool = True
    exclude_tokenized_stocks: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
