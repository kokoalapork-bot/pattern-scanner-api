from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    coingecko_api_key: str | None = Field(default=None, alias="COINGECKO_API_KEY")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    default_top_k: int = Field(default=20, alias="DEFAULT_TOP_K")
    default_min_age_days: int = Field(default=14, alias="DEFAULT_MIN_AGE_DAYS")
    default_max_age_days: int = Field(default=450, alias="DEFAULT_MAX_AGE_DAYS")
    default_max_coins_to_evaluate: int = Field(default=250, alias="DEFAULT_MAX_COINS_TO_EVALUATE")

    min_market_cap_usd: float = Field(default=1_000_000, alias="MIN_MARKET_CAP_USD")
    min_24h_volume_usd: float = Field(default=100_000, alias="MIN_24H_VOLUME_USD")

    exclude_stables: bool = Field(default=True, alias="EXCLUDE_STABLES")
    exclude_tokenized_stocks: bool = Field(default=True, alias="EXCLUDE_TOKENIZED_STOCKS")


settings = Settings()
