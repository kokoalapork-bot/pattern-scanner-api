from typing import Literal, Optional
from pydantic import BaseModel, Field


PatternName = Literal["crown_shelf_right_spike"]


class ScanRequest(BaseModel):
    pattern_name: PatternName = "crown_shelf_right_spike"
    min_age_days: int = Field(default=14, ge=1, le=5000)
    max_age_days: int = Field(default=450, ge=1, le=5000)
    top_k: int = Field(default=20, ge=1, le=100)
    max_coins_to_evaluate: int = Field(default=250, ge=10, le=2000)
    vs_currency: str = Field(default="usd")
    include_notes: bool = True
    symbols: Optional[list[str]] = None
    exclude_symbols: Optional[list[str]] = None


class MatchBreakdown(BaseModel):
    crown: float
    drop: float
    shelf: float
    right_spike: float
    reversion: float
    asymmetry: float
    template_shape: float


class ScanResult(BaseModel):
    coingecko_id: str
    symbol: str
    name: str
    age_days: int
    market_cap_usd: float | None = None
    volume_24h_usd: float | None = None
    similarity: float
    breakdown: MatchBreakdown
    notes: list[str] = []
    source: str = "coingecko"


class ScanResponse(BaseModel):
    pattern_name: PatternName
    evaluated_count: int
    returned_count: int
    results: list[ScanResult]


class ErrorResponse(BaseModel):
    detail: str
