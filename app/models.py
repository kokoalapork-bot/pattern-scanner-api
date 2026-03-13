# models.py

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

PatternName = Literal["crown_shelf_right_spike"]


class ScanRequest(BaseModel):
    pattern_name: PatternName = "crown_shelf_right_spike"
    min_age_days: int = Field(default=14, ge=1, le=5000)
    max_age_days: int = Field(default=450, ge=1, le=5000)
    top_k: int = Field(default=20, ge=1, le=100)
    max_coins_to_evaluate: int = Field(default=80, ge=1, le=500)
    vs_currency: str = Field(default="usd")
    include_notes: bool = True
    debug: bool = False

    symbols: Optional[list[str]] = None
    coingecko_ids: Optional[list[str]] = None
    exclude_symbols: Optional[list[str]] = None


class MatchBreakdown(BaseModel):
    crown: float
    drop: float
    shelf: float
    right_spike: float
    reversion: float
    asymmetry: float
    template_shape: float


class BestWindow(BaseModel):
    start_idx: int
    end_idx: int
    length_days: int
    candidate_windows_count: int


class DebugSymbolInfo(BaseModel):
    resolved: bool = False
    coingecko_id: str | None = None
    status: str = "unknown"
    stage: str = "unknown"
    reason: str | None = None
    endpoint: str | None = None
    http_status: int | None = None
    request_params: Dict[str, Any] | None = None
    error_message: str | None = None
    auth_mode: str | None = None
    base_url: str | None = None
    api_key_present: bool | None = None


class ScanResult(BaseModel):
    coingecko_id: str
    symbol: str
    name: str
    age_days: int
    market_cap_usd: float | None = None
    volume_24h_usd: float | None = None

    similarity: float
    raw_similarity: float
    label: str
    stage: str

    breakdown: MatchBreakdown
    best_window: BestWindow

    notes: list[str] = []
    source: str = "coingecko"


class ScanResponse(BaseModel):
    pattern_name: PatternName

    evaluated_count: int
    returned_count: int

    resolved_symbols: list[str] = []
    unresolved_symbols: list[str] = []
    evaluated_symbols: list[str] = []
    skipped_symbols: list[str] = []
    skip_reasons: Dict[str, str] = {}
    debug_by_symbol: Dict[str, DebugSymbolInfo] = {}

    results: list[ScanResult]


class ErrorResponse(BaseModel):
    detail: str
