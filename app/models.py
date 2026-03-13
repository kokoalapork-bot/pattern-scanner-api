from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

PatternName = Literal["crown_shelf_right_spike"]


class ScanRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "min_age_days": 14,
                    "max_age_days": 450,
                    "top_k": 10,
                    "max_coins_to_evaluate": 10,
                    "vs_currency": "usd",
                    "include_notes": True,
                    "symbols": ["RIVER", "SIREN"],
                },
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "min_age_days": 14,
                    "max_age_days": 450,
                    "top_k": 10,
                    "max_coins_to_evaluate": 10,
                    "vs_currency": "usd",
                    "include_notes": True,
                    "coingecko_ids": ["stakestone", "river", "siren-2"],
                },
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "min_age_days": 14,
                    "max_age_days": 450,
                    "top_k": 10,
                    "max_coins_to_evaluate": 10,
                    "vs_currency": "usd",
                    "include_notes": True,
                    "symbols": ["RIVER"],
                    "coingecko_ids": ["siren-2"],
                },
            ]
        }
    )

    pattern_name: PatternName = "crown_shelf_right_spike"
    min_age_days: int = Field(default=14, ge=1, le=5000)
    max_age_days: int = Field(default=450, ge=1, le=5000)
    top_k: int = Field(default=20, ge=1, le=100)
    max_coins_to_evaluate: int = Field(default=80, ge=1, le=500)
    vs_currency: str = Field(default="usd")
    include_notes: bool = True
    debug: bool = False

    symbols: Optional[list[str]] = Field(
        default=None,
        description="Optional ticker symbols to resolve through CoinGecko markets list, e.g. ['RIVER', 'SIREN']",
    )
    coingecko_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional direct CoinGecko asset ids to scan without symbol resolution, e.g. ['stakestone', 'river', 'siren-2']",
    )
    exclude_symbols: Optional[list[str]] = Field(default=None)


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
    best_age_days: int
    candidate_windows_count: int


class DebugSymbolInfo(BaseModel):
    input_symbol: str | None = None
    input_coingecko_id: str | None = None
    source_type: str | None = None

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
    auth_header_name: str | None = None

    universe_filter_status: str | None = None
    universe_filter_reason: str | None = None

    candidate_windows_count: int | None = None
    best_window: Dict[str, Any] | None = None

    structural_score: float | None = None
    exemplar_consistency_score: float | None = None
    distance_to_siren_breakdown: float | None = None
    distance_to_river_breakdown: float | None = None
    reference_band_passed: bool | None = None

    raw_similarity: float | None = None
    label: str | None = None


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

    structural_score: float
    exemplar_consistency_score: float
    distance_to_siren_breakdown: float
    distance_to_river_breakdown: float
    reference_band_passed: bool

    universe_filter_status: str
    universe_filter_reason: str

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

    resolved_coingecko_ids: list[str] = []
    invalid_coingecko_ids: list[str] = []

    evaluated_symbols: list[str] = []
    skipped_symbols: list[str] = []

    evaluated_assets: list[str] = []
    skipped_assets: list[str] = []

    skip_reasons: Dict[str, str] = {}
    debug_by_symbol: Dict[str, DebugSymbolInfo] = {}

    results: list[ScanResult]


class ErrorResponse(BaseModel):
    detail: str
