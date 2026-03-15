from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PatternName = Literal["crown_shelf_right_spike"]
StageMode = Literal["legacy", "pre_breakout_only"]


class ScanRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "min_age_days": 14,
                    "max_age_days": 394,
                    "top_k": 30,
                    "max_coins_to_evaluate": 300,
                    "vs_currency": "usd",
                    "include_notes": True,
                    "market_offset": 0,
                    "market_batch_size": 20,
                    "return_pre_filter_candidates": True,
                    "compact_response": False,
                },
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "min_age_days": 14,
                    "max_age_days": 394,
                    "top_k": 30,
                    "max_coins_to_evaluate": 300,
                    "vs_currency": "usd",
                    "include_notes": True,
                    "coingecko_ids": ["stakestone", "river", "siren-2"],
                    "compact_response": True,
                },
            ]
        }
    )

    pattern_name: PatternName = "crown_shelf_right_spike"
    min_age_days: int = Field(default=14, ge=1, le=5000)
    max_age_days: int = Field(default=394, ge=1, le=5000)
    top_k: int = Field(default=30, ge=1, le=100)
    max_coins_to_evaluate: int = Field(default=300, ge=1, le=500)
    vs_currency: str = Field(default="usd")
    include_notes: bool = True
    debug: bool = False
    symbols: Optional[list[str]] = Field(default=None)
    coingecko_ids: Optional[list[str]] = Field(default=None)
    exclude_symbols: Optional[list[str]] = Field(default=None)
    market_offset: int = Field(default=0, ge=0)
    market_batch_size: Optional[int] = Field(default=None, ge=1, le=500)
    return_pre_filter_candidates: bool = True
    compact_response: bool = False
    stage_mode: StageMode = "legacy"


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
    pre_breakout_base_score: float | None = None
    raw_similarity: float | None = None
    label: str | None = None
    early_impulse_score: float | None = None
    return_to_base_score: float | None = None
    base_duration_score: float | None = None
    base_compaction_score: float | None = None
    right_side_tightening_score: float | None = None
    breakout_not_started_score: float | None = None
    late_breakout_penalty: float | None = None
    post_breakout_extension_penalty: float | None = None
    selected_window_stage: str | None = None


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
    label_before_final_gate: str | None = None
    stage: str
    structural_score: float
    exemplar_consistency_score: float
    distance_to_siren_breakdown: float
    distance_to_river_breakdown: float
    reference_band_passed: bool
    pre_breakout_base_score: float | None = None
    early_impulse_score: float | None = None
    return_to_base_score: float | None = None
    base_duration_score: float | None = None
    base_compaction_score: float | None = None
    right_side_tightening_score: float | None = None
    breakout_not_started_score: float | None = None
    late_breakout_penalty: float | None = None
    post_breakout_extension_penalty: float | None = None
    selected_window_stage: str | None = None
    universe_filter_status: str
    universe_filter_reason: str
    breakdown: MatchBreakdown
    best_window: BestWindow
    notes: list[str] = Field(default_factory=list)
    source: str = "coingecko"


class CompactScanResult(BaseModel):
    coingecko_id: str
    symbol: str
    name: str
    similarity: float
    label: str


class ScanResponse(BaseModel):
    pattern_name: PatternName
    evaluated_count: int
    returned_count: int
    resolved_symbols: list[str] = Field(default_factory=list)
    unresolved_symbols: list[str] = Field(default_factory=list)
    resolved_coingecko_ids: list[str] = Field(default_factory=list)
    invalid_coingecko_ids: list[str] = Field(default_factory=list)
    evaluated_symbols: list[str] = Field(default_factory=list)
    skipped_symbols: list[str] = Field(default_factory=list)
    evaluated_assets: list[str] = Field(default_factory=list)
    skipped_assets: list[str] = Field(default_factory=list)
    universe_source: str = "coingecko_markets"
    universe_total_count: int = 0
    universe_filtered_count: int = 0
    market_offset: int = 0
    market_batch_size: int = 0
    market_batch_ids: list[str] = Field(default_factory=list)
    skip_reasons: Dict[str, str] = Field(default_factory=dict)
    debug_by_symbol: Dict[str, DebugSymbolInfo] = Field(default_factory=dict)
    results: list[ScanResult]
    pre_filter_candidates: list[ScanResult] = Field(default_factory=list)


class CompactScanResponse(BaseModel):
    pattern_name: PatternName
    evaluated_count: int
    returned_count: int
    market_offset: int = 0
    market_batch_size: int = 0
    market_batch_ids: list[str] = Field(default_factory=list)
    results: list[CompactScanResult]


class ErrorResponse(BaseModel):
    detail: str
