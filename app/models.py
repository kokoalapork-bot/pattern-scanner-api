from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PatternName = Literal["crown_shelf_right_spike"]
ProviderName = Literal["coingecko", "coinmarketcap", "auto"]
StageMode = Literal["legacy", "pre_breakout_only"]


class ScanRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pattern_name": "crown_shelf_right_spike",
                    "universe_provider": "coinmarketcap",
                    "history_provider": "auto",
                    "min_age_days": 14,
                    "max_age_days": 394,
                    "top_k": 20,
                    "max_coins_to_evaluate": 200,
                    "market_batch_size": 20,
                    "vs_currency": "usd",
                    "max_market_cap_usd": 1000000000,
                    "min_listing_date_after": "2025-02-14",
                    "exclude_stables": True,
                    "exclude_tokenized_stocks": True,
                    "require_early_dump": True,
                    "require_recovery": True,
                    "require_crown": True,
                    "require_strong_dump_after_crown": True,
                    "require_live_base": True,
                    "compact_response": False
                }
            ]
        }
    )

    pattern_name: PatternName = "crown_shelf_right_spike"
    universe_provider: ProviderName = "coingecko"
    history_provider: ProviderName = "auto"
    min_age_days: int = Field(default=14, ge=1, le=5000)
    max_age_days: int = Field(default=394, ge=1, le=5000)
    top_k: int = Field(default=15, ge=1, le=100)
    max_coins_to_evaluate: int = Field(default=150, ge=1, le=5000)
    market_offset: int = Field(default=0, ge=0)
    market_batch_size: int = Field(default=20, ge=1, le=250)
    search_pages: Optional[int] = Field(default=None, ge=1, le=200)
    max_market_cap_usd: Optional[float] = Field(default=1_000_000_000.0, ge=0)
    min_market_cap_usd: Optional[float] = Field(default=0.0, ge=0)
    min_24h_volume_usd: Optional[float] = Field(default=0.0, ge=0)
    min_listing_date_after: Optional[str] = "2025-02-14"
    exclude_stables: bool = True
    exclude_tokenized_stocks: bool = True
    symbols: Optional[list[str]] = None
    coingecko_ids: Optional[list[str]] = None
    coinmarketcap_ids: Optional[list[int]] = None
    exclude_symbols: Optional[list[str]] = None
    require_early_dump: bool = True
    require_recovery: bool = True
    require_crown: bool = True
    require_strong_dump_after_crown: bool = True
    require_live_base: bool = True
    vs_currency: str = "usd"
    include_notes: bool = True
    compact_response: bool = False
    return_pre_filter_candidates: bool = True
    auto_throttle: bool = True
    debug: bool = False
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
    input_coinmarketcap_id: int | None = None
    provider: str | None = None
    resolved: bool = False
    coingecko_id: str | None = None
    coinmarketcap_id: int | None = None
    status: str = "unknown"
    reason: str | None = None
    endpoint: str | None = None
    http_status: int | None = None
    request_params: Dict[str, Any] | None = None
    error_message: str | None = None
    notes: list[str] = Field(default_factory=list)


class PhaseGateReport(BaseModel):
    early_dump: bool
    recovery: bool
    crown: bool
    strong_dump_after_crown: bool
    live_base: bool
    early_dump_score: float
    recovery_score: float
    crown_score: float
    strong_dump_after_crown_score: float
    live_base_score: float
    phase_score: float


class ScanResult(BaseModel):
    coingecko_id: str | None = None
    coinmarketcap_id: int | None = None
    symbol: str
    name: str
    age_days: int
    listing_date: str | None = None
    market_cap_usd: float | None = None
    volume_24h_usd: float | None = None
    similarity: float
    raw_similarity: float
    structural_score: float
    phase_score: float
    label: str
    stage: str
    universe_provider: str
    history_provider: str
    phase_gate_report: PhaseGateReport
    breakdown: MatchBreakdown
    best_window: BestWindow
    notes: list[str] = Field(default_factory=list)
    source: str = "market_data"


class CompactScanResult(BaseModel):
    coingecko_id: str | None = None
    coinmarketcap_id: int | None = None
    symbol: str
    name: str
    similarity: float
    age_days: int
    label: str


class ScanResponse(BaseModel):
    pattern_name: PatternName
    universe_provider: str
    history_provider: str
    evaluated_count: int
    returned_count: int
    market_offset: int
    market_batch_size: int
    pages_scanned: int
    total_candidates_seen: int
    results: list[ScanResult]
    pre_filter_candidates: list[ScanResult] = Field(default_factory=list)
    debug_by_symbol: Dict[str, DebugSymbolInfo] = Field(default_factory=dict)


class CompactScanResponse(BaseModel):
    pattern_name: PatternName
    universe_provider: str
    history_provider: str
    evaluated_count: int
    returned_count: int
    market_offset: int
    market_batch_size: int
    pages_scanned: int
    total_candidates_seen: int
    results: list[CompactScanResult]


class ErrorResponse(BaseModel):
    detail: str
