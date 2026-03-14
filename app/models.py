 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app/models.py b/app/models.py
index c36f324acb09e2e08e2bd5e75b571ca10fd3c54c..ea449040acd90296f2303abd29fc0ce339d7c6d3 100644
--- a/app/models.py
+++ b/app/models.py
@@ -28,50 +28,51 @@ class ScanRequest(BaseModel):
                     "max_coins_to_evaluate": 10,
                     "vs_currency": "usd",
                     "include_notes": True,
                     "coingecko_ids": ["stakestone", "river", "siren-2"],
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
 
     symbols: Optional[list[str]] = Field(default=None)
     coingecko_ids: Optional[list[str]] = Field(default=None)
     exclude_symbols: Optional[list[str]] = Field(default=None)
 
     market_offset: int = Field(default=0, ge=0)
     market_batch_size: Optional[int] = Field(default=None, ge=1, le=500)
     return_pre_filter_candidates: bool = True
+    compact_response: bool = False
 
 
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
 
@@ -117,59 +118,77 @@ class ScanResult(BaseModel):
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
 
     universe_filter_status: str
     universe_filter_reason: str
 
     breakdown: MatchBreakdown
     best_window: BestWindow
 
     notes: list[str] = []
     source: str = "coingecko"
 
 
+class CompactScanResult(BaseModel):
+    coingecko_id: str
+    symbol: str
+    name: str
+    similarity: float
+    label: str
+
+
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
 
     universe_source: str = "coingecko_markets"
     universe_total_count: int = 0
     universe_filtered_count: int = 0
     market_offset: int = 0
     market_batch_size: int = 0
     market_batch_ids: list[str] = []
 
     skip_reasons: Dict[str, str] = {}
     debug_by_symbol: Dict[str, DebugSymbolInfo] = {}
 
     results: list[ScanResult]
     pre_filter_candidates: list[ScanResult] = []
 
 
+class CompactScanResponse(BaseModel):
+    pattern_name: PatternName
+    evaluated_count: int
+    returned_count: int
+    market_offset: int = 0
+    market_batch_size: int = 0
+    market_batch_ids: list[str] = []
+    results: list[CompactScanResult]
+
+
 class ErrorResponse(BaseModel):
     detail: str
 
EOF
)
