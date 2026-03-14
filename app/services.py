 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app/services.py b/app/services.py
index c0914c0b347ade77458863e6ef0a439382622b24..cffa4946b9a27977e662a58fdd37e550c758bbd0 100644
--- a/app/services.py
+++ b/app/services.py
@@ -1,45 +1,47 @@
 from __future__ import annotations
 
 from dataclasses import asdict
 from typing import Dict, Optional
 
 import httpx
 from fastapi import HTTPException
 
 from .config import settings
 from .data_sources import (
     CoinGeckoClient,
     DEMO_HISTORY_MAX_DAYS,
     classify_behavioral_universe_filter,
     coingecko_daily_closes,
     looks_like_major,
     looks_like_stable,
     looks_like_tokenized_stock,
 )
 from .models import (
     BestWindow,
+    CompactScanResponse,
+    CompactScanResult,
     DebugSymbolInfo,
     MatchBreakdown,
     ScanRequest,
     ScanResponse,
     ScanResult,
 )
 from .patterns import score_crown_shelf_right_spike
 
 FEATURE_KEYS = [
     "crown",
     "drop",
     "shelf",
     "right_spike",
     "reversion",
     "asymmetry",
     "template_shape",
 ]
 
 EXEMPLAR_BREAKDOWNS: dict[str, dict[str, float]] = {
     "SIREN": {
         "crown": 0.42,
         "drop": 0.63,
         "shelf": 0.77,
         "right_spike": 0.71,
         "reversion": 0.58,
@@ -301,112 +303,269 @@ def classify_final_label(
 
     return "reject"
 
 
 def should_surface_pre_filter_candidate(best: dict) -> bool:
     nearest_distance = min(
         float(best["distance_to_siren_breakdown"]),
         float(best["distance_to_river_breakdown"]),
     )
     return (
         float(best["raw_similarity"]) >= 42.0
         or float(best["structural_score"]) >= 40.0
         or float(best["exemplar_consistency_score"]) >= 48.0
         or nearest_distance <= 0.22
     )
 
 
 def combine_scores(structural_score: float, exemplar_consistency_score: float) -> float:
     return round(
         settings.structural_score_weight * structural_score
         + settings.exemplar_consistency_weight * exemplar_consistency_score,
         2,
     )
 
 
-def find_best_window(closes: list[float], min_age_days: int, max_age_days: int):
+
+
+def _percentile(values: list[float], q: float) -> float:
+    if not values:
+        return 0.0
+    xs = sorted(values)
+    if len(xs) == 1:
+        return xs[0]
+    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
+    lo = int(pos)
+    hi = min(len(xs) - 1, lo + 1)
+    t = pos - lo
+    return xs[lo] * (1 - t) + xs[hi] * t
+
+
+def _safe_mean(values: list[float]) -> float:
+    return sum(values) / len(values) if values else 0.0
+
+
+def _safe_std(values: list[float]) -> float:
+    if len(values) < 2:
+        return 0.0
+    m = _safe_mean(values)
+    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
+
+
+def compute_pre_breakout_window_features(window_closes: list[float]) -> dict[str, float | str]:
+    if len(window_closes) < 20:
+        return {
+            "early_impulse_score": 0.0,
+            "return_to_base_score": 0.0,
+            "base_duration_score": 0.0,
+            "base_compaction_score": 0.0,
+            "right_side_tightening_score": 0.0,
+            "breakout_not_started_score": 0.0,
+            "late_breakout_penalty": 1.0,
+            "post_breakout_extension_penalty": 1.0,
+            "selected_window_stage": "post_breakout",
+            "setup_score": 0.0,
+            "penalty_score": 100.0,
+            "prebreakout_structural": 0.0,
+        }
+
+    n = len(window_closes)
+    left_end = max(4, int(n * 0.22))
+    mid_start = max(left_end, int(n * 0.25))
+    mid_end = max(mid_start + 4, int(n * 0.80))
+    right_start = max(mid_start + 2, int(n * 0.75))
+
+    left = window_closes[:left_end]
+    middle = window_closes[mid_start:mid_end]
+    right = window_closes[right_start:]
+
+    if not middle:
+        middle = window_closes[left_end:right_start] or window_closes
+
+    base_q25 = _percentile(middle, 0.25)
+    base_q75 = _percentile(middle, 0.75)
+    base_iqr = max(1e-9, base_q75 - base_q25)
+    base_mid = _safe_mean(middle)
+    total_range = max(1e-9, max(window_closes) - min(window_closes))
+
+    base_low = base_q25 - 0.45 * base_iqr
+    base_high = base_q75 + 0.45 * base_iqr
+
+    left_peak_idx = max(range(len(left)), key=lambda i: left[i]) if left else 0
+    left_peak = left[left_peak_idx] if left else window_closes[0]
+
+    early_impulse_raw = max(0.0, left_peak - base_mid) / max(total_range, base_iqr * 1.8)
+    early_impulse_score = max(0.0, min(1.0, early_impulse_raw / 0.42))
+
+    post_peak_segment = window_closes[left_peak_idx + 1:mid_end]
+    if post_peak_segment:
+        in_base_after_peak = sum(1 for x in post_peak_segment if base_low <= x <= base_high) / len(post_peak_segment)
+    else:
+        in_base_after_peak = 0.0
+    return_to_base_score = max(0.0, min(1.0, (in_base_after_peak - 0.25) / 0.65))
+
+    middle_in_base = sum(1 for x in middle if base_low <= x <= base_high) / max(1, len(middle))
+    base_duration_score = max(0.0, min(1.0, (middle_in_base - 0.35) / 0.55))
+
+    middle_std = _safe_std(middle)
+    compaction_ratio = middle_std / max(1e-9, total_range)
+    base_compaction_score = max(0.0, min(1.0, 1.0 - compaction_ratio / 0.22))
+
+    right_std = _safe_std(right)
+    right_slope = (right[-1] - right[0]) / max(1, len(right) - 1) if len(right) >= 2 else 0.0
+    slope_ratio = abs(right_slope) / max(1e-9, total_range)
+    volatility_ratio = right_std / max(1e-9, middle_std + 1e-9)
+    near_base_top = sum(1 for x in right if base_mid <= x <= (base_high + 0.35 * base_iqr)) / max(1, len(right))
+    right_side_tightening_score = max(0.0, min(1.0, 0.45 * (1.0 - min(1.0, slope_ratio / 0.06)) + 0.25 * (1.0 - min(1.0, (volatility_ratio - 1.0) / 1.5)) + 0.30 * near_base_top))
+
+    last20_start = max(0, int(n * 0.80))
+    last20 = window_closes[last20_start:]
+    late_excess = max(0.0, (max(last20) if last20 else window_closes[-1]) - (base_high + 0.35 * base_iqr)) / max(total_range, base_iqr)
+    late_breakout_penalty = max(0.0, min(1.0, late_excess / 0.65))
+
+    tail = window_closes[max(0, int(n * 0.90)):]
+    tail_mean = _safe_mean(tail)
+    post_ext = max(0.0, tail_mean - (base_high + 0.40 * base_iqr)) / max(total_range, base_iqr)
+    post_breakout_extension_penalty = max(0.0, min(1.0, post_ext / 0.55))
+
+    breakout_not_started_score = max(0.0, min(1.0, 1.0 - (0.60 * late_breakout_penalty + 0.40 * post_breakout_extension_penalty)))
+
+    setup_score = 100.0 * (
+        0.18 * early_impulse_score
+        + 0.16 * return_to_base_score
+        + 0.22 * base_duration_score
+        + 0.18 * base_compaction_score
+        + 0.14 * right_side_tightening_score
+        + 0.12 * breakout_not_started_score
+    )
+    penalty_score = 100.0 * (0.55 * late_breakout_penalty + 0.45 * post_breakout_extension_penalty)
+
+    if post_breakout_extension_penalty >= 0.58 or late_breakout_penalty >= 0.65:
+        selected_window_stage = "post_breakout"
+    elif breakout_not_started_score < 0.45:
+        selected_window_stage = "breakout"
+    else:
+        selected_window_stage = "pre_breakout"
+
+    prebreakout_structural = max(0.0, min(100.0, 0.45 * setup_score + 0.55 * (100.0 * breakout_not_started_score) - 0.85 * penalty_score))
+
+    return {
+        "early_impulse_score": round(early_impulse_score, 4),
+        "return_to_base_score": round(return_to_base_score, 4),
+        "base_duration_score": round(base_duration_score, 4),
+        "base_compaction_score": round(base_compaction_score, 4),
+        "right_side_tightening_score": round(right_side_tightening_score, 4),
+        "breakout_not_started_score": round(breakout_not_started_score, 4),
+        "late_breakout_penalty": round(late_breakout_penalty, 4),
+        "post_breakout_extension_penalty": round(post_breakout_extension_penalty, 4),
+        "selected_window_stage": selected_window_stage,
+        "setup_score": round(setup_score, 2),
+        "penalty_score": round(penalty_score, 2),
+        "prebreakout_structural": round(prebreakout_structural, 2),
+    }
+
+def find_best_window(closes: list[float], min_age_days: int, max_age_days: int, stage_mode: str = "legacy"):
     best_effective = -1.0
     best = None
     candidate_windows_count = 0
 
     for window_closes, window_len, start_idx, end_idx, total_len in iter_windows(
         closes, min_age_days, max_age_days
     ):
         candidate_windows_count += 1
 
         try:
             result = score_crown_shelf_right_spike(window_closes)
         except Exception as e:
             raise ScoringError(f"{type(e).__name__}: {e}") from e
 
         pos = position_bonus(start_idx, end_idx, total_len)
         structural_score = float(result.similarity)
         effective_structural = structural_score * (0.70 + 0.30 * pos)
 
         if (total_len - end_idx) <= 1:
             effective_structural *= 0.40
 
         exemplar_metrics = compute_exemplar_metrics(result.breakdown)
         breakdown_dict = breakdown_to_dict(result.breakdown)
         pre_breakout = compute_pre_breakout_guardrails(
             breakdown_dict=breakdown_dict,
             stage=result.stage,
             best_window_end=end_idx,
             total_len=total_len,
         )
 
+        pre_breakout_window = compute_pre_breakout_window_features(window_closes)
+        structural_for_blend = effective_structural
+        if stage_mode == "pre_breakout_only":
+            structural_for_blend = 0.55 * effective_structural + 0.45 * float(pre_breakout_window["prebreakout_structural"])
+
         final_score = combine_scores(
-            structural_score=effective_structural,
+            structural_score=structural_for_blend,
             exemplar_consistency_score=float(exemplar_metrics["exemplar_consistency_score"]),
         )
         final_score = round(
             final_score * (0.82 + 0.18 * (float(pre_breakout["pre_breakout_base_score"]) / 100.0)),
             2,
         )
+        if stage_mode == "pre_breakout_only":
+            final_score = round(
+                max(0.0, final_score - 12.0 * float(pre_breakout_window["late_breakout_penalty"]) - 10.0 * float(pre_breakout_window["post_breakout_extension_penalty"])),
+                2,
+            )
 
         if final_score > best_effective:
             best_effective = final_score
             best = {
                 "similarity": round(final_score, 2),
                 "raw_similarity": round(structural_score, 2),
                 "structural_score": round(effective_structural, 2),
                 "base_label": result.label,
                 "stage": result.stage,
                 "breakdown": result.breakdown,
                 "notes": result.notes,
                 "best_window_len": window_len,
                 "best_window_start": start_idx,
                 "best_window_end": end_idx,
                 "best_age_days": window_len,
                 "exemplar_consistency_score": float(exemplar_metrics["exemplar_consistency_score"]),
                 "distance_to_siren_breakdown": float(exemplar_metrics["distance_to_siren_breakdown"]),
                 "distance_to_river_breakdown": float(exemplar_metrics["distance_to_river_breakdown"]),
                 "reference_band_passed": bool(exemplar_metrics["reference_band_passed"]),
                 "left_structure_ok": bool(pre_breakout["left_structure_ok"]),
                 "pre_breakout_tail_ok": bool(pre_breakout["pre_breakout_tail_ok"]),
                 "stage_ok": bool(pre_breakout["stage_ok"]),
                 "pre_breakout_base_score": float(pre_breakout["pre_breakout_base_score"]),
+                "early_impulse_score": float(pre_breakout_window["early_impulse_score"]),
+                "return_to_base_score": float(pre_breakout_window["return_to_base_score"]),
+                "base_duration_score": float(pre_breakout_window["base_duration_score"]),
+                "base_compaction_score": float(pre_breakout_window["base_compaction_score"]),
+                "right_side_tightening_score": float(pre_breakout_window["right_side_tightening_score"]),
+                "breakout_not_started_score": float(pre_breakout_window["breakout_not_started_score"]),
+                "late_breakout_penalty": float(pre_breakout_window["late_breakout_penalty"]),
+                "post_breakout_extension_penalty": float(pre_breakout_window["post_breakout_extension_penalty"]),
+                "selected_window_stage": str(pre_breakout_window["selected_window_stage"]),
             }
 
     if best is None:
         return None
 
     best["candidate_windows_count"] = candidate_windows_count
     return best
 
 
 def build_symbol_index(markets: list[dict]) -> Dict[str, list[dict]]:
     index: Dict[str, list[dict]] = {}
     for coin in markets:
         symbol = str(coin.get("symbol", "")).lower()
         if not symbol:
             continue
         index.setdefault(symbol, []).append(coin)
     return index
 
 
 def build_id_index(markets: list[dict]) -> Dict[str, dict]:
     return {str(c.get("id", "")).lower(): c for c in markets if str(c.get("id", "")).strip()}
 
 
 def make_asset_key_from_symbol(symbol: str) -> str:
     return f"symbol:{symbol.upper()}"
@@ -730,58 +889,71 @@ async def build_automatic_market_universe(
         asset_sources[coin_id.lower()] = {
             "source_type": "market_universe",
             "input_symbol": symbol,
             "input_coingecko_id": coin_id,
         }
 
     all_candidates = dedupe_candidates_by_id(all_candidates)
     universe_total_count = len(all_candidates)
 
     batch_size = req.market_batch_size or req.max_coins_to_evaluate or 50
     market_offset = max(0, int(req.market_offset))
     sliced_candidates = all_candidates[market_offset: market_offset + batch_size]
     market_batch_ids = [str(c.get("id", "")).lower() for c in sliced_candidates if str(c.get("id", "")).strip()]
 
     # Keep debug compact: only selected batch + explicit exclusions.
     selected_keys = {make_asset_key_from_id(cid) for cid in market_batch_ids}
     local_debug = {
         key: value
         for key, value in local_debug.items()
         if key in selected_keys or value.universe_filter_status != "included_for_scoring"
     }
 
     return sliced_candidates, local_debug, local_skip_reasons, asset_sources, universe_total_count, len(sliced_candidates), market_batch_ids
 
 
-async def scan_pattern(req: ScanRequest) -> ScanResponse:
+def to_compact_scan_result(result: ScanResult) -> CompactScanResult:
+    return CompactScanResult(
+        coingecko_id=result.coingecko_id,
+        symbol=result.symbol,
+        name=result.name,
+        similarity=result.similarity,
+        label=result.label,
+    )
+
+
+async def scan_pattern(req: ScanRequest) -> ScanResponse | CompactScanResponse:
     if req.min_age_days > req.max_age_days:
         raise HTTPException(status_code=400, detail="min_age_days must be <= max_age_days")
 
     client = CoinGeckoClient()
     universe_total_count = 0
     market_batch_size = req.market_batch_size or req.max_coins_to_evaluate
     market_batch_ids: list[str] = []
+    compact_response = req.compact_response
+    include_notes = req.include_notes and not compact_response
+    return_pre_filter_candidates = req.return_pre_filter_candidates and not compact_response
 
     try:
         if req.symbols or req.coingecko_ids:
             pages = max(
                 1,
                 min(
                     settings.market_universe_pages,
                     max((len(req.symbols or []) + len(req.coingecko_ids or []) + 249) // 250, 1),
                 ),
             )
         else:
             # For automatic scans, fetch a broad universe but return only a compact batch.
             pages = settings.market_universe_pages
 
         try:
             markets = await client.get_markets(
                 vs_currency=req.vs_currency,
                 pages=pages,
                 per_page=settings.market_universe_per_page,
             )
         except httpx.HTTPStatusError as e:
             raise HTTPException(status_code=503, detail=f"CoinGecko markets error: {e.response.status_code}")
         except httpx.HTTPError as e:
             raise HTTPException(status_code=503, detail=f"CoinGecko connection error: {str(e)}")
 
@@ -1103,51 +1275,51 @@ async def scan_pattern(req: ScanRequest) -> ScanResponse:
                 )
                 continue
 
             debug_by_symbol[asset_key] = DebugSymbolInfo(
                 input_symbol=source_meta.get("input_symbol"),
                 input_coingecko_id=source_meta.get("input_coingecko_id"),
                 source_type=source_meta.get("source_type"),
                 resolved=True,
                 coingecko_id=coin_id,
                 status="building_windows",
                 stage="build_windows",
                 reason=None,
                 endpoint=fetch.endpoint,
                 http_status=200,
                 request_params=fetch.request_params,
                 error_message=None,
                 auth_mode=fetch.auth_mode,
                 base_url=fetch.base_url,
                 api_key_present=fetch.api_key_present,
                 auth_header_name=fetch.auth_header_name,
                 universe_filter_status="included_for_scoring",
                 universe_filter_reason="included_for_scoring",
             )
 
             try:
-                best = find_best_window(closes, req.min_age_days, min(req.max_age_days, len(closes)))
+                best = find_best_window(closes, req.min_age_days, min(req.max_age_days, len(closes)), stage_mode=req.stage_mode)
             except ScoringError as e:
                 mark_skipped(
                     asset_key=asset_key,
                     coingecko_id=coin_id,
                     reason="scoring_error",
                     stage="score_windows",
                     skipped_assets=skipped_assets,
                     skip_reasons=skip_reasons,
                     debug_by_symbol=debug_by_symbol,
                     error_message=str(e),
                     auth_mode=fetch.auth_mode,
                     base_url=fetch.base_url,
                     api_key_present=fetch.api_key_present,
                     auth_header_name=fetch.auth_header_name,
                     universe_filter_status="included_for_scoring",
                     universe_filter_reason="included_for_scoring",
                 )
                 continue
             except Exception as e:
                 mark_skipped(
                     asset_key=asset_key,
                     coingecko_id=coin_id,
                     reason="window_generation_failed",
                     stage="build_windows",
                     skipped_assets=skipped_assets,
@@ -1159,214 +1331,268 @@ async def scan_pattern(req: ScanRequest) -> ScanResponse:
                     api_key_present=fetch.api_key_present,
                     auth_header_name=fetch.auth_header_name,
                     universe_filter_status="included_for_scoring",
                     universe_filter_reason="included_for_scoring",
                 )
                 continue
 
             if best is None:
                 mark_skipped(
                     asset_key=asset_key,
                     coingecko_id=coin_id,
                     reason="insufficient_history",
                     stage="build_windows",
                     skipped_assets=skipped_assets,
                     skip_reasons=skip_reasons,
                     debug_by_symbol=debug_by_symbol,
                     auth_mode=fetch.auth_mode,
                     base_url=fetch.base_url,
                     api_key_present=fetch.api_key_present,
                     auth_header_name=fetch.auth_header_name,
                     universe_filter_status="included_for_scoring",
                     universe_filter_reason="included_for_scoring",
                 )
                 continue
 
+            best.setdefault("early_impulse_score", 0.0)
+            best.setdefault("return_to_base_score", 0.0)
+            best.setdefault("base_duration_score", 0.0)
+            best.setdefault("base_compaction_score", 0.0)
+            best.setdefault("right_side_tightening_score", 0.0)
+            best.setdefault("breakout_not_started_score", 0.0)
+            best.setdefault("late_breakout_penalty", 0.0)
+            best.setdefault("post_breakout_extension_penalty", 0.0)
+            best.setdefault("selected_window_stage", str(best.get("stage") or "active"))
+
             final_label = classify_final_label(
                 base_label=str(best["base_label"]),
                 structural_score=float(best["structural_score"]),
                 exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                 reference_band_passed=bool(best["reference_band_passed"]),
                 universe_filter_status="included_for_scoring",
                 left_structure_ok=bool(best["left_structure_ok"]),
                 pre_breakout_tail_ok=bool(best["pre_breakout_tail_ok"]),
                 stage_ok=bool(best["stage_ok"]),
                 pre_breakout_base_score=float(best["pre_breakout_base_score"]),
                 distance_to_siren_breakdown=float(best["distance_to_siren_breakdown"]),
                 distance_to_river_breakdown=float(best["distance_to_river_breakdown"]),
             )
 
             label_before_final_gate = final_label if final_label != "reject" else str(best["base_label"])
             notes = list(best["notes"])
             notes.append(
                 f"Structural score: {best['structural_score']} | Exemplar consistency: {best['exemplar_consistency_score']}"
             )
             notes.append(
                 f"Distances: siren={best['distance_to_siren_breakdown']}, river={best['distance_to_river_breakdown']}"
             )
             notes.append(
                 f"Pre-breakout base score: {best['pre_breakout_base_score']} | tail_ok={best['pre_breakout_tail_ok']} | left_structure_ok={best['left_structure_ok']}"
             )
+            notes.append(
+                f"Setup metrics: early={best['early_impulse_score']}, return={best['return_to_base_score']}, base_dur={best['base_duration_score']}, base_comp={best['base_compaction_score']}, right={best['right_side_tightening_score']}"
+            )
+            notes.append(
+                f"Breakout guard: not_started={best['breakout_not_started_score']}, late_penalty={best['late_breakout_penalty']}, post_ext_penalty={best['post_breakout_extension_penalty']}, window_stage={best['selected_window_stage']}"
+            )
             if not best["reference_band_passed"]:
                 notes.append("Reference band guardrail failed: candidate can still surface as watchlist/pre-filter.")
 
             result_obj = ScanResult(
                 coingecko_id=coin_id,
                 symbol=symbol or coin_id.upper(),
                 name=name or coin_id,
                 age_days=asset_age_days,
                 market_cap_usd=float(coin.get("market_cap") or 0) if coin.get("market_cap") is not None else None,
                 volume_24h_usd=float(coin.get("total_volume") or 0) if coin.get("total_volume") is not None else None,
                 similarity=float(best["similarity"]),
                 raw_similarity=float(best["raw_similarity"]),
                 label=final_label if final_label != "reject" else "watchlist candidate",
                 label_before_final_gate=label_before_final_gate,
                 stage=str(best["stage"]),
                 structural_score=float(best["structural_score"]),
                 exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                 distance_to_siren_breakdown=float(best["distance_to_siren_breakdown"]),
                 distance_to_river_breakdown=float(best["distance_to_river_breakdown"]),
                 reference_band_passed=bool(best["reference_band_passed"]),
                 pre_breakout_base_score=float(best["pre_breakout_base_score"]),
+                early_impulse_score=float(best["early_impulse_score"]),
+                return_to_base_score=float(best["return_to_base_score"]),
+                base_duration_score=float(best["base_duration_score"]),
+                base_compaction_score=float(best["base_compaction_score"]),
+                right_side_tightening_score=float(best["right_side_tightening_score"]),
+                breakout_not_started_score=float(best["breakout_not_started_score"]),
+                late_breakout_penalty=float(best["late_breakout_penalty"]),
+                post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
+                selected_window_stage=str(best["selected_window_stage"]),
                 universe_filter_status="included_for_scoring",
                 universe_filter_reason="included_for_scoring",
                 breakdown=MatchBreakdown(**asdict(best["breakdown"])),
                 best_window=BestWindow(
                     start_idx=int(best["best_window_start"]),
                     end_idx=int(best["best_window_end"]),
                     length_days=int(best["best_window_len"]),
                     best_age_days=int(best["best_age_days"]),
                     candidate_windows_count=int(best["candidate_windows_count"]),
                 ),
-                notes=notes if req.include_notes else [],
+                notes=notes if include_notes else [],
             )
 
-            if req.return_pre_filter_candidates and should_surface_pre_filter_candidate(best):
+            if return_pre_filter_candidates and should_surface_pre_filter_candidate(best):
                 pre_filter_candidates.append(result_obj)
 
             if final_label == "reject":
                 mark_skipped(
                     asset_key=asset_key,
                     coingecko_id=coin_id,
                     reason="filtered_after_scoring",
                     stage="score_windows",
                     skipped_assets=skipped_assets,
                     skip_reasons=skip_reasons,
                     debug_by_symbol=debug_by_symbol,
                     endpoint=fetch.endpoint,
                     http_status=200,
                     request_params=fetch.request_params,
                     auth_mode=fetch.auth_mode,
                     base_url=fetch.base_url,
                     api_key_present=fetch.api_key_present,
                     auth_header_name=fetch.auth_header_name,
                     universe_filter_status="included_for_scoring",
                     universe_filter_reason="included_for_scoring",
                     candidate_windows_count=int(best["candidate_windows_count"]),
                     best_window={
                         "start_idx": int(best["best_window_start"]),
                         "end_idx": int(best["best_window_end"]),
                         "length_days": int(best["best_window_len"]),
                         "best_age_days": int(best["best_age_days"]),
                     },
                     structural_score=float(best["structural_score"]),
                     exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                     distance_to_siren_breakdown=float(best["distance_to_siren_breakdown"]),
                     distance_to_river_breakdown=float(best["distance_to_river_breakdown"]),
                     reference_band_passed=bool(best["reference_band_passed"]),
                     raw_similarity=float(best["raw_similarity"]),
                     label="watchlist candidate" if should_surface_pre_filter_candidate(best) else "reject",
+                    early_impulse_score=float(best["early_impulse_score"]),
+                    return_to_base_score=float(best["return_to_base_score"]),
+                    base_duration_score=float(best["base_duration_score"]),
+                    base_compaction_score=float(best["base_compaction_score"]),
+                    right_side_tightening_score=float(best["right_side_tightening_score"]),
+                    breakout_not_started_score=float(best["breakout_not_started_score"]),
+                    late_breakout_penalty=float(best["late_breakout_penalty"]),
+                    post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
+                    selected_window_stage=str(best["selected_window_stage"]),
                 )
                 continue
 
             evaluated_assets.append(asset_key)
             if symbol:
                 evaluated_symbols.append(symbol)
 
             debug_by_symbol[asset_key] = DebugSymbolInfo(
                 input_symbol=source_meta.get("input_symbol"),
                 input_coingecko_id=source_meta.get("input_coingecko_id"),
                 source_type=source_meta.get("source_type"),
                 resolved=True,
                 coingecko_id=coin_id,
                 status="evaluated",
                 stage="score_windows",
                 reason=None,
                 endpoint=fetch.endpoint,
                 http_status=200,
                 request_params=fetch.request_params,
                 error_message=None,
                 auth_mode=fetch.auth_mode,
                 base_url=fetch.base_url,
                 api_key_present=fetch.api_key_present,
                 auth_header_name=fetch.auth_header_name,
                 universe_filter_status="included_for_scoring",
                 universe_filter_reason="included_for_scoring",
                 candidate_windows_count=int(best["candidate_windows_count"]),
                 best_window={
                     "start_idx": int(best["best_window_start"]),
                     "end_idx": int(best["best_window_end"]),
                     "length_days": int(best["best_window_len"]),
                     "best_age_days": int(best["best_age_days"]),
                 },
                 structural_score=float(best["structural_score"]),
                 exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                 distance_to_siren_breakdown=float(best["distance_to_siren_breakdown"]),
                 distance_to_river_breakdown=float(best["distance_to_river_breakdown"]),
                 reference_band_passed=bool(best["reference_band_passed"]),
                 pre_breakout_base_score=float(best["pre_breakout_base_score"]),
                 raw_similarity=float(best["raw_similarity"]),
                 label=final_label,
+                early_impulse_score=float(best["early_impulse_score"]),
+                return_to_base_score=float(best["return_to_base_score"]),
+                base_duration_score=float(best["base_duration_score"]),
+                base_compaction_score=float(best["base_compaction_score"]),
+                right_side_tightening_score=float(best["right_side_tightening_score"]),
+                breakout_not_started_score=float(best["breakout_not_started_score"]),
+                late_breakout_penalty=float(best["late_breakout_penalty"]),
+                post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
+                selected_window_stage=str(best["selected_window_stage"]),
             )
 
             results.append(result_obj)
 
         results.sort(key=lambda x: x.similarity, reverse=True)
         pre_filter_candidates.sort(key=lambda x: x.similarity, reverse=True)
         final_results = results[: req.top_k]
         final_prefilter = pre_filter_candidates[: max(req.top_k * 3, req.top_k)]
 
         invalid_or_unresolved_assets = (
             [make_asset_key_from_symbol(s) for s in unresolved_symbols]
             + [make_asset_key_from_id(cid) for cid in invalid_coingecko_ids]
         )
 
         skipped_symbols = [
             debug.input_symbol
             for key, debug in debug_by_symbol.items()
             if key in skipped_assets and debug.input_symbol
         ]
 
         validate_scan_invariants(
             invalid_or_unresolved_assets=invalid_or_unresolved_assets,
             skipped_assets=skipped_assets,
             evaluated_assets=evaluated_assets,
             skip_reasons=skip_reasons,
             debug_by_symbol=debug_by_symbol,
             evaluated_count=len(evaluated_assets),
         )
 
+        if compact_response:
+            return CompactScanResponse(
+                pattern_name=req.pattern_name,
+                evaluated_count=len(evaluated_assets),
+                returned_count=len(final_results),
+                market_offset=req.market_offset,
+                market_batch_size=market_batch_size or len(candidates),
+                market_batch_ids=market_batch_ids,
+                results=[to_compact_scan_result(result) for result in final_results],
+            )
+
         return ScanResponse(
             pattern_name=req.pattern_name,
             evaluated_count=len(evaluated_assets),
             returned_count=len(final_results),
             resolved_symbols=resolved_symbols,
             unresolved_symbols=unresolved_symbols,
             resolved_coingecko_ids=sorted(set(resolved_coingecko_ids)),
             invalid_coingecko_ids=sorted(set(invalid_coingecko_ids)),
             evaluated_symbols=evaluated_symbols,
             skipped_symbols=skipped_symbols,
             evaluated_assets=evaluated_assets,
             skipped_assets=skipped_assets,
             universe_source="coingecko_markets",
             universe_total_count=universe_total_count or len(candidates),
             universe_filtered_count=len(candidates),
             market_offset=req.market_offset,
             market_batch_size=market_batch_size or len(candidates),
             market_batch_ids=market_batch_ids,
             skip_reasons=skip_reasons,
             debug_by_symbol=debug_by_symbol,
             results=final_results,
-            pre_filter_candidates=final_prefilter if req.return_pre_filter_candidates else [],
+            pre_filter_candidates=final_prefilter if return_pre_filter_candidates else [],
         )
     finally:
         await client.close()
 
EOF
)
