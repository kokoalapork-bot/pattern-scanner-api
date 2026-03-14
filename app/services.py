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
    CompactScanResponse,
    CompactScanResult,
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
        "asymmetry": 0.60,
        "template_shape": 0.69,
    },
    "RIVER": {
        "crown": 0.35,
        "drop": 0.56,
        "shelf": 0.73,
        "right_spike": 0.64,
        "reversion": 0.52,
        "asymmetry": 0.57,
        "template_shape": 0.64,
    },
}

REFERENCE_TOLERANCES: dict[str, tuple[float, float]] = {
    "crown": (0.22, 0.16),
    "drop": (0.16, 0.16),
    "shelf": (0.20, 0.16),
    "right_spike": (0.18, 0.16),
    "reversion": (0.18, 0.18),
    "asymmetry": (0.18, 0.18),
    "template_shape": (0.16, 0.16),
}


class ScoringError(RuntimeError):
    pass


def age_from_chart_days(chart: dict) -> int | None:
    prices = chart.get("prices", [])
    if not prices:
        return None

    first_ts_ms = prices[0][0]
    last_ts_ms = prices[-1][0]
    if first_ts_ms is None or last_ts_ms is None:
        return None

    age_days = int((last_ts_ms - first_ts_ms) / 86_400_000)
    return max(age_days, 0)


def generate_window_lengths(min_age_days: int, max_age_days: int) -> list[int]:
    candidates = [30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 300, 360]
    out = [w for w in candidates if min_age_days <= w <= max_age_days]
    if not out:
        out = [max(30, min(max_age_days, 60))]
    return out


def iter_windows(closes: list[float], min_age_days: int, max_age_days: int):
    n = len(closes)
    if n < max(30, min_age_days):
        return

    window_lengths = generate_window_lengths(min_age_days, max_age_days)

    for w in window_lengths:
        if n < w:
            continue

        step = max(3, w // 8)
        yielded_last = False

        for end in range(w, n + 1, step):
            start = end - w
            yield closes[start:end], w, start, end, n
            if end == n:
                yielded_last = True

        if not yielded_last:
            yield closes[n - w:n], w, n - w, n, n


def position_bonus(start_idx: int, end_idx: int, total_len: int) -> float:
    if total_len <= 0:
        return 0.0

    end_ratio = end_idx / total_len
    start_ratio = start_idx / total_len
    right_tail = total_len - end_idx

    if end_ratio < 0.35:
        return 0.25

    if right_tail <= 1:
        return 0.10
    if right_tail <= 3:
        return 0.30
    if right_tail <= 6:
        return 0.55
    if right_tail <= 14:
        return 1.00

    target = 0.82
    distance = abs(end_ratio - target)
    pos_score = max(0.0, 1.0 - distance / 0.35)

    if start_ratio > 0.80:
        pos_score *= 0.55
    elif start_ratio > 0.70:
        pos_score *= 0.78

    return max(0.0, min(1.0, pos_score))


def breakdown_to_dict(breakdown) -> dict[str, float]:
    return {
        "crown": float(breakdown.crown),
        "drop": float(breakdown.drop),
        "shelf": float(breakdown.shelf),
        "right_spike": float(breakdown.right_spike),
        "reversion": float(breakdown.reversion),
        "asymmetry": float(breakdown.asymmetry),
        "template_shape": float(breakdown.template_shape),
    }


def mean_abs_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a[k] - b[k]) for k in FEATURE_KEYS) / len(FEATURE_KEYS)


def compute_reference_band_passed(candidate: dict[str, float]) -> bool:
    for key in FEATURE_KEYS:
        v1 = EXEMPLAR_BREAKDOWNS["SIREN"][key]
        v2 = EXEMPLAR_BREAKDOWNS["RIVER"][key]
        lo = min(v1, v2)
        hi = max(v1, v2)
        lower_tol, upper_tol = REFERENCE_TOLERANCES[key]

        if candidate[key] < (lo - lower_tol):
            return False
        if candidate[key] > (hi + upper_tol):
            return False

    return True


def compute_exemplar_metrics(breakdown) -> dict[str, float | bool]:
    candidate = breakdown_to_dict(breakdown)

    distance_to_siren = mean_abs_distance(candidate, EXEMPLAR_BREAKDOWNS["SIREN"])
    distance_to_river = mean_abs_distance(candidate, EXEMPLAR_BREAKDOWNS["RIVER"])
    nearest_distance = min(distance_to_siren, distance_to_river)

    exemplar_consistency = max(0.0, 100.0 * (1.0 - min(1.0, nearest_distance / 0.45)))
    reference_band_passed = compute_reference_band_passed(candidate)

    return {
        "exemplar_consistency_score": round(exemplar_consistency, 2),
        "distance_to_siren_breakdown": round(distance_to_siren, 4),
        "distance_to_river_breakdown": round(distance_to_river, 4),
        "reference_band_passed": reference_band_passed,
    }


def compute_pre_breakout_guardrails(
    *,
    breakdown_dict: dict[str, float],
    stage: str,
    best_window_end: int,
    total_len: int,
) -> dict[str, float | bool]:
    shelf = breakdown_dict["shelf"]
    crown = breakdown_dict["crown"]
    spike = breakdown_dict["right_spike"]
    reversion = breakdown_dict["reversion"]
    template = breakdown_dict["template_shape"]

    right_tail = max(0, total_len - best_window_end)
    pre_breakout_tail_ok = 3 <= right_tail <= 20
    left_structure_ok = (
        shelf >= 0.55
        and template >= 0.50
        and crown >= 0.18
        and spike >= 0.35
        and reversion >= 0.30
    )
    stage_ok = stage in {"forming", "active"}

    pre_breakout_base_score = 0.0
    pre_breakout_base_score += min(1.0, max(0.0, (shelf - 0.45) / 0.35)) * 0.30
    pre_breakout_base_score += min(1.0, max(0.0, (template - 0.40) / 0.35)) * 0.20
    pre_breakout_base_score += min(1.0, max(0.0, (crown - 0.10) / 0.35)) * 0.15
    pre_breakout_base_score += min(1.0, max(0.0, (spike - 0.25) / 0.40)) * 0.15
    pre_breakout_base_score += min(1.0, max(0.0, (reversion - 0.20) / 0.35)) * 0.10
    pre_breakout_base_score += (1.0 if pre_breakout_tail_ok else 0.0) * 0.10
    pre_breakout_base_score = round(pre_breakout_base_score * 100.0, 2)

    return {
        "left_structure_ok": left_structure_ok,
        "pre_breakout_tail_ok": pre_breakout_tail_ok,
        "stage_ok": stage_ok,
        "pre_breakout_base_score": pre_breakout_base_score,
    }


def classify_final_label(
    *,
    base_label: str,
    structural_score: float,
    exemplar_consistency_score: float,
    reference_band_passed: bool,
    universe_filter_status: str,
    left_structure_ok: bool = True,
    pre_breakout_tail_ok: bool = True,
    stage_ok: bool = True,
    pre_breakout_base_score: float = 50.0,
    distance_to_siren_breakdown: float | None = None,
    distance_to_river_breakdown: float | None = None,
) -> str:
    if universe_filter_status != "included_for_scoring":
        return "reject"

    if structural_score < 28.0:
        return "reject"

    if not left_structure_ok:
        return "reject"

    nearest_distance = min(
        distance_to_siren_breakdown if distance_to_siren_breakdown is not None else 9.0,
        distance_to_river_breakdown if distance_to_river_breakdown is not None else 9.0,
    )

    if (
        structural_score >= 60.0
        and exemplar_consistency_score >= 56.0
        and pre_breakout_base_score >= 55.0
        and nearest_distance <= 0.205
        and (reference_band_passed or pre_breakout_tail_ok or stage_ok)
    ):
        return "strong match"

    if (
        structural_score >= 45.0
        and exemplar_consistency_score >= 42.0
        and pre_breakout_base_score >= 42.0
        and nearest_distance <= 0.285
    ):
        if base_label == "weak-crown variant":
            return "weak-crown variant"
        return "partial match"

    if (
        structural_score >= 36.0
        and exemplar_consistency_score >= 38.0
        and pre_breakout_base_score >= 34.0
        and nearest_distance <= 0.33
    ):
        return "watchlist candidate"

    if structural_score >= 55.0:
        return "weak match"

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




def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    t = pos - lo
    return xs[lo] * (1 - t) + xs[hi] * t


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def compute_pre_breakout_window_features(window_closes: list[float]) -> dict[str, float | str]:
    if len(window_closes) < 20:
        return {
            "early_impulse_score": 0.0,
            "return_to_base_score": 0.0,
            "base_duration_score": 0.0,
            "base_compaction_score": 0.0,
            "right_side_tightening_score": 0.0,
            "breakout_not_started_score": 0.0,
            "late_breakout_penalty": 1.0,
            "post_breakout_extension_penalty": 1.0,
            "selected_window_stage": "post_breakout",
            "setup_score": 0.0,
            "penalty_score": 100.0,
            "prebreakout_structural": 0.0,
        }

    n = len(window_closes)
    left_end = max(4, int(n * 0.22))
    mid_start = max(left_end, int(n * 0.25))
    mid_end = max(mid_start + 4, int(n * 0.80))
    right_start = max(mid_start + 2, int(n * 0.75))

    left = window_closes[:left_end]
    middle = window_closes[mid_start:mid_end]
    right = window_closes[right_start:]

    if not middle:
        middle = window_closes[left_end:right_start] or window_closes

    base_q25 = _percentile(middle, 0.25)
    base_q75 = _percentile(middle, 0.75)
    base_iqr = max(1e-9, base_q75 - base_q25)
    base_mid = _safe_mean(middle)
    total_range = max(1e-9, max(window_closes) - min(window_closes))

    base_low = base_q25 - 0.45 * base_iqr
    base_high = base_q75 + 0.45 * base_iqr

    left_peak_idx = max(range(len(left)), key=lambda i: left[i]) if left else 0
    left_peak = left[left_peak_idx] if left else window_closes[0]

    early_impulse_raw = max(0.0, left_peak - base_mid) / max(total_range, base_iqr * 1.8)
    early_impulse_score = max(0.0, min(1.0, early_impulse_raw / 0.42))

    post_peak_segment = window_closes[left_peak_idx + 1:mid_end]
    if post_peak_segment:
        in_base_after_peak = sum(1 for x in post_peak_segment if base_low <= x <= base_high) / len(post_peak_segment)
    else:
        in_base_after_peak = 0.0
    return_to_base_score = max(0.0, min(1.0, (in_base_after_peak - 0.25) / 0.65))

    middle_in_base = sum(1 for x in middle if base_low <= x <= base_high) / max(1, len(middle))
    base_duration_score = max(0.0, min(1.0, (middle_in_base - 0.35) / 0.55))

    middle_std = _safe_std(middle)
    compaction_ratio = middle_std / max(1e-9, total_range)
    base_compaction_score = max(0.0, min(1.0, 1.0 - compaction_ratio / 0.22))

    right_std = _safe_std(right)
    right_slope = (right[-1] - right[0]) / max(1, len(right) - 1) if len(right) >= 2 else 0.0
    slope_ratio = abs(right_slope) / max(1e-9, total_range)
    volatility_ratio = right_std / max(1e-9, middle_std + 1e-9)
    near_base_top = sum(1 for x in right if base_mid <= x <= (base_high + 0.35 * base_iqr)) / max(1, len(right))
    right_side_tightening_score = max(0.0, min(1.0, 0.45 * (1.0 - min(1.0, slope_ratio / 0.06)) + 0.25 * (1.0 - min(1.0, (volatility_ratio - 1.0) / 1.5)) + 0.30 * near_base_top))

    last20_start = max(0, int(n * 0.80))
    last20 = window_closes[last20_start:]
    late_excess = max(0.0, (max(last20) if last20 else window_closes[-1]) - (base_high + 0.35 * base_iqr)) / max(total_range, base_iqr)
    late_breakout_penalty = max(0.0, min(1.0, late_excess / 0.65))

    tail = window_closes[max(0, int(n * 0.90)):]
    tail_mean = _safe_mean(tail)
    post_ext = max(0.0, tail_mean - (base_high + 0.40 * base_iqr)) / max(total_range, base_iqr)
    post_breakout_extension_penalty = max(0.0, min(1.0, post_ext / 0.55))

    breakout_not_started_score = max(0.0, min(1.0, 1.0 - (0.60 * late_breakout_penalty + 0.40 * post_breakout_extension_penalty)))

    setup_score = 100.0 * (
        0.18 * early_impulse_score
        + 0.16 * return_to_base_score
        + 0.22 * base_duration_score
        + 0.18 * base_compaction_score
        + 0.14 * right_side_tightening_score
        + 0.12 * breakout_not_started_score
    )
    penalty_score = 100.0 * (0.55 * late_breakout_penalty + 0.45 * post_breakout_extension_penalty)

    if post_breakout_extension_penalty >= 0.58 or late_breakout_penalty >= 0.65:
        selected_window_stage = "post_breakout"
    elif breakout_not_started_score < 0.45:
        selected_window_stage = "breakout"
    else:
        selected_window_stage = "pre_breakout"

    prebreakout_structural = max(0.0, min(100.0, 0.45 * setup_score + 0.55 * (100.0 * breakout_not_started_score) - 0.85 * penalty_score))

    return {
        "early_impulse_score": round(early_impulse_score, 4),
        "return_to_base_score": round(return_to_base_score, 4),
        "base_duration_score": round(base_duration_score, 4),
        "base_compaction_score": round(base_compaction_score, 4),
        "right_side_tightening_score": round(right_side_tightening_score, 4),
        "breakout_not_started_score": round(breakout_not_started_score, 4),
        "late_breakout_penalty": round(late_breakout_penalty, 4),
        "post_breakout_extension_penalty": round(post_breakout_extension_penalty, 4),
        "selected_window_stage": selected_window_stage,
        "setup_score": round(setup_score, 2),
        "penalty_score": round(penalty_score, 2),
        "prebreakout_structural": round(prebreakout_structural, 2),
    }

def find_best_window(closes: list[float], min_age_days: int, max_age_days: int, stage_mode: str = "legacy"):
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

        pre_breakout_window = compute_pre_breakout_window_features(window_closes)
        structural_for_blend = effective_structural
        if stage_mode == "pre_breakout_only":
            structural_for_blend = 0.55 * effective_structural + 0.45 * float(pre_breakout_window["prebreakout_structural"])

        final_score = combine_scores(
            structural_score=structural_for_blend,
            exemplar_consistency_score=float(exemplar_metrics["exemplar_consistency_score"]),
        )
        final_score = round(
            final_score * (0.82 + 0.18 * (float(pre_breakout["pre_breakout_base_score"]) / 100.0)),
            2,
        )
        if stage_mode == "pre_breakout_only":
            final_score = round(
                max(0.0, final_score - 12.0 * float(pre_breakout_window["late_breakout_penalty"]) - 10.0 * float(pre_breakout_window["post_breakout_extension_penalty"])),
                2,
            )

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
                "early_impulse_score": float(pre_breakout_window["early_impulse_score"]),
                "return_to_base_score": float(pre_breakout_window["return_to_base_score"]),
                "base_duration_score": float(pre_breakout_window["base_duration_score"]),
                "base_compaction_score": float(pre_breakout_window["base_compaction_score"]),
                "right_side_tightening_score": float(pre_breakout_window["right_side_tightening_score"]),
                "breakout_not_started_score": float(pre_breakout_window["breakout_not_started_score"]),
                "late_breakout_penalty": float(pre_breakout_window["late_breakout_penalty"]),
                "post_breakout_extension_penalty": float(pre_breakout_window["post_breakout_extension_penalty"]),
                "selected_window_stage": str(pre_breakout_window["selected_window_stage"]),
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


def make_asset_key_from_id(coingecko_id: str) -> str:
    return f"id:{coingecko_id.lower()}"


def dedupe_candidates_by_id(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []

    for coin in candidates:
        coin_id = str(coin.get("id", "")).lower()
        if not coin_id or coin_id in seen:
            continue
        seen.add(coin_id)
        out.append(coin)

    return out


def resolve_requested_assets(
    markets: list[dict],
    symbols: Optional[list[str]],
    coingecko_ids: Optional[list[str]],
):
    resolved_candidates: list[dict] = []

    resolved_symbols: list[str] = []
    unresolved_symbols: list[str] = []

    resolved_coingecko_ids: list[str] = []
    pending_coingecko_ids: list[str] = []

    skip_reasons: dict[str, str] = {}
    debug_by_symbol: dict[str, DebugSymbolInfo] = {}

    by_symbol = build_symbol_index(markets)
    by_id = build_id_index(markets)

    for sym in symbols or []:
        key = make_asset_key_from_symbol(sym)
        matches = by_symbol.get(sym.lower(), [])

        if matches:
            chosen = sorted(matches, key=lambda x: float(x.get("market_cap") or 0), reverse=True)[0]
            resolved_candidates.append(chosen)
            resolved_symbols.append(sym.upper())

            debug_by_symbol[key] = DebugSymbolInfo(
                input_symbol=sym.upper(),
                input_coingecko_id=None,
                source_type="symbol",
                resolved=True,
                coingecko_id=str(chosen.get("id")),
                status="resolved",
                stage="resolve_symbol",
                reason=None,
            )
        else:
            unresolved_symbols.append(sym.upper())
            skip_reasons[key] = "unresolved_symbol"
            debug_by_symbol[key] = DebugSymbolInfo(
                input_symbol=sym.upper(),
                input_coingecko_id=None,
                source_type="symbol",
                resolved=False,
                coingecko_id=None,
                status="unresolved",
                stage="resolve_symbol",
                reason="unresolved_symbol",
            )

    for cid in coingecko_ids or []:
        cid_norm = cid.lower()
        key = make_asset_key_from_id(cid_norm)

        coin = by_id.get(cid_norm)
        if coin:
            resolved_candidates.append(coin)
            resolved_coingecko_ids.append(cid_norm)

            debug_by_symbol[key] = DebugSymbolInfo(
                input_symbol=None,
                input_coingecko_id=cid_norm,
                source_type="coingecko_id",
                resolved=True,
                coingecko_id=cid_norm,
                status="resolved",
                stage="resolve_coingecko_id",
                reason=None,
            )
        else:
            pending_coingecko_ids.append(cid_norm)
            debug_by_symbol[key] = DebugSymbolInfo(
                input_symbol=None,
                input_coingecko_id=cid_norm,
                source_type="coingecko_id",
                resolved=True,
                coingecko_id=cid_norm,
                status="pending_lookup",
                stage="resolve_coingecko_id",
                reason=None,
            )

    resolved_candidates = dedupe_candidates_by_id(resolved_candidates)

    return (
        resolved_candidates,
        pending_coingecko_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    )


def build_asset_sources(debug_by_symbol: dict[str, DebugSymbolInfo]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}

    for _, debug in debug_by_symbol.items():
        if not debug.resolved or not debug.coingecko_id:
            continue

        cid = debug.coingecko_id.lower()
        slot = mapping.setdefault(
            cid,
            {
                "source_type": debug.source_type,
                "input_symbol": debug.input_symbol,
                "input_coingecko_id": debug.input_coingecko_id,
            },
        )

        if slot["source_type"] != debug.source_type:
            slot["source_type"] = "mixed"

        if not slot["input_symbol"] and debug.input_symbol:
            slot["input_symbol"] = debug.input_symbol

        if not slot["input_coingecko_id"] and debug.input_coingecko_id:
            slot["input_coingecko_id"] = debug.input_coingecko_id

    return mapping


def merge_coin_candidates(existing: list[dict], new_coin: dict) -> list[dict]:
    coin_id = str(new_coin.get("id", "")).lower()
    if not coin_id:
        return existing

    seen = {str(c.get("id", "")).lower() for c in existing}
    if coin_id in seen:
        return existing
    return existing + [new_coin]


def mark_skipped(
    asset_key: str,
    coingecko_id: str | None,
    reason: str,
    stage: str,
    skipped_assets: list[str],
    skip_reasons: dict[str, str],
    debug_by_symbol: dict[str, DebugSymbolInfo],
    endpoint: str | None = None,
    http_status: int | None = None,
    request_params: dict | None = None,
    error_message: str | None = None,
    auth_mode: str | None = None,
    base_url: str | None = None,
    api_key_present: bool | None = None,
    auth_header_name: str | None = None,
    universe_filter_status: str | None = None,
    universe_filter_reason: str | None = None,
    candidate_windows_count: int | None = None,
    best_window: dict | None = None,
    structural_score: float | None = None,
    exemplar_consistency_score: float | None = None,
    distance_to_siren_breakdown: float | None = None,
    distance_to_river_breakdown: float | None = None,
    reference_band_passed: bool | None = None,
    raw_similarity: float | None = None,
    label: str | None = None,
):
    if asset_key not in skipped_assets:
        skipped_assets.append(asset_key)

    skip_reasons[asset_key] = reason

    existing = debug_by_symbol.get(asset_key, DebugSymbolInfo())
    debug_by_symbol[asset_key] = DebugSymbolInfo(
        input_symbol=existing.input_symbol,
        input_coingecko_id=existing.input_coingecko_id,
        source_type=existing.source_type,
        resolved=coingecko_id is not None,
        coingecko_id=coingecko_id,
        status="skipped",
        stage=stage,
        reason=reason,
        endpoint=endpoint,
        http_status=http_status,
        request_params=request_params,
        error_message=error_message,
        auth_mode=auth_mode,
        base_url=base_url,
        api_key_present=api_key_present,
        auth_header_name=auth_header_name,
        universe_filter_status=universe_filter_status,
        universe_filter_reason=universe_filter_reason,
        candidate_windows_count=candidate_windows_count,
        best_window=best_window,
        structural_score=structural_score,
        exemplar_consistency_score=exemplar_consistency_score,
        distance_to_siren_breakdown=distance_to_siren_breakdown,
        distance_to_river_breakdown=distance_to_river_breakdown,
        reference_band_passed=reference_band_passed,
        raw_similarity=raw_similarity,
        label=label,
    )


def validate_scan_invariants(
    invalid_or_unresolved_assets: list[str],
    skipped_assets: list[str],
    evaluated_assets: list[str],
    skip_reasons: dict[str, str],
    debug_by_symbol: dict[str, DebugSymbolInfo],
    evaluated_count: int,
) -> None:
    if evaluated_count > 0 and not evaluated_assets:
        raise RuntimeError("Invariant violation: evaluated_count > 0 but evaluated_assets is empty")

    for asset_key in skipped_assets:
        if asset_key not in skip_reasons:
            raise RuntimeError(f"Invariant violation: skipped asset {asset_key} missing skip_reason")
        if asset_key not in debug_by_symbol:
            raise RuntimeError(f"Invariant violation: skipped asset {asset_key} missing debug entry")
        if debug_by_symbol[asset_key].reason != skip_reasons[asset_key]:
            raise RuntimeError(f"Invariant violation: skip reason mismatch for {asset_key}")

    overlaps = (
        set(invalid_or_unresolved_assets) & set(skipped_assets),
        set(invalid_or_unresolved_assets) & set(evaluated_assets),
        set(skipped_assets) & set(evaluated_assets),
    )
    if any(x for x in overlaps):
        raise RuntimeError("Invariant violation: an asset appears in more than one final category")


def classify_universe_filter_from_market(coin: dict) -> tuple[str, str]:
    symbol = str(coin.get("symbol", "")).upper()
    name = str(coin.get("name", ""))
    coin_id = str(coin.get("id", ""))
    market_cap = float(coin.get("market_cap") or 0)

    is_major, major_reason = looks_like_major(symbol, coin_id)
    if is_major:
        return "excluded_major", major_reason or "excluded_major"

    if market_cap > settings.max_market_cap_usd_for_pattern:
        return "excluded_large_cap", "excluded_large_cap"

    if settings.exclude_stables and looks_like_stable(symbol, name, coin_id):
        return "excluded_stablecoin", "excluded_stablecoin_denylist"

    if settings.exclude_tokenized_stocks and looks_like_tokenized_stock(name):
        return "excluded_stablecoin", "excluded_stablecoin_denylist"

    return "included_for_scoring", "included_for_scoring"


async def build_automatic_market_universe(
    client: CoinGeckoClient,
    req: ScanRequest,
    markets: list[dict],
) -> tuple[list[dict], dict[str, DebugSymbolInfo], dict[str, str], dict[str, dict], int, int, list[str]]:
    all_candidates: list[dict] = []
    local_debug: dict[str, DebugSymbolInfo] = {}
    local_skip_reasons: dict[str, str] = {}
    asset_sources: dict[str, dict] = {}

    excluded_symbols = {s.lower() for s in req.exclude_symbols or []}

    for coin in dedupe_candidates_by_id(markets):
        symbol = str(coin.get("symbol", "")).upper()
        coin_id = str(coin.get("id", ""))
        asset_key = make_asset_key_from_id(coin_id)

        if str(coin.get("symbol", "")).lower() in excluded_symbols:
            continue

        market_cap = float(coin.get("market_cap") or 0)
        volume_24h = float(coin.get("total_volume") or 0)

        if market_cap < settings.min_market_cap_usd or volume_24h < settings.min_24h_volume_usd:
            continue

        universe_status, universe_reason = classify_universe_filter_from_market(coin)
        local_debug[asset_key] = DebugSymbolInfo(
            input_symbol=symbol,
            input_coingecko_id=coin_id,
            source_type="market_universe",
            resolved=True,
            coingecko_id=coin_id,
            status="candidate" if universe_status == "included_for_scoring" else "skipped",
            stage="resolve_market_universe",
            reason=None if universe_status == "included_for_scoring" else universe_reason,
            universe_filter_status=universe_status,
            universe_filter_reason=universe_reason,
        )

        if universe_status != "included_for_scoring":
            local_skip_reasons[asset_key] = universe_reason
            continue

        all_candidates.append(coin)
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


def to_compact_scan_result(result: ScanResult) -> CompactScanResult:
    return CompactScanResult(
        coingecko_id=result.coingecko_id,
        symbol=result.symbol,
        name=result.name,
        similarity=result.similarity,
        label=result.label,
    )


async def scan_pattern(req: ScanRequest) -> ScanResponse | CompactScanResponse:
    if req.min_age_days > req.max_age_days:
        raise HTTPException(status_code=400, detail="min_age_days must be <= max_age_days")

    client = CoinGeckoClient()
    universe_total_count = 0
    market_batch_size = req.market_batch_size or req.max_coins_to_evaluate
    market_batch_ids: list[str] = []
    compact_response = req.compact_response
    include_notes = req.include_notes and not compact_response
    return_pre_filter_candidates = req.return_pre_filter_candidates and not compact_response

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

        (
            candidates,
            pending_coingecko_ids,
            resolved_symbols,
            unresolved_symbols,
            resolved_coingecko_ids,
            skip_reasons,
            debug_by_symbol,
        ) = resolve_requested_assets(
            markets=markets,
            symbols=req.symbols,
            coingecko_ids=req.coingecko_ids,
        )

        invalid_coingecko_ids: list[str] = []
        for cid in pending_coingecko_ids:
            asset_key = make_asset_key_from_id(cid)
            snapshot = await client.fetch_coin_snapshot_safe(cid, vs_currency=req.vs_currency)

            if not snapshot.ok:
                reason = snapshot.reason or "invalid_coingecko_id"
                if reason == "history_http_404":
                    reason = "invalid_coingecko_id"
                if reason == "invalid_coingecko_id":
                    invalid_coingecko_ids.append(cid)

                existing = debug_by_symbol.get(asset_key, DebugSymbolInfo(
                    input_symbol=None,
                    input_coingecko_id=cid,
                    source_type="coingecko_id",
                ))
                debug_by_symbol[asset_key] = DebugSymbolInfo(
                    input_symbol=existing.input_symbol,
                    input_coingecko_id=existing.input_coingecko_id,
                    source_type=existing.source_type,
                    resolved=False,
                    coingecko_id=cid,
                    status="invalid",
                    stage="resolve_coingecko_id",
                    reason=reason,
                    endpoint=snapshot.endpoint,
                    http_status=snapshot.http_status,
                    request_params=snapshot.request_params,
                    error_message=snapshot.error_message,
                    auth_mode=snapshot.auth_mode,
                    base_url=snapshot.base_url,
                    api_key_present=snapshot.api_key_present,
                    auth_header_name=snapshot.auth_header_name,
                )
                skip_reasons[asset_key] = reason
                continue

            coin = snapshot.coin or {}
            candidates = merge_coin_candidates(candidates, coin)
            resolved_coingecko_ids.append(cid)

            existing = debug_by_symbol.get(asset_key, DebugSymbolInfo(
                input_symbol=None,
                input_coingecko_id=cid,
                source_type="coingecko_id",
            ))
            debug_by_symbol[asset_key] = DebugSymbolInfo(
                input_symbol=existing.input_symbol,
                input_coingecko_id=existing.input_coingecko_id,
                source_type=existing.source_type,
                resolved=True,
                coingecko_id=str(coin.get("id") or cid),
                status="resolved",
                stage="resolve_coingecko_id",
                reason=None,
                endpoint=snapshot.endpoint,
                http_status=snapshot.http_status,
                request_params=snapshot.request_params,
                error_message=None,
                auth_mode=snapshot.auth_mode,
                base_url=snapshot.base_url,
                api_key_present=snapshot.api_key_present,
                auth_header_name=snapshot.auth_header_name,
            )

        asset_sources = build_asset_sources(debug_by_symbol=debug_by_symbol)

        if not (req.symbols or req.coingecko_ids):
            (
                candidates,
                auto_debug,
                auto_skip_reasons,
                auto_asset_sources,
                universe_total_count,
                market_batch_size,
                market_batch_ids,
            ) = await build_automatic_market_universe(
                client=client,
                req=req,
                markets=markets,
            )
            debug_by_symbol.update(auto_debug)
            skip_reasons.update(auto_skip_reasons)
            asset_sources.update(auto_asset_sources)
        else:
            market_batch_ids = [str(c.get("id", "")).lower() for c in dedupe_candidates_by_id(candidates)]
            universe_total_count = len(market_batch_ids)

        candidates = dedupe_candidates_by_id(candidates)

        evaluated_symbols: list[str] = []
        evaluated_assets: list[str] = []
        skipped_assets: list[str] = []
        results: list[ScanResult] = []
        pre_filter_candidates: list[ScanResult] = []

        for coin in candidates:
            coin_id = str(coin.get("id", ""))
            symbol = str(coin.get("symbol", "")).upper()
            name = str(coin.get("name", ""))

            source_meta = asset_sources.get(
                coin_id.lower(),
                {
                    "source_type": "coingecko_id",
                    "input_symbol": symbol or None,
                    "input_coingecko_id": coin_id,
                },
            )
            asset_key = make_asset_key_from_id(coin_id)

            if not coin_id:
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=None,
                    reason="coingecko_id_missing",
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
                )
                continue

            pre_universe_status, pre_universe_reason = classify_universe_filter_from_market(coin)
            if pre_universe_status != "included_for_scoring":
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason=pre_universe_reason,
                    stage="resolve_asset",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    universe_filter_status=pre_universe_status,
                    universe_filter_reason=pre_universe_reason,
                )
                continue

            requested_history_days = min(450, req.max_age_days)
            effective_history_days, days_capped = client.normalize_history_days(requested_history_days)

            debug_by_symbol[asset_key] = DebugSymbolInfo(
                input_symbol=source_meta.get("input_symbol"),
                input_coingecko_id=source_meta.get("input_coingecko_id"),
                source_type=source_meta.get("source_type"),
                resolved=True,
                coingecko_id=coin_id,
                status="fetching",
                stage="fetch_market_data",
                reason=None,
                endpoint="/coins/{id}/market_chart",
                request_params={
                    "vs_currency": req.vs_currency,
                    "days": str(effective_history_days),
                    "interval": "daily",
                    "requested_days": str(requested_history_days),
                    "plan_limit_days": DEMO_HISTORY_MAX_DAYS if client.auth.mode == "demo" else None,
                    "days_capped_by_plan": days_capped,
                },
                error_message=(
                    f"Requested range was capped to {effective_history_days} days by plan limits."
                    if days_capped else None
                ),
                auth_mode=client.auth.mode,
                base_url=client.auth.base_url,
                api_key_present=client.auth.api_key_present,
                auth_header_name=client.auth.header_name,
                universe_filter_status="included_for_scoring",
                universe_filter_reason="included_for_scoring",
            )

            fetch = await client.fetch_market_history(
                coingecko_id=coin_id,
                vs_currency=req.vs_currency,
                days=requested_history_days,
                interval="daily",
            )

            if not fetch.ok:
                reason = fetch.reason or "history_fetch_failed"
                if reason == "history_http_404":
                    reason = "invalid_coingecko_id"

                if reason == "invalid_coingecko_id" and source_meta.get("source_type") == "coingecko_id":
                    if coin_id.lower() not in invalid_coingecko_ids:
                        invalid_coingecko_ids.append(coin_id.lower())

                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason=reason,
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    endpoint=fetch.endpoint,
                    http_status=fetch.http_status,
                    request_params=fetch.request_params,
                    error_message=fetch.error_message,
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
                    api_key_present=fetch.api_key_present,
                    auth_header_name=fetch.auth_header_name,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
                )
                continue

            chart = fetch.chart or {}
            closes = coingecko_daily_closes(chart)

            behavioral_status, behavioral_reason = classify_behavioral_universe_filter(closes)
            if behavioral_status != "included_for_scoring":
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason=behavioral_reason,
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    endpoint=fetch.endpoint,
                    http_status=fetch.http_status,
                    request_params=fetch.request_params,
                    error_message="Excluded by behavioral universe filter before scoring",
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
                    api_key_present=fetch.api_key_present,
                    auth_header_name=fetch.auth_header_name,
                    universe_filter_status=behavioral_status,
                    universe_filter_reason=behavioral_reason,
                )
                continue

            if len(closes) == 0:
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason="history_empty",
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    endpoint=fetch.endpoint,
                    http_status=fetch.http_status,
                    request_params=fetch.request_params,
                    error_message="prices list is empty after close extraction",
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
                    api_key_present=fetch.api_key_present,
                    auth_header_name=fetch.auth_header_name,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
                )
                continue

            if len(closes) < 30:
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason="insufficient_history",
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    endpoint=fetch.endpoint,
                    http_status=fetch.http_status,
                    request_params=fetch.request_params,
                    error_message=f"need >= 30 closes, got {len(closes)}",
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
                    api_key_present=fetch.api_key_present,
                    auth_header_name=fetch.auth_header_name,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
                )
                continue

            asset_age_days = age_from_chart_days(chart)
            if asset_age_days is None:
                mark_skipped(
                    asset_key=asset_key,
                    coingecko_id=coin_id,
                    reason="history_bad_response_schema",
                    stage="fetch_market_data",
                    skipped_assets=skipped_assets,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    endpoint=fetch.endpoint,
                    http_status=fetch.http_status,
                    request_params=fetch.request_params,
                    error_message="unable to derive age from chart",
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
                    api_key_present=fetch.api_key_present,
                    auth_header_name=fetch.auth_header_name,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
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
                best = find_best_window(closes, req.min_age_days, min(req.max_age_days, len(closes)), stage_mode=req.stage_mode)
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
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    error_message=f"{type(e).__name__}: {e}",
                    auth_mode=fetch.auth_mode,
                    base_url=fetch.base_url,
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

            best.setdefault("early_impulse_score", 0.0)
            best.setdefault("return_to_base_score", 0.0)
            best.setdefault("base_duration_score", 0.0)
            best.setdefault("base_compaction_score", 0.0)
            best.setdefault("right_side_tightening_score", 0.0)
            best.setdefault("breakout_not_started_score", 0.0)
            best.setdefault("late_breakout_penalty", 0.0)
            best.setdefault("post_breakout_extension_penalty", 0.0)
            best.setdefault("selected_window_stage", str(best.get("stage") or "active"))

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
            notes.append(
                f"Setup metrics: early={best['early_impulse_score']}, return={best['return_to_base_score']}, base_dur={best['base_duration_score']}, base_comp={best['base_compaction_score']}, right={best['right_side_tightening_score']}"
            )
            notes.append(
                f"Breakout guard: not_started={best['breakout_not_started_score']}, late_penalty={best['late_breakout_penalty']}, post_ext_penalty={best['post_breakout_extension_penalty']}, window_stage={best['selected_window_stage']}"
            )
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
                early_impulse_score=float(best["early_impulse_score"]),
                return_to_base_score=float(best["return_to_base_score"]),
                base_duration_score=float(best["base_duration_score"]),
                base_compaction_score=float(best["base_compaction_score"]),
                right_side_tightening_score=float(best["right_side_tightening_score"]),
                breakout_not_started_score=float(best["breakout_not_started_score"]),
                late_breakout_penalty=float(best["late_breakout_penalty"]),
                post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
                selected_window_stage=str(best["selected_window_stage"]),
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
                notes=notes if include_notes else [],
            )

            if return_pre_filter_candidates and should_surface_pre_filter_candidate(best):
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
                    early_impulse_score=float(best["early_impulse_score"]),
                    return_to_base_score=float(best["return_to_base_score"]),
                    base_duration_score=float(best["base_duration_score"]),
                    base_compaction_score=float(best["base_compaction_score"]),
                    right_side_tightening_score=float(best["right_side_tightening_score"]),
                    breakout_not_started_score=float(best["breakout_not_started_score"]),
                    late_breakout_penalty=float(best["late_breakout_penalty"]),
                    post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
                    selected_window_stage=str(best["selected_window_stage"]),
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
                early_impulse_score=float(best["early_impulse_score"]),
                return_to_base_score=float(best["return_to_base_score"]),
                base_duration_score=float(best["base_duration_score"]),
                base_compaction_score=float(best["base_compaction_score"]),
                right_side_tightening_score=float(best["right_side_tightening_score"]),
                breakout_not_started_score=float(best["breakout_not_started_score"]),
                late_breakout_penalty=float(best["late_breakout_penalty"]),
                post_breakout_extension_penalty=float(best["post_breakout_extension_penalty"]),
                selected_window_stage=str(best["selected_window_stage"]),
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

        if compact_response:
            return CompactScanResponse(
                pattern_name=req.pattern_name,
                evaluated_count=len(evaluated_assets),
                returned_count=len(final_results),
                market_offset=req.market_offset,
                market_batch_size=market_batch_size or len(candidates),
                market_batch_ids=market_batch_ids,
                results=[to_compact_scan_result(result) for result in final_results],
            )

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
            pre_filter_candidates=final_prefilter if return_pre_filter_candidates else [],
):
    ...
    finally:
        await client.close()
