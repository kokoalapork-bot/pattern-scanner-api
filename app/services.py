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

# Seed calibration profiles for exemplar-consistency.
# Pattern remains primary; these only constrain valid market-shaped variants.
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
    "crown": (0.20, 0.18),       # lower tolerance, upper tolerance
    "drop": (0.14, 0.16),
    "shelf": (0.18, 0.16),
    "right_spike": (0.18, 0.18),
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
        return 0.35

    if right_tail <= 1:
        return 0.20
    if right_tail <= 3:
        return 0.40
    if right_tail <= 5:
        return 0.60

    target = 0.82
    distance = abs(end_ratio - target)
    pos_score = max(0.0, 1.0 - distance / 0.35)

    if start_ratio > 0.80:
        pos_score *= 0.60
    elif start_ratio > 0.70:
        pos_score *= 0.80

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

    # soft calibration score: 0..100, bounded and not replacing structural score
    exemplar_consistency = max(0.0, 100.0 * (1.0 - min(1.0, nearest_distance / 0.45)))
    reference_band_passed = compute_reference_band_passed(candidate)

    return {
        "exemplar_consistency_score": round(exemplar_consistency, 2),
        "distance_to_siren_breakdown": round(distance_to_siren, 4),
        "distance_to_river_breakdown": round(distance_to_river, 4),
        "reference_band_passed": reference_band_passed,
    }


def classify_final_label(
    base_label: str,
    structural_score: float,
    exemplar_consistency_score: float,
    reference_band_passed: bool,
    universe_filter_status: str,
) -> str:
    if universe_filter_status != "included_for_scoring":
        return "reject"

    if structural_score < 30.0:
        return "reject"

    if structural_score >= 65.0 and exemplar_consistency_score >= 55.0 and reference_band_passed:
        return "strong match"

    if structural_score >= 45.0 and exemplar_consistency_score >= 30.0:
        if base_label == "weak-crown variant":
            return "weak-crown variant"
        return "partial match"

    return "weak match"


def combine_scores(structural_score: float, exemplar_consistency_score: float) -> float:
    return round(
        settings.structural_score_weight * structural_score
        + settings.exemplar_consistency_weight * exemplar_consistency_score,
        2,
    )


def find_best_window(closes: list[float], min_age_days: int, max_age_days: int):
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
        effective_structural = structural_score * (0.72 + 0.28 * pos)

        if (total_len - end_idx) <= 1:
            effective_structural *= 0.50

        exemplar_metrics = compute_exemplar_metrics(result.breakdown)
        final_score = combine_scores(
            structural_score=effective_structural,
            exemplar_consistency_score=float(exemplar_metrics["exemplar_consistency_score"]),
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


def resolve_requested_coins(
    markets: list[dict],
    symbols: Optional[list[str]],
    coingecko_ids: Optional[list[str]],
):
    resolved: list[dict] = []
    resolved_symbols: list[str] = []
    unresolved_symbols: list[str] = []
    skip_reasons: dict[str, str] = {}
    debug_by_symbol: dict[str, DebugSymbolInfo] = {}

    by_id = {str(c.get("id")): c for c in markets}
    by_symbol = build_symbol_index(markets)

    if coingecko_ids:
        for cid in coingecko_ids:
            coin = by_id.get(cid)
            if coin:
                sym = str(coin.get("symbol", "")).upper()
                resolved.append(coin)
                resolved_symbols.append(sym)
                debug_by_symbol[sym] = DebugSymbolInfo(
                    input_symbol=cid,
                    resolved=True,
                    coingecko_id=str(coin.get("id")),
                    status="resolved",
                    stage="resolve_symbol",
                    reason=None,
                )
            else:
                unresolved_symbols.append(cid)
                skip_reasons[cid] = "unresolved_symbol"
                debug_by_symbol[cid] = DebugSymbolInfo(
                    input_symbol=cid,
                    resolved=False,
                    coingecko_id=None,
                    status="unresolved",
                    stage="resolve_symbol",
                    reason="unresolved_symbol",
                )
        return resolved, resolved_symbols, unresolved_symbols, skip_reasons, debug_by_symbol

    if symbols:
        for sym in symbols:
            key = sym.upper()
            matches = by_symbol.get(sym.lower(), [])
            if matches:
                chosen = sorted(matches, key=lambda x: float(x.get("market_cap") or 0), reverse=True)[0]
                resolved.append(chosen)
                resolved_symbols.append(key)
                debug_by_symbol[key] = DebugSymbolInfo(
                    input_symbol=key,
                    resolved=True,
                    coingecko_id=str(chosen.get("id")),
                    status="resolved",
                    stage="resolve_symbol",
                    reason=None,
                )
            else:
                unresolved_symbols.append(key)
                skip_reasons[key] = "unresolved_symbol"
                debug_by_symbol[key] = DebugSymbolInfo(
                    input_symbol=key,
                    resolved=False,
                    coingecko_id=None,
                    status="unresolved",
                    stage="resolve_symbol",
                    reason="unresolved_symbol",
                )

    return resolved, resolved_symbols, unresolved_symbols, skip_reasons, debug_by_symbol


def mark_skipped(
    symbol: str,
    coingecko_id: str | None,
    reason: str,
    stage: str,
    skipped_symbols: list[str],
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
    if symbol not in skipped_symbols:
        skipped_symbols.append(symbol)

    skip_reasons[symbol] = reason
    debug_by_symbol[symbol] = DebugSymbolInfo(
        input_symbol=symbol,
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
    unresolved_symbols: list[str],
    skipped_symbols: list[str],
    evaluated_symbols: list[str],
    skip_reasons: dict[str, str],
    debug_by_symbol: dict[str, DebugSymbolInfo],
    evaluated_count: int,
) -> None:
    if evaluated_count > 0 and not evaluated_symbols:
        raise RuntimeError("Invariant violation: evaluated_count > 0 but evaluated_symbols is empty")

    for symbol in skipped_symbols:
        if symbol not in skip_reasons:
            raise RuntimeError(f"Invariant violation: skipped symbol {symbol} missing skip_reason")
        if symbol not in debug_by_symbol:
            raise RuntimeError(f"Invariant violation: skipped symbol {symbol} missing debug entry")
        if debug_by_symbol[symbol].reason != skip_reasons[symbol]:
            raise RuntimeError(f"Invariant violation: skip reason mismatch for {symbol}")

    final_sets_overlap = (
        set(unresolved_symbols) & set(skipped_symbols),
        set(unresolved_symbols) & set(evaluated_symbols),
        set(skipped_symbols) & set(evaluated_symbols),
    )
    if any(overlap for overlap in final_sets_overlap):
        raise RuntimeError("Invariant violation: a symbol appears in more than one final category")


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


async def scan_pattern(req: ScanRequest) -> ScanResponse:
    if req.min_age_days > req.max_age_days:
        raise HTTPException(status_code=400, detail="min_age_days must be <= max_age_days")

    client = CoinGeckoClient()

    try:
        pages = max(1, min(6, (req.max_coins_to_evaluate + 249) // 250))

        try:
            markets = await client.get_markets(
                vs_currency=req.vs_currency,
                pages=pages,
                per_page=250,
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=503, detail=f"CoinGecko markets error: {e.response.status_code}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"CoinGecko connection error: {str(e)}")

        explicit_mode = bool(req.symbols or req.coingecko_ids)

        resolved_symbols: list[str] = []
        unresolved_symbols: list[str] = []
        evaluated_symbols: list[str] = []
        skipped_symbols: list[str] = []
        skip_reasons: dict[str, str] = {}
        debug_by_symbol: dict[str, DebugSymbolInfo] = {}

        if explicit_mode:
            candidates, resolved_symbols, unresolved_symbols, initial_skip_reasons, initial_debug = resolve_requested_coins(
                markets=markets,
                symbols=req.symbols,
                coingecko_ids=req.coingecko_ids,
            )
            skip_reasons.update(initial_skip_reasons)
            debug_by_symbol.update(initial_debug)
        else:
            candidates = []
            excluded_symbols = {s.lower() for s in req.exclude_symbols or []}

            for coin in markets:
                symbol = str(coin.get("symbol", "")).upper()
                name = str(coin.get("name", ""))
                coin_id = str(coin.get("id", ""))

                if str(coin.get("symbol", "")).lower() in excluded_symbols:
                    continue

                market_cap = float(coin.get("market_cap") or 0)
                volume_24h = float(coin.get("total_volume") or 0)

                if market_cap < settings.min_market_cap_usd or volume_24h < settings.min_24h_volume_usd:
                    continue

                universe_status, universe_reason = classify_universe_filter_from_market(coin)
                debug_by_symbol[symbol] = DebugSymbolInfo(
                    input_symbol=symbol,
                    resolved=True,
                    coingecko_id=coin_id,
                    status="candidate" if universe_status == "included_for_scoring" else "skipped",
                    stage="resolve_symbol",
                    reason=None if universe_status == "included_for_scoring" else universe_reason,
                    universe_filter_status=universe_status,
                    universe_filter_reason=universe_reason,
                )

                if universe_status != "included_for_scoring":
                    skipped_symbols.append(symbol)
                    skip_reasons[symbol] = universe_reason
                    continue

                candidates.append(coin)

                if len(candidates) >= req.max_coins_to_evaluate:
                    break

        results: list[ScanResult] = []

        for coin in candidates:
            coin_id = str(coin.get("id", ""))
            symbol = str(coin.get("symbol", "")).upper()
            name = str(coin.get("name", ""))

            if not coin_id:
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=None,
                    reason="coingecko_id_missing",
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    universe_filter_status="included_for_scoring",
                    universe_filter_reason="included_for_scoring",
                )
                continue

            pre_universe_status, pre_universe_reason = classify_universe_filter_from_market(coin)
            if pre_universe_status != "included_for_scoring":
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason=pre_universe_reason,
                    stage="resolve_symbol",
                    skipped_symbols=skipped_symbols,
                    skip_reasons=skip_reasons,
                    debug_by_symbol=debug_by_symbol,
                    universe_filter_status=pre_universe_status,
                    universe_filter_reason=pre_universe_reason,
                )
                continue

            requested_history_days = min(450, req.max_age_days)
            effective_history_days, days_capped = client.normalize_history_days(requested_history_days)

            debug_by_symbol[symbol] = DebugSymbolInfo(
                input_symbol=symbol,
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
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason=fetch.reason or "history_fetch_failed",
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason=behavioral_reason,
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="history_empty",
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="insufficient_history",
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="history_bad_response_schema",
                    stage="fetch_market_data",
                    skipped_symbols=skipped_symbols,
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

            if not explicit_mode and (asset_age_days < req.min_age_days or asset_age_days > req.max_age_days):
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="filtered_before_scoring",
                    stage="filter_result",
                    skipped_symbols=skipped_symbols,
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

            debug_by_symbol[symbol] = DebugSymbolInfo(
                input_symbol=symbol,
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
                best = find_best_window(closes, req.min_age_days, min(req.max_age_days, len(closes)))
            except ScoringError as e:
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="scoring_error",
                    stage="score_windows",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="window_generation_failed",
                    stage="build_windows",
                    skipped_symbols=skipped_symbols,
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
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="insufficient_history",
                    stage="build_windows",
                    skipped_symbols=skipped_symbols,
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

            final_label = classify_final_label(
                base_label=str(best["base_label"]),
                structural_score=float(best["structural_score"]),
                exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                reference_band_passed=bool(best["reference_band_passed"]),
                universe_filter_status="included_for_scoring",
            )

            if final_label == "reject":
                mark_skipped(
                    symbol=symbol,
                    coingecko_id=coin_id,
                    reason="filtered_after_scoring",
                    stage="score_windows",
                    skipped_symbols=skipped_symbols,
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
                    label=final_label,
                )
                continue

            evaluated_symbols.append(symbol)
            debug_by_symbol[symbol] = DebugSymbolInfo(
                input_symbol=symbol,
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
                raw_similarity=float(best["raw_similarity"]),
                label=final_label,
            )

            notes = list(best["notes"])
            notes.append(
                f"Structural score: {best['structural_score']} | "
                f"Exemplar consistency: {best['exemplar_consistency_score']}"
            )
            notes.append(
                f"Distances: siren={best['distance_to_siren_breakdown']}, "
                f"river={best['distance_to_river_breakdown']}"
            )
            if not best["reference_band_passed"]:
                notes.append("Reference band guardrail failed: cannot be promoted to strong match.")

            if req.include_notes:
                notes.append(f"Best window: {best['best_window_start']}..{best['best_window_end']}")
                notes.append(f"Best window length: {best['best_window_len']} days")
                notes.append(f"Raw similarity: {best['raw_similarity']}%")

            results.append(
                ScanResult(
                    coingecko_id=coin_id,
                    symbol=symbol,
                    name=name,
                    age_days=asset_age_days,
                    market_cap_usd=float(coin.get("market_cap") or 0),
                    volume_24h_usd=float(coin.get("total_volume") or 0),
                    similarity=float(best["similarity"]),
                    raw_similarity=float(best["raw_similarity"]),
                    label=final_label,
                    stage=str(best["stage"]),
                    structural_score=float(best["structural_score"]),
                    exemplar_consistency_score=float(best["exemplar_consistency_score"]),
                    distance_to_siren_breakdown=float(best["distance_to_siren_breakdown"]),
                    distance_to_river_breakdown=float(best["distance_to_river_breakdown"]),
                    reference_band_passed=bool(best["reference_band_passed"]),
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
                    notes=notes if req.include_notes else [],
                )
            )

        results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = results[: req.top_k]

        validate_scan_invariants(
            unresolved_symbols=unresolved_symbols,
            skipped_symbols=skipped_symbols,
            evaluated_symbols=evaluated_symbols,
            skip_reasons=skip_reasons,
            debug_by_symbol=debug_by_symbol,
            evaluated_count=len(evaluated_symbols),
        )

        return ScanResponse(
            pattern_name=req.pattern_name,
            evaluated_count=len(evaluated_symbols),
            returned_count=len(final_results),
            resolved_symbols=resolved_symbols,
            unresolved_symbols=unresolved_symbols,
            evaluated_symbols=evaluated_symbols,
            skipped_symbols=skipped_symbols,
            skip_reasons=skip_reasons,
            debug_by_symbol=debug_by_symbol,
            results=final_results,
        )
    finally:
        await client.close()
