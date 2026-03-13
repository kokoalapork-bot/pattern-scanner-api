from dataclasses import asdict
from typing import Dict, Optional

import httpx
from fastapi import HTTPException

from .config import settings
from .data_sources import (
    CoinGeckoClient,
    coingecko_daily_closes,
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
    candidates = [30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 300, 360, 420, 450]
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
        raw_similarity = float(result.similarity)
        effective_similarity = raw_similarity * (0.72 + 0.28 * pos)

        if (total_len - end_idx) <= 1:
            effective_similarity *= 0.50

        if effective_similarity > best_effective:
            best_effective = effective_similarity
            best = {
                "similarity": round(effective_similarity, 2),
                "raw_similarity": round(raw_similarity, 2),
                "label": result.label,
                "stage": result.stage,
                "breakdown": result.breakdown,
                "notes": result.notes,
                "best_window_len": window_len,
                "best_window_start": start_idx,
                "best_window_end": end_idx,
                "best_age_days": window_len,
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
    requested_keys: list[str] = []

    by_id = {str(c.get("id")): c for c in markets}
    by_symbol = build_symbol_index(markets)

    if coingecko_ids:
        for cid in coingecko_ids:
            requested_keys.append(cid)
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
        return resolved, resolved_symbols, unresolved_symbols, skip_reasons, debug_by_symbol, requested_keys

    if symbols:
        for sym in symbols:
            key = sym.upper()
            requested_keys.append(key)
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

    return resolved, resolved_symbols, unresolved_symbols, skip_reasons, debug_by_symbol, requested_keys


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
    candidate_windows_count: int | None = None,
    best_window: dict | None = None,
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
        candidate_windows_count=candidate_windows_count,
        best_window=best_window,
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
            (
                candidates,
                resolved_symbols,
                unresolved_symbols,
                initial_skip_reasons,
                initial_debug,
                _requested_keys,
            ) = resolve_requested_coins(
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

                if settings.exclude_stables and looks_like_stable(str(coin.get("symbol", "")), name):
                    continue

                if settings.exclude_tokenized_stocks and looks_like_tokenized_stock(name):
                    continue

                candidates.append(coin)
                debug_by_symbol[symbol] = DebugSymbolInfo(
                    input_symbol=symbol,
                    resolved=True,
                    coingecko_id=coin_id,
                    status="candidate",
                    stage="resolve_symbol",
                    reason=None,
                )

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
                )
                continue

            history_days = min(450, req.max_age_days)

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
                    "days": str(history_days),
                    "interval": "daily",
                },
                auth_mode=client.auth.mode,
                base_url=client.auth.base_url,
                api_key_present=client.auth.api_key_present,
                auth_header_name=client.auth.header_name,
            )

            fetch = await client.fetch_market_history(
                coingecko_id=coin_id,
                vs_currency=req.vs_currency,
                days=history_days,
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
                )
                continue

            chart = fetch.chart or {}
            closes = coingecko_daily_closes(chart)

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
            )

            try:
                best = find_best_window(closes, req.min_age_days, req.max_age_days)
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
                candidate_windows_count=int(best["candidate_windows_count"]),
                best_window={
                    "start_idx": int(best["best_window_start"]),
                    "end_idx": int(best["best_window_end"]),
                    "length_days": int(best["best_window_len"]),
                    "best_age_days": int(best["best_age_days"]),
                },
                raw_similarity=float(best["raw_similarity"]),
                label=str(best["label"]),
            )

            notes = list(best["notes"])
            if req.include_notes:
                notes.append(f"Best window: {best['best_window_start']}..{best['best_window_end']}")
                notes.append(f"Best window length: {best['best_window_len']} days")
                notes.append(f"Raw similarity: {best['raw_similarity']}%")
            else:
                notes = []

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
                    label=str(best["label"]),
                    stage=str(best["stage"]),
                    breakdown=MatchBreakdown(**asdict(best["breakdown"])),
                    best_window=BestWindow(
                        start_idx=int(best["best_window_start"]),
                        end_idx=int(best["best_window_end"]),
                        length_days=int(best["best_window_len"]),
                        best_age_days=int(best["best_age_days"]),
                        candidate_windows_count=int(best["candidate_windows_count"]),
                    ),
                    notes=notes,
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
