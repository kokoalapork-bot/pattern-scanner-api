from __future__ import annotations

from datetime import date, datetime, timezone

from .config import settings
from .data_sources import (
    AssetUniverseItem,
    CoinGeckoClient,
    CoinMarketCapClient,
    MarketDataFetchResult,
    classify_behavioral_universe_filter,
    coingecko_daily_closes,
    parse_date_safe,
)
from .models import (
    BestWindow,
    CompactScanResponse,
    CompactScanResult,
    DebugSymbolInfo,
    MatchBreakdown,
    PhaseGateReport,
    ScanRequest,
    ScanResponse,
    ScanResult,
)
from .patterns import score_crown_shelf_right_spike


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


def listing_date_from_chart(chart: dict) -> date | None:
    prices = chart.get("prices", [])
    if not prices:
        return None
    first_ts_ms = prices[0][0]
    if first_ts_ms is None:
        return None
    try:
        return datetime.utcfromtimestamp(first_ts_ms / 1000.0).date()
    except Exception:
        return None


def generate_window_lengths(min_age_days: int, max_age_days: int) -> list[int]:
    candidates = [14, 18, 21, 24, 28, 30, 36, 45, 60, 75, 90, 120, 150, 180, 210, 240, 300, 360]
    out = [w for w in candidates if min_age_days <= w <= max_age_days]
    if not out:
        out = [max(14, min(max_age_days, min_age_days))]
    return out


def iter_windows(closes: list[float], min_age_days: int, max_age_days: int):
    n = len(closes)
    if n < max(14, min_age_days):
        return
    for w in generate_window_lengths(min_age_days, max_age_days):
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


def compute_phase_gate_scores(closes: list[float]) -> PhaseGateReport:
    if not closes or len(closes) < 14:
        return PhaseGateReport(
            early_dump=False,
            recovery=False,
            crown=False,
            strong_dump_after_crown=False,
            live_base=False,
            early_dump_score=0.0,
            recovery_score=0.0,
            crown_score=0.0,
            strong_dump_after_crown_score=0.0,
            live_base_score=0.0,
            phase_score=0.0,
        )

    n = len(closes)
    first_third_end = max(4, int(n * 0.33))
    second_third_end = max(first_third_end + 3, int(n * 0.66))
    head = closes[:first_third_end]
    middle = closes[first_third_end:second_third_end]
    tail = closes[second_third_end:]

    first_price = closes[0]
    early_low = min(head)
    post_bottom_high = max(middle + tail)
    crown_high = max(middle) if middle else max(closes)
    base_tail = tail[-max(4, len(tail)//3):] if tail else closes[-4:]
    tail_low = min(base_tail)
    tail_high = max(base_tail)
    tail_mean = sum(base_tail) / len(base_tail)

    early_dump_pct = 0.0 if first_price <= 0 else (first_price - early_low) / first_price
    recovery_pct = 0.0 if early_low <= 0 else (post_bottom_high - early_low) / early_low
    crown_vs_middle = 0.0 if early_low <= 0 else (crown_high - (sum(middle)/len(middle) if middle else early_low)) / max(early_low, 1e-9)

    crown_idx = closes.index(max(closes))
    after_crown = closes[crown_idx:] if crown_idx < n - 1 else [closes[-1]]
    after_crown_low = min(after_crown)
    dump_after_crown_pct = 0.0 if closes[crown_idx] <= 0 else (closes[crown_idx] - after_crown_low) / closes[crown_idx]

    base_range_ratio = 1.0 if tail_mean <= 0 else (tail_high - tail_low) / tail_mean
    live_base_score = max(0.0, min(1.0, 1.0 - abs(base_range_ratio - 0.35) / 0.35))

    early_dump_score = max(0.0, min(1.0, early_dump_pct / 0.35))
    recovery_score = max(0.0, min(1.0, recovery_pct / 1.20))
    crown_score = max(0.0, min(1.0, crown_vs_middle / 0.60))
    strong_dump_after_crown_score = max(0.0, min(1.0, dump_after_crown_pct / 0.45))

    phase_score = round(
        (0.22 * early_dump_score + 0.22 * recovery_score + 0.20 * crown_score + 0.20 * strong_dump_after_crown_score + 0.16 * live_base_score) * 100.0,
        2,
    )

    return PhaseGateReport(
        early_dump=early_dump_score >= 0.55,
        recovery=recovery_score >= 0.55,
        crown=crown_score >= 0.45,
        strong_dump_after_crown=strong_dump_after_crown_score >= 0.50,
        live_base=live_base_score >= 0.35,
        early_dump_score=round(early_dump_score * 100.0, 2),
        recovery_score=round(recovery_score * 100.0, 2),
        crown_score=round(crown_score * 100.0, 2),
        strong_dump_after_crown_score=round(strong_dump_after_crown_score * 100.0, 2),
        live_base_score=round(live_base_score * 100.0, 2),
        phase_score=phase_score,
    )


def phase_gates_pass(req: ScanRequest, report: PhaseGateReport) -> bool:
    if req.require_early_dump and not report.early_dump:
        return False
    if req.require_recovery and not report.recovery:
        return False
    if req.require_crown and not report.crown:
        return False
    if req.require_strong_dump_after_crown and not report.strong_dump_after_crown:
        return False
    if req.require_live_base and not report.live_base:
        return False
    return True


def _breakdown_to_model(breakdown) -> MatchBreakdown:
    return MatchBreakdown(
        crown=round(float(getattr(breakdown, "crown", 0.0)), 4),
        drop=round(float(getattr(breakdown, "drop", 0.0)), 4),
        shelf=round(float(getattr(breakdown, "shelf", 0.0)), 4),
        right_spike=round(float(getattr(breakdown, "right_spike", 0.0)), 4),
        reversion=round(float(getattr(breakdown, "reversion", 0.0)), 4),
        asymmetry=round(float(getattr(breakdown, "asymmetry", 0.0)), 4),
        template_shape=round(float(getattr(breakdown, "template_shape", 0.0)), 4),
    )


def _compact_result(r: ScanResult) -> CompactScanResult:
    return CompactScanResult(
        coingecko_id=r.coingecko_id,
        coinmarketcap_id=r.coinmarketcap_id,
        symbol=r.symbol,
        name=r.name,
        similarity=r.similarity,
        age_days=r.age_days,
        label=r.label,
    )


def resolve_history_provider(req: ScanRequest) -> str:
    if req.history_provider == "coingecko":
        return "coingecko"
    if req.history_provider == "coinmarketcap":
        return "coinmarketcap"
    return "coingecko"


async def _fetch_universe(req: ScanRequest) -> tuple[list[AssetUniverseItem], int]:
    total_seen = 0
    universe: list[AssetUniverseItem] = []

    if req.universe_provider == "coinmarketcap":
        client = CoinMarketCapClient()
        start = max(1, req.market_offset + 1)
        step = req.market_batch_size
        pages = req.search_pages or settings.market_universe_pages
        for _ in range(pages):
            page_items = await client.fetch_market_universe_page(start=start, limit=step, convert=req.vs_currency)
            if not page_items:
                break
            total_seen += len(page_items)
            universe.extend(page_items)
            start += step
            if len(universe) >= req.max_coins_to_evaluate or len(page_items) < step:
                break
        return universe, total_seen

    client = CoinGeckoClient()
    page = max(1, (req.market_offset // req.market_batch_size) + 1)
    per_page = req.market_batch_size
    pages = req.search_pages or settings.market_universe_pages
    for _ in range(pages):
        page_items = await client.fetch_market_universe_page(page=page, per_page=per_page, vs_currency=req.vs_currency)
        if not page_items:
            break
        total_seen += len(page_items)
        universe.extend(page_items)
        page += 1
        if len(universe) >= req.max_coins_to_evaluate or len(page_items) < per_page:
            break
    return universe, total_seen


def _apply_universe_filters(req: ScanRequest, items: list[AssetUniverseItem]) -> list[AssetUniverseItem]:
    now_date = datetime.now(timezone.utc).date()
    min_listing_date_after = parse_date_safe(req.min_listing_date_after)
    out: list[AssetUniverseItem] = []
    excluded_symbols = {s.upper() for s in (req.exclude_symbols or [])}
    explicit_symbols = {s.upper() for s in (req.symbols or [])}
    explicit_cg_ids = set(req.coingecko_ids or [])
    explicit_cmc_ids = set(req.coinmarketcap_ids or [])

    for item in items:
        if item.symbol.upper() in excluded_symbols:
            continue

        if explicit_symbols or explicit_cg_ids or explicit_cmc_ids:
            symbol_ok = item.symbol.upper() in explicit_symbols if explicit_symbols else False
            cg_ok = item.coingecko_id in explicit_cg_ids if explicit_cg_ids else False
            cmc_ok = item.coinmarketcap_id in explicit_cmc_ids if explicit_cmc_ids else False
            if not any([symbol_ok, cg_ok, cmc_ok]):
                continue

        ok, _reason = classify_behavioral_universe_filter(
            item,
            max_market_cap_usd=req.max_market_cap_usd,
            min_market_cap_usd=req.min_market_cap_usd,
            min_24h_volume_usd=req.min_24h_volume_usd,
            min_listing_date_after=min_listing_date_after,
            exclude_stables=req.exclude_stables,
            exclude_tokenized_stocks=req.exclude_tokenized_stocks,
            min_age_days=req.min_age_days,
            max_age_days=req.max_age_days,
            now_date=now_date,
        )
        if ok:
            out.append(item)
        if len(out) >= req.max_coins_to_evaluate:
            break
    return out


async def _fetch_history_for_item(item: AssetUniverseItem, req: ScanRequest) -> tuple[str, MarketDataFetchResult]:
    history_provider = resolve_history_provider(req)
    if history_provider == "coingecko":
        cg = CoinGeckoClient()
        if item.coingecko_id:
            return history_provider, await cg.fetch_history_by_coingecko_id(coingecko_id=item.coingecko_id, days=req.max_age_days, vs_currency=req.vs_currency)
        return history_provider, MarketDataFetchResult(
            ok=False,
            endpoint="/coins/{id}/market_chart",
            reason="missing_coingecko_id",
            error_message="CoinMarketCap universe item has no mapped CoinGecko id. Add coingecko_ids explicitly for this asset or enrich resolver.",
        )

    cmc = CoinMarketCapClient()
    if item.coinmarketcap_id is None:
        return history_provider, MarketDataFetchResult(ok=False, endpoint="/v1/cryptocurrency/ohlcv/historical", reason="missing_coinmarketcap_id", error_message="Missing CoinMarketCap id for history fetch.")
    return history_provider, await cmc.fetch_history_by_coinmarketcap_id(coinmarketcap_id=item.coinmarketcap_id, days=req.max_age_days, vs_currency=req.vs_currency)


async def scan_pattern(req: ScanRequest):
    if req.min_age_days > req.max_age_days:
        raise ValueError("min_age_days cannot be greater than max_age_days")

    universe_items, total_seen = await _fetch_universe(req)
    filtered_items = _apply_universe_filters(req, universe_items)

    results: list[ScanResult] = []
    pre_filter_candidates: list[ScanResult] = []
    debug_by_symbol: dict[str, DebugSymbolInfo] = {}

    for item in filtered_items:
        history_provider_used, history_result = await _fetch_history_for_item(item, req)
        debug_by_symbol[item.symbol] = DebugSymbolInfo(
            input_symbol=item.symbol,
            input_coingecko_id=item.coingecko_id,
            input_coinmarketcap_id=item.coinmarketcap_id,
            provider=item.provider,
            resolved=history_result.ok,
            coingecko_id=item.coingecko_id,
            coinmarketcap_id=item.coinmarketcap_id,
            status="ok" if history_result.ok else "skipped",
            reason=history_result.reason,
            endpoint=history_result.endpoint,
            http_status=history_result.http_status,
            request_params=history_result.request_params,
            error_message=history_result.error_message,
            notes=list(history_result.notes or []),
        )
        if not history_result.ok or not history_result.chart:
            continue

        closes = coingecko_daily_closes(history_result.chart)
        if len(closes) < max(14, req.min_age_days):
            continue

        listing_date = item.listing_date or listing_date_from_chart(history_result.chart)
        age_days = age_from_chart_days(history_result.chart) or (len(closes) - 1)

        best_score = -1.0
        best_structural = None
        best_window = None
        candidate_windows_count = 0

        for window_closes, w, start_idx, end_idx, total_len in iter_windows(closes, req.min_age_days, req.max_age_days):
            candidate_windows_count += 1
            structural = score_crown_shelf_right_spike(window_closes)
            if structural.similarity > best_score:
                best_score = structural.similarity
                best_structural = structural
                best_window = (start_idx, end_idx, w)

        if best_structural is None or best_window is None:
            continue

        phase_report = compute_phase_gate_scores(closes)
        blended_similarity = round(settings.structural_score_weight * best_structural.similarity + settings.phase_score_weight * phase_report.phase_score, 2)

        result = ScanResult(
            coingecko_id=item.coingecko_id,
            coinmarketcap_id=item.coinmarketcap_id,
            symbol=item.symbol,
            name=item.name,
            age_days=age_days,
            listing_date=listing_date.isoformat() if listing_date else None,
            market_cap_usd=item.market_cap_usd,
            volume_24h_usd=item.volume_24h_usd,
            similarity=blended_similarity,
            raw_similarity=round(best_structural.similarity, 2),
            structural_score=round(best_structural.similarity, 2),
            phase_score=phase_report.phase_score,
            label=best_structural.label,
            stage=best_structural.stage,
            universe_provider=req.universe_provider,
            history_provider=history_provider_used,
            phase_gate_report=phase_report,
            breakdown=_breakdown_to_model(best_structural.breakdown),
            best_window=BestWindow(
                start_idx=best_window[0],
                end_idx=best_window[1],
                length_days=best_window[2],
                best_age_days=best_window[2],
                candidate_windows_count=candidate_windows_count,
            ),
            notes=list(best_structural.notes) if req.include_notes else [],
            source="market_data",
        )

        pre_filter_candidates.append(result)
        if phase_gates_pass(req, phase_report):
            results.append(result)

    results.sort(key=lambda r: r.similarity, reverse=True)
    pre_filter_candidates.sort(key=lambda r: r.similarity, reverse=True)
    results = results[: req.top_k]
    pre_filter_candidates = pre_filter_candidates[: max(req.top_k, 25)] if req.return_pre_filter_candidates else []

    pages_scanned = max(1, (len(universe_items) + req.market_batch_size - 1) // req.market_batch_size)

    if req.compact_response:
        return CompactScanResponse(
            pattern_name=req.pattern_name,
            universe_provider=req.universe_provider,
            history_provider=resolve_history_provider(req),
            evaluated_count=len(filtered_items),
            returned_count=len(results),
            market_offset=req.market_offset,
            market_batch_size=req.market_batch_size,
            pages_scanned=pages_scanned,
            total_candidates_seen=total_seen,
            results=[_compact_result(r) for r in results],
        )

    return ScanResponse(
        pattern_name=req.pattern_name,
        universe_provider=req.universe_provider,
        history_provider=resolve_history_provider(req),
        evaluated_count=len(filtered_items),
        returned_count=len(results),
        market_offset=req.market_offset,
        market_batch_size=req.market_batch_size,
        pages_scanned=pages_scanned,
        total_candidates_seen=total_seen,
        results=results,
        pre_filter_candidates=pre_filter_candidates,
        debug_by_symbol=debug_by_symbol if req.debug else {},
    )
