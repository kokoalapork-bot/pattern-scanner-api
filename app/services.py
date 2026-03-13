from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

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
    MatchBreakdown,
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


def generate_window_lengths(min_age_days: int, max_age_days: int) -> list[int]:
    candidates = [30, 45, 60, 75, 90, 120, 150, 180, 210, 240]
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
    """
    Хотим ловить окно в середине или ближе к концу паттерна,
    но не совсем в самом хвосте графика.
    """
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

    for window_closes, window_len, start_idx, end_idx, total_len in iter_windows(
        closes, min_age_days, max_age_days
    ):
        try:
            result = score_crown_shelf_right_spike(window_closes)
        except Exception:
            continue

        pos = position_bonus(start_idx, end_idx, total_len)

        # raw similarity от scorer-а
        raw_similarity = float(result.similarity)

        # торговая поправка:
        # не убиваем score, но слегка штрафуем окна в самом хвосте
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
            }

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
) -> tuple[list[dict], list[str], list[str]]:
    """
    Возвращает:
    - resolved coin objects
    - resolved_symbols
    - unresolved_symbols
    """
    resolved: list[dict] = []
    resolved_symbols: list[str] = []
    unresolved_symbols: list[str] = []

    by_id = {str(c.get("id")): c for c in markets}
    by_symbol = build_symbol_index(markets)

    if coingecko_ids:
        for cid in coingecko_ids:
            coin = by_id.get(cid)
            if coin:
                resolved.append(coin)
                resolved_symbols.append(str(coin.get("symbol", "")).upper())
        # unresolved для ids тут не выводим отдельным полем, не просили в schema
        return resolved, resolved_symbols, unresolved_symbols

    if symbols:
        for sym in symbols:
            key = sym.lower()
            matches = by_symbol.get(key, [])
            if matches:
                # если по символу несколько монет — берем самую крупную
                chosen = sorted(
                    matches,
                    key=lambda x: float(x.get("market_cap") or 0),
                    reverse=True,
                )[0]
                resolved.append(chosen)
                resolved_symbols.append(sym.upper())
            else:
                unresolved_symbols.append(sym.upper())

    return resolved, resolved_symbols, unresolved_symbols


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

        if explicit_mode:
            candidates, resolved_symbols, unresolved_symbols = resolve_requested_coins(
                markets=markets,
                symbols=req.symbols,
                coingecko_ids=req.coingecko_ids,
            )
        else:
            candidates = []
            excluded_symbols = {s.lower() for s in req.exclude_symbols or []}

            for coin in markets:
                symbol = str(coin.get("symbol", "")).lower()
                name = str(coin.get("name", ""))

                if symbol in excluded_symbols:
                    continue

                market_cap = float(coin.get("market_cap") or 0)
                volume_24h = float(coin.get("total_volume") or 0)

                if market_cap < settings.min_market_cap_usd or volume_24h < settings.min_24h_volume_usd:
                    continue

                if settings.exclude_stables and looks_like_stable(symbol, name):
                    continue

                if settings.exclude_tokenized_stocks and looks_like_tokenized_stock(name):
                    continue

                candidates.append(coin)

                if len(candidates) >= req.max_coins_to_evaluate:
                    break

        results: list[ScanResult] = []

        for coin in candidates:
            coin_id = str(coin["id"])
            symbol = str(coin["symbol"]).upper()
            name = str(coin["name"])

            try:
                chart = await client.get_market_chart(
                    coin_id,
                    vs_currency=req.vs_currency,
                    days=min(450, req.max_age_days if not explicit_mode else 450),
                )
            except httpx.HTTPStatusError:
                skipped_symbols.append(symbol)
                continue
            except httpx.HTTPError:
                skipped_symbols.append(symbol)
                continue
            except Exception:
                skipped_symbols.append(symbol)
                continue

            closes = coingecko_daily_closes(chart)
            if len(closes) < 30:
                skipped_symbols.append(symbol)
                continue

            asset_age_days = age_from_chart_days(chart)
            if asset_age_days is None:
                skipped_symbols.append(symbol)
                continue

            # Для рыночного скана фильтр возраста актива сохраняем.
            # Для explicit-mode не режем актив до поиска лучшего окна.
            if not explicit_mode:
                if asset_age_days < req.min_age_days or asset_age_days > req.max_age_days:
                    skipped_symbols.append(symbol)
                    continue

            best = find_best_window(closes, req.min_age_days, req.max_age_days)
            if best is None:
                skipped_symbols.append(symbol)
                continue

            evaluated_symbols.append(symbol)

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
                    ),
                    notes=notes,
                )
            )

        results.sort(key=lambda x: x.similarity, reverse=True)

        # debug-режим: всегда возвращаем top-k лучших, даже если совпадения средние
        if req.debug:
            final_results = results[: req.top_k]
        else:
            final_results = results[: req.top_k]

        return ScanResponse(
            pattern_name=req.pattern_name,
            evaluated_count=len(evaluated_symbols),
            returned_count=len(final_results),
            resolved_symbols=resolved_symbols,
            unresolved_symbols=unresolved_symbols,
            evaluated_symbols=evaluated_symbols,
            skipped_symbols=skipped_symbols,
            results=final_results,
        )
    finally:
        await client.close()
