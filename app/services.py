
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .config import get_settings
from .data_sources import CoinGeckoClient
from .models import (
    CompactScanResponse,
    CompactScanResult,
    DebugSymbolInfo,
    ScanRequest,
    ScanResponse,
    ScanResult,
)
from .patterns import score_crown_shelf_right_spike


STABLE_KEYWORDS = {
    "usd", "usdt", "usdc", "dai", "tusd", "usde", "fdusd", "usdd", "eurc", "pyusd",
    "susd", "gusd", "usdb", "usdm", "rlusd",
}
TOKENIZED_STOCK_KEYWORDS = {"tesla", "apple", "nvidia", "microsoft", "gold", "silver", "etf", "stock", "share"}


def _is_stable(coin: dict[str, Any]) -> bool:
    text = f"{coin.get('id','')} {coin.get('symbol','')} {coin.get('name','')}".lower()
    return any(word in text.split() or word in text for word in STABLE_KEYWORDS)


def _is_tokenized_stock(coin: dict[str, Any]) -> bool:
    text = f"{coin.get('id','')} {coin.get('symbol','')} {coin.get('name','')}".lower()
    return any(word in text for word in TOKENIZED_STOCK_KEYWORDS)


def _parse_iso_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _score_symbol_candidate(item: dict[str, Any], target_symbol: str) -> tuple[int, int, int]:
    symbol_match = 1 if str(item.get("symbol", "")).upper() == target_symbol.upper() else 0
    # lower market_cap_rank is better; treat missing as very weak
    rank = item.get("market_cap_rank")
    rank_score = 1_000_000 if rank in (None, 0) else int(rank)
    # prefer ids that are not clearly stale duplicates like "siren" over "siren-2" only when rank is better
    return (
        symbol_match,
        -rank_score,
        -len(str(item.get("id", ""))),
    )


async def _resolve_symbol(client: CoinGeckoClient, symbol: str) -> str | None:
    matches = await client.search_symbol(symbol)
    if not matches:
        return None

    exact = [item for item in matches if str(item.get("symbol", "")).upper() == symbol.upper()]
    candidates = exact or matches

    best = sorted(candidates, key=lambda item: _score_symbol_candidate(item, symbol), reverse=True)[0]
    return best.get("id")


async def _collect_universe(req: ScanRequest, client: CoinGeckoClient) -> tuple[list[dict[str, Any]], int, int]:
    if req.coingecko_ids:
        items = []
        for coin_id in req.coingecko_ids:
            try:
                coin = await client.fetch_coin(coin_id)
                market = coin.get("market_data", {})
                items.append(
                    {
                        "id": coin["id"],
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "market_cap": market.get("market_cap", {}).get(req.vs_currency, 0),
                        "total_volume": market.get("total_volume", {}).get(req.vs_currency, 0),
                        "atl_date": None,
                        "ath_date": market.get("ath_date", {}).get(req.vs_currency),
                        "genesis_date": coin.get("genesis_date"),
                        "market_cap_rank": market.get("market_cap_rank"),
                    }
                )
            except httpx.HTTPError:
                continue
        return items, 1, len(items)

    ids_from_symbols = []
    if req.symbols:
        for sym in req.symbols:
            resolved = await _resolve_symbol(client, sym)
            if resolved:
                ids_from_symbols.append(resolved)
        if ids_from_symbols:
            req.coingecko_ids = ids_from_symbols
            return await _collect_universe(req, client)

    per_page = req.market_batch_size or min(250, req.max_coins_to_evaluate)
    page = req.market_offset // per_page + 1
    markets = await client.fetch_markets(vs_currency=req.vs_currency, page=page, per_page=per_page)
    return markets, 1, len(markets)


def _age_days_from_history(history: dict[str, Any]) -> int | None:
    prices = history.get("prices", [])
    if not prices:
        return None
    first = prices[0]
    if not (isinstance(first, list) and len(first) >= 2):
        return None
    try:
        ts_ms = int(first[0])
        first_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - first_dt).days)
    except Exception:
        return None


def _age_days_fallback(coin: dict[str, Any]) -> int | None:
    now = datetime.now(timezone.utc)
    for key in ("genesis_date", "ath_date", "atl_date"):
        dt = _parse_iso_date(coin.get(key))
        if dt is not None:
            return max(0, (now - dt).days)
    return None


async def scan_pattern(req: ScanRequest):
    settings = get_settings()
    client = CoinGeckoClient()

    if req.min_age_days > req.max_age_days:
        raise ValueError("min_age_days must be <= max_age_days")

    universe, pages_scanned, total_candidates_seen = await _collect_universe(req, client)
    evaluated: list[ScanResult] = []
    exclude_symbols_upper = {s.upper() for s in (req.exclude_symbols or [])}

    for coin in universe:
        if settings.exclude_stables and _is_stable(coin):
            continue
        if settings.exclude_tokenized_stocks and _is_tokenized_stock(coin):
            continue
        if coin.get("symbol", "").upper() in exclude_symbols_upper:
            continue

        market_cap = coin.get("market_cap") or coin.get("market_cap_usd")
        volume = coin.get("total_volume") or coin.get("volume_24h_usd") or 0.0
        if market_cap is not None and market_cap < settings.min_market_cap_usd:
            continue
        if volume is not None and volume < settings.min_24h_volume_usd:
            continue

        coin_id = coin["id"]
        try:
            history = await client.fetch_market_chart(coin_id, vs_currency=req.vs_currency, days=min(req.max_age_days, 450))
        except httpx.HTTPError:
            continue

        prices = [p[1] for p in history.get("prices", []) if isinstance(p, list) and len(p) >= 2]
        if len(prices) < 30:
            continue

        age_days = _age_days_from_history(history) or _age_days_fallback(coin)
        if age_days is None:
            continue
        if not (req.min_age_days <= age_days <= req.max_age_days):
            continue

        score = score_crown_shelf_right_spike(prices)
        evaluated.append(
            ScanResult(
                coingecko_id=coin_id,
                symbol=str(coin.get("symbol", "")).upper(),
                name=coin.get("name", coin_id),
                age_days=age_days,
                market_cap_usd=float(market_cap) if market_cap is not None else None,
                volume_24h_usd=float(volume) if volume is not None else None,
                similarity=score.similarity,
                raw_similarity=score.raw_similarity,
                label=score.label,
                stage=score.stage,
                structural_score=score.structural_score,
                breakdown=score.breakdown,
                best_window=score.best_window,
                notes=score.notes if req.include_notes else [],
                source="market_data",
            )
        )

        if len(evaluated) >= req.max_coins_to_evaluate:
            break

    evaluated.sort(key=lambda x: x.similarity, reverse=True)
    results = evaluated[: req.top_k]

    if req.compact_response:
        return CompactScanResponse(
            pattern_name=req.pattern_name,
            evaluated_count=len(evaluated),
            returned_count=len(results),
            market_offset=req.market_offset,
            market_batch_size=req.market_batch_size or min(250, req.max_coins_to_evaluate),
            pages_scanned=pages_scanned,
            total_candidates_seen=total_candidates_seen,
            results=[
                CompactScanResult(
                    coingecko_id=item.coingecko_id,
                    symbol=item.symbol,
                    name=item.name,
                    similarity=item.similarity,
                    age_days=item.age_days,
                    label=item.label,
                )
                for item in results
            ],
        )

    return ScanResponse(
        pattern_name=req.pattern_name,
        evaluated_count=len(evaluated),
        returned_count=len(results),
        market_offset=req.market_offset,
        market_batch_size=req.market_batch_size or min(250, req.max_coins_to_evaluate),
        pages_scanned=pages_scanned,
        total_candidates_seen=total_candidates_seen,
        results=results,
        pre_filter_candidates=evaluated[: min(len(evaluated), max(req.top_k, 20))] if req.return_pre_filter_candidates else [],
        debug_by_symbol={},
    )
