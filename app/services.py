from dataclasses import asdict

from fastapi import HTTPException

from .config import settings
from .data_sources import (
    CoinGeckoClient,
    age_in_days,
    coingecko_daily_closes,
    looks_like_stable,
    looks_like_tokenized_stock,
    parse_date,
)
from .models import MatchBreakdown, ScanRequest, ScanResponse, ScanResult
from .patterns import score_crown_shelf_right_spike


async def scan_pattern(req: ScanRequest) -> ScanResponse:
    if req.min_age_days > req.max_age_days:
        raise HTTPException(status_code=400, detail="min_age_days must be <= max_age_days")

    client = CoinGeckoClient()

    try:
        pages = max(1, min(8, (req.max_coins_to_evaluate + 249) // 250))
        markets = await client.get_markets(vs_currency=req.vs_currency, pages=pages, per_page=250)

        candidates = []
        requested_symbols = {s.lower() for s in req.symbols or []}
        excluded_symbols = {s.lower() for s in req.exclude_symbols or []}

        for coin in markets:
            symbol = str(coin.get("symbol", "")).lower()
            name = str(coin.get("name", ""))

            if requested_symbols and symbol not in requested_symbols:
                continue
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

        results = []

        for coin in candidates:
            coin_id = coin["id"]
            symbol = coin["symbol"].upper()
            name = coin["name"]

            try:
                coin_meta = await client.get_coin(coin_id)
                chart = await client.get_market_chart(
                    coin_id,
                    vs_currency=req.vs_currency,
                    days=min(450, req.max_age_days),
                )
            except Exception:
                continue

            closes = coingecko_daily_closes(chart)
            if len(closes) < max(30, req.min_age_days):
                continue

            genesis_dt = parse_date(coin_meta.get("genesis_date"))
            history_first_ts = chart.get("prices", [[None, None]])[0][0] if chart.get("prices") else None

            age_days = age_in_days(genesis_dt, history_first_ts)
            if age_days is None:
                continue

            if age_days < req.min_age_days or age_days > req.max_age_days:
                continue

            try:
                similarity, breakdown, notes = score_crown_shelf_right_spike(closes)
            except Exception:
                continue

            if not req.include_notes:
                notes = []

            results.append(
                ScanResult(
                    coingecko_id=coin_id,
                    symbol=symbol,
                    name=name,
                    age_days=age_days,
                    market_cap_usd=float(coin.get("market_cap") or 0),
                    volume_24h_usd=float(coin.get("total_volume") or 0),
                    similarity=similarity,
                    breakdown=MatchBreakdown(**asdict(breakdown)),
                    notes=notes,
                )
            )

        results.sort(key=lambda x: x.similarity, reverse=True)
        results = results[: req.top_k]

        return ScanResponse(
            pattern_name=req.pattern_name,
            evaluated_count=len(candidates),
            returned_count=len(results),
            results=results,
        )
    finally:
        await client.close()
