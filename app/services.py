from dataclasses import asdict

from fastapi import HTTPException
import httpx

from .config import settings
from .data_sources import (
    CoinGeckoClient,
    coingecko_daily_closes,
    looks_like_stable,
    looks_like_tokenized_stock,
)
from .models import MatchBreakdown, ScanRequest, ScanResponse, ScanResult
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
    candidates = [30, 45, 60, 75, 90, 120, 150, 180]
    out = [w for w in candidates if min_age_days <= w <= max_age_days]
    if not out:
        out = [max(30, min(max_age_days, 60))]
    return out


def iter_windows(closes: list[float], min_age_days: int, max_age_days: int):
    """
    Перебирает несколько подокон внутри всей истории
    и отдает (window_closes, window_len, end_idx).
    """
    n = len(closes)
    if n < max(30, min_age_days):
        return

    window_lengths = generate_window_lengths(min_age_days, max_age_days)

    for w in window_lengths:
        if n < w:
            continue

        # чем длиннее окно, тем реже шаг, чтобы не убить API по CPU
        step = max(5, w // 6)

        # идем по всей истории скользящим окном
        for end in range(w, n + 1, step):
            start = end - w
            yield closes[start:end], w, end

        # гарантируем, что самое последнее окно тоже проверено
        if (n - w) % step != 0:
            yield closes[n - w:n], w, n


def find_best_window(closes: list[float], min_age_days: int, max_age_days: int):
    best_similarity = -1.0
    best_breakdown = None
    best_notes = []
    best_window_len = None
    best_window_end = None

    for window_closes, window_len, end_idx in iter_windows(closes, min_age_days, max_age_days):
        try:
            similarity, breakdown, notes = score_crown_shelf_right_spike(window_closes)
        except Exception:
            continue

        if similarity > best_similarity:
            best_similarity = similarity
            best_breakdown = breakdown
            best_notes = notes
            best_window_len = window_len
            best_window_end = end_idx

    if best_similarity < 0 or best_breakdown is None:
        return None

    return {
        "similarity": best_similarity,
        "breakdown": best_breakdown,
        "notes": best_notes,
        "best_window_len": best_window_len,
        "best_window_end": best_window_end,
    }


async def scan_pattern(req: ScanRequest) -> ScanResponse:
    if req.min_age_days > req.max_age_days:
        raise HTTPException(status_code=400, detail="min_age_days must be <= max_age_days")

    client = CoinGeckoClient()

    try:
        pages = max(1, min(4, (req.max_coins_to_evaluate + 249) // 250))

        try:
            markets = await client.get_markets(vs_currency=req.vs_currency, pages=pages, per_page=250)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=503, detail=f"CoinGecko markets error: {e.response.status_code}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"CoinGecko connection error: {str(e)}")

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
                chart = await client.get_market_chart(
                    coin_id,
                    vs_currency=req.vs_currency,
                    days=min(450, req.max_age_days),
                )
            except httpx.HTTPStatusError:
                continue
            except httpx.HTTPError:
                continue
            except Exception:
                continue

            closes = coingecko_daily_closes(chart)
            if len(closes) < max(30, req.min_age_days):
                continue

            asset_age_days = age_from_chart_days(chart)
            if asset_age_days is None:
                continue

            # Актив должен быть в нужном возрастном диапазоне
            if asset_age_days < req.min_age_days or asset_age_days > req.max_age_days:
                continue

            best = find_best_window(closes, req.min_age_days, req.max_age_days)
            if best is None:
                continue

            notes = best["notes"][:]
            notes.append(f"Лучшее окно: {best['best_window_len']} дней.")
            notes.append(f"Конец лучшего окна: свеча #{best['best_window_end']} в доступной истории.")

            if not req.include_notes:
                notes = []

            results.append(
                ScanResult(
                    coingecko_id=coin_id,
                    symbol=symbol,
                    name=name,
                    age_days=asset_age_days,
                    market_cap_usd=float(coin.get("market_cap") or 0),
                    volume_24h_usd=float(coin.get("total_volume") or 0),
                    similarity=best["similarity"],
                    breakdown=MatchBreakdown(**asdict(best["breakdown"])),
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
