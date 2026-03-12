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
    """
    min/max трактуются как допустимая длина локального окна.
    """
    base = [30, 45, 60, 75, 90, 120, 150, 180, 210, 240]
    out = [w for w in base if min_age_days <= w <= max_age_days]

    if not out:
        fallback = max(30, min(max_age_days, 60))
        out = [fallback]

    return out


def iter_windows(closes: list[float], min_age_days: int, max_age_days: int):
    """
    Перебирает подокна внутри всей истории.
    """
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


def tradability_position_bonus(start_idx: int, end_idx: int, total_len: int) -> float:
    """
    Хотим торгуемый паттерн:
    - НЕ в самом конце графика
    - НЕ слишком рано
    - лучше в середине или ближе к правой части, но с запасом справа

    Возвращает множитель 0..1.
    """
    if total_len <= 0:
        return 0.0

    end_ratio = end_idx / total_len
    start_ratio = start_idx / total_len
    right_tail = total_len - end_idx

    # Слишком рано — еще неинтересно
    if end_ratio < 0.35:
        return 0.35

    # Слишком поздно — паттерн уже "в самом конце графика"
    if right_tail <= 2:
        return 0.20
    if right_tail <= 5:
        return 0.40
    if right_tail <= 8:
        return 0.60

    # Идеальная зона: окно заканчивается примерно на 70-90% истории
    # То есть паттерн уже сформирован, но справа еще есть место для торговли.
    center_target = 0.82
    distance = abs(end_ratio - center_target)

    position_score = max(0.0, 1.0 - distance / 0.35)

    # Доп.бонус если окно не стартует слишком поздно:
    # паттерн должен занимать осмысленный участок, а не хвостик.
    start_penalty = 1.0
    if start_ratio > 0.80:
        start_penalty = 0.6
    elif start_ratio > 0.70:
        start_penalty = 0.8

    return max(0.0, min(1.0, position_score * start_penalty))


def find_best_window(closes: list[float], min_age_days: int, max_age_days: int):
    best_effective_similarity = -1.0
    best_raw_similarity = -1.0
    best_breakdown = None
    best_notes = []
    best_window_len = None
    best_window_start = None
    best_window_end = None
    best_position_bonus = None

    for window_closes, window_len, start_idx, end_idx, total_len in iter_windows(closes, min_age_days, max_age_days):
        try:
            raw_similarity, breakdown, notes = score_crown_shelf_right_spike(window_closes)
        except Exception:
            continue

        pos_bonus = tradability_position_bonus(start_idx, end_idx, total_len)

        # Эффективный score с учетом торговой пригодности позиции окна
        effective_similarity = raw_similarity * (0.70 + 0.30 * pos_bonus)

        # Жесткий отсев совсем "хвостовых" окон
        # если окно упирается почти в самый конец — нам это не нужно
        if (total_len - end_idx) <= 1:
            effective_similarity *= 0.5

        if effective_similarity > best_effective_similarity:
            best_effective_similarity = effective_similarity
            best_raw_similarity = raw_similarity
            best_breakdown = breakdown
            best_notes = notes
            best_window_len = window_len
            best_window_start = start_idx
            best_window_end = end_idx
            best_position_bonus = pos_bonus

    if best_effective_similarity < 0 or best_breakdown is None:
        return None

    return {
        "similarity": round(best_effective_similarity, 2),
        "raw_similarity": round(best_raw_similarity, 2),
        "breakdown": best_breakdown,
        "notes": best_notes,
        "best_window_len": best_window_len,
        "best_window_start": best_window_start,
        "best_window_end": best_window_end,
        "position_bonus": round(best_position_bonus or 0.0, 4),
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
        explicit_symbol_mode = len(requested_symbols) > 0

        for coin in markets:
            symbol = str(coin.get("symbol", "")).lower()
            name = str(coin.get("name", ""))

            if requested_symbols and symbol not in requested_symbols:
                continue
            if symbol in excluded_symbols:
                continue

            market_cap = float(coin.get("market_cap") or 0)
            volume_24h = float(coin.get("total_volume") or 0)

            if not explicit_symbol_mode:
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
                    days=min(450, req.max_age_days if not explicit_symbol_mode else 450),
                )
            except httpx.HTTPStatusError:
                continue
            except httpx.HTTPError:
                continue
            except Exception:
                continue

            closes = coingecko_daily_closes(chart)
            if len(closes) < 30:
                continue

            asset_age_days = age_from_chart_days(chart)
            if asset_age_days is None:
                continue

            # Для рыночного скана фильтр возраста актива сохраняем.
            if not explicit_symbol_mode:
                if asset_age_days < req.min_age_days or asset_age_days > req.max_age_days:
                    continue

            best = find_best_window(closes, req.min_age_days, req.max_age_days)
            if best is None:
                continue

            notes = best["notes"][:]
            notes.append(f"Лучшее окно: {best['best_window_len']} дней.")
            notes.append(f"Старт лучшего окна: свеча #{best['best_window_start']}.")
            notes.append(f"Конец лучшего окна: свеча #{best['best_window_end']}.")
            notes.append(f"Raw similarity: {best['raw_similarity']}%.")
            notes.append(f"Position bonus: {best['position_bonus']}.")

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
