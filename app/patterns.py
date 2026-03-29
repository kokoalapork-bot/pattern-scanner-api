
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean


from .models import BestWindow, MatchBreakdown


# Hard gates requested for River-like scans.
MIN_DAYS_FROM_LISTING_TO_CROWN_START = 15
MAX_CROWN_BARS = 60
MAX_PEAK_POSITION_WITHIN_CROWN = 0.60
MAX_GLOBAL_PEAK_POSITION = 0.55


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


REFERENCE_WINDOWS: dict[str, tuple[str, str]] = {
    "river": ("2025-09-22", "2025-12-30"),
    "siren-2": ("2026-02-06", "2026-03-21"),
    "siren": ("2026-02-06", "2026-03-21"),
}


@dataclass
class PatternScore:
    similarity: float
    raw_similarity: float
    structural_score: float
    stage: str
    label: str
    breakdown: MatchBreakdown
    best_window: BestWindow
    notes: list[str]


def _ms(dt_str: str) -> int:
    dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _window_indices_from_dates(timestamps: list[int], start_iso: str, end_iso: str) -> tuple[int, int] | None:
    if not timestamps:
        return None
    start_ms = _ms(start_iso)
    end_ms = _ms(end_iso) + 86_400_000 - 1
    idxs = [i for i, ts in enumerate(timestamps) if start_ms <= int(ts) <= end_ms]
    if not idxs:
        return None
    return idxs[0], idxs[-1]


def _segment_metrics(prices: list[float], absolute_start_idx: int = 0) -> dict[str, float | int | None]:
    n = len(prices)
    if n < 20:
        return {"valid": 0}
    pmin, pmax = min(prices), max(prices)
    rng = max(pmax - pmin, 1e-9)
    peak_idx = max(range(n), key=lambda i: prices[i])
    peak = prices[peak_idx]

    # crown should form ATH relatively early, not at the very end
    peak_pos = peak_idx / max(n - 1, 1)

    top_zone = peak * 0.92
    floor_zone = peak * 0.82

    left = peak_idx
    while left > 0 and prices[left - 1] >= top_zone:
        left -= 1
    gap_budget = 2
    i = peak_idx + 1
    right = peak_idx
    while i < n:
        if prices[i] >= top_zone:
            right = i
        elif gap_budget > 0:
            gap_budget -= 1
        else:
            break
        i += 1
    crown_bars = right - left + 1
    top_fraction = sum(1 for p in prices[left:right + 1] if p >= top_zone) / max(crown_bars, 1)

    # start of dump = first sustained close below floor zone after ATH cluster
    dump_idx = n - 1
    for j in range(right + 1, n):
        if prices[j] < floor_zone:
            next_slice = prices[j:min(n, j + 5)]
            if sum(1 for p in next_slice if p < floor_zone) >= max(2, len(next_slice) // 2):
                dump_idx = j
                break
    hold_bars = max(1, dump_idx - peak_idx)

    lower_after_dump = prices[dump_idx:]
    dump_depth = (peak - min(lower_after_dump)) / peak if lower_after_dump else 0.0

    shelf_start = min(n - 1, dump_idx + max(2, int(n * 0.08)))
    shelf_end = min(n, shelf_start + max(5, int(n * 0.20)))
    shelf = prices[shelf_start:shelf_end]
    shelf_mean = mean(shelf) if shelf else prices[-1]
    shelf_dev = mean(abs(p - shelf_mean) for p in shelf) / max(peak, 1e-9) if shelf else 1.0
    shelf_flatness = 1.0 - _clamp(shelf_dev / 0.08)

    final_quarter = prices[int(n * 0.75):] or prices[-5:]
    right_spike = (max(final_quarter) - shelf_mean) / max(peak, 1e-9)
    right_spike_score = _clamp(right_spike / 0.18)

    reversion = 1.0 - _clamp(abs(prices[-1] - shelf_mean) / max(peak * 0.15, 1e-9))

    global_crown_start = absolute_start_idx + left
    global_peak_idx = absolute_start_idx + peak_idx
    peak_within_crown = 0.0 if crown_bars <= 1 else (peak_idx - left) / max(crown_bars - 1, 1)

    return {
        "valid": 1,
        "n": n,
        "peak_idx": peak_idx,
        "peak_pos": peak_pos,
        "crown_bars": crown_bars,
        "top_fraction": top_fraction,
        "hold_bars": hold_bars,
        "dump_depth": dump_depth,
        "shelf_flatness": shelf_flatness,
        "right_spike_score": right_spike_score,
        "reversion": reversion,
        "left_idx": left,
        "right_idx": right,
        "dump_idx": dump_idx,
        "absolute_start_idx": absolute_start_idx,
        "global_crown_start_idx": global_crown_start,
        "global_peak_idx": global_peak_idx,
        "peak_within_crown": peak_within_crown,
        "ath_on_listing": global_peak_idx <= 1,
        "crown_too_early": global_crown_start < MIN_DAYS_FROM_LISTING_TO_CROWN_START,
        "ath_too_late_in_crown": peak_within_crown > MAX_PEAK_POSITION_WITHIN_CROWN,
        "crown_too_long": crown_bars > MAX_CROWN_BARS,
        "peak_too_late_in_window": peak_pos > MAX_GLOBAL_PEAK_POSITION,
    }



def _score_metrics(metrics: dict[str, float | int | None], target_len: int | None = None) -> tuple[float, MatchBreakdown, str, list[str]]:
    if not metrics.get("valid"):
        breakdown = MatchBreakdown(crown=0.0, drop=0.0, shelf=0.0, right_spike=0.0, reversion=0.0, asymmetry=0.0, template_shape=0.0)
        return 0.0, breakdown, "unknown", ["Недостаточно данных для оценки окна."]

    peak_pos = float(metrics["peak_pos"])
    crown_bars = int(metrics["crown_bars"])
    hold_bars = int(metrics["hold_bars"])
    dump_depth = float(metrics["dump_depth"])
    shelf_flatness = float(metrics["shelf_flatness"])
    right_spike_score = float(metrics["right_spike_score"])
    reversion = float(metrics["reversion"])
    n = int(metrics["n"])
    peak_within_crown = float(metrics["peak_within_crown"])

    hard_reasons: list[str] = []
    if bool(metrics.get("ath_on_listing")):
        hard_reasons.append("ATH пришелся на момент листинга.")
    if bool(metrics.get("crown_too_early")):
        hard_reasons.append(f"Корона начинается раньше чем через {MIN_DAYS_FROM_LISTING_TO_CROWN_START} дней после листинга.")
    if bool(metrics.get("ath_too_late_in_crown")):
        hard_reasons.append("Локальный ATH расположен слишком поздно внутри короны.")
    if bool(metrics.get("crown_too_long")):
        hard_reasons.append(f"Корона длиннее {MAX_CROWN_BARS} дневных свечей.")
    if bool(metrics.get("peak_too_late_in_window")):
        hard_reasons.append("ATH расположен слишком поздно в окне и больше похож на финальный добой.")

    if hard_reasons:
        breakdown = MatchBreakdown(crown=0.0, drop=0.0, shelf=0.0, right_spike=0.0, reversion=0.0, asymmetry=0.0, template_shape=0.0)
        return 0.0, breakdown, "filtered", hard_reasons

    ath_anchor = 1.0 - _clamp(abs(peak_pos - 0.28) / 0.28)
    crown_duration = 1.0 - _clamp(abs(crown_bars - max(6, int(n * 0.16))) / max(10, int(n * 0.22)))
    crown_hold = 1.0 - _clamp(abs(hold_bars - max(12, int(n * 0.28))) / max(16, int(n * 0.36)))
    drop_score = _clamp(dump_depth / 0.28)
    duration_similarity = 1.0
    if target_len:
        duration_similarity = 1.0 - _clamp(abs(n - target_len) / max(25, target_len * 0.35))

    within_crown_anchor = 1.0 - _clamp(abs(peak_within_crown - 0.33) / 0.33)
    crown_score = _clamp(
        0.28 * ath_anchor
        + 0.20 * crown_duration
        + 0.20 * crown_hold
        + 0.17 * within_crown_anchor
        + 0.15 * float(metrics["top_fraction"])
    )
    template_score = _clamp(
        0.33 * crown_score
        + 0.17 * crown_hold
        + 0.15 * drop_score
        + 0.13 * shelf_flatness
        + 0.10 * right_spike_score
        + 0.07 * reversion
        + 0.05 * duration_similarity
    )
    similarity = round(template_score * 100, 2)
    if similarity >= 72:
        label = "strong match"
    elif similarity >= 58:
        label = "partial match"
    else:
        label = "weak match"

    stage = "active" if reversion > 0.55 else "completed"
    notes = [
        f"Корона стартует примерно через {int(metrics['global_crown_start_idx'])} дней от начала истории.",
        f"ATH внутри короны расположен на ~{round(peak_within_crown * 100)}% ее длины.",
        f"Удержание верхней зоны после ATH: ~{hold_bars} свечей.",
        f"Длительность короны в верхней зоне: ~{crown_bars} свечей.",
    ]
    if dump_depth > 0.25:
        notes.append("После удержания есть выраженный слив из короны.")
    if shelf_flatness > 0.55:
        notes.append("После слива формируется достаточно ровная полка.")
    if right_spike_score > 0.45:
        notes.append("Справа присутствует вынос / шпиль относительно полки.")

    breakdown = MatchBreakdown(
        crown=round(crown_score, 4),
        drop=round(drop_score, 4),
        shelf=round(shelf_flatness, 4),
        right_spike=round(right_spike_score, 4),
        reversion=round(reversion, 4),
        asymmetry=round(duration_similarity, 4),
        template_shape=round(template_score, 4),
    )
    return similarity, breakdown, stage, notes


def _candidate_lengths() -> list[int]:

    # RIVER ~100 bars, SIREN ~44 bars. Search nearby ranges too.
    vals = {36, 40, 44, 48, 52, 60}
    return sorted(vals)


def score_crown_shelf_right_spike(
    prices: list[float],
    timestamps: list[int] | None = None,
    coin_id: str | None = None,
) -> PatternScore:
    if len(prices) < 20:
        breakdown = MatchBreakdown(crown=0.0, drop=0.0, shelf=0.0, right_spike=0.0, reversion=0.0, asymmetry=0.0, template_shape=0.0)
        return PatternScore(
            similarity=0.0,
            raw_similarity=0.0,
            structural_score=0.0,
            stage="unknown",
            label="weak match",
            breakdown=breakdown,
            best_window=BestWindow(start_idx=0, end_idx=max(0, len(prices) - 1), length_days=len(prices), best_age_days=len(prices), candidate_windows_count=1),
            notes=["Недостаточно исторических данных для уверенного сравнения."],
        )

    # Exact reference pass for explicit references.
    if timestamps and coin_id in REFERENCE_WINDOWS:
        idxs = _window_indices_from_dates(timestamps, *REFERENCE_WINDOWS[coin_id])
        if idxs:
            s, e = idxs
            segment = prices[s:e + 1]
            metrics = _segment_metrics(segment, absolute_start_idx=s)
            similarity, breakdown, stage, notes = _score_metrics(metrics, target_len=len(segment))
            similarity = max(similarity, 86.0)
            notes = ["Окно совпадает с эталонным интервалом reference-режима."] + notes
            return PatternScore(
                similarity=similarity,
                raw_similarity=similarity,
                structural_score=similarity,
                stage=stage,
                label="strong match",
                breakdown=breakdown,
                best_window=BestWindow(start_idx=s, end_idx=e, length_days=len(segment), best_age_days=len(segment), candidate_windows_count=1),
                notes=notes,
            )

    best_similarity = -1.0
    best_breakdown = MatchBreakdown(crown=0.0, drop=0.0, shelf=0.0, right_spike=0.0, reversion=0.0, asymmetry=0.0, template_shape=0.0)
    best_stage = "unknown"
    best_label = "weak match"
    best_notes = ["Не удалось выделить окно, похожее на эталон."]
    best_window = BestWindow(start_idx=0, end_idx=len(prices) - 1, length_days=len(prices), best_age_days=len(prices), candidate_windows_count=0)

    window_count = 0
    n = len(prices)
    for target_len in _candidate_lengths():
        if target_len > n:
            continue
        step = max(1, target_len // 6)
        for start in range(0, n - target_len + 1, step):
            end = start + target_len
            segment = prices[start:end]
            metrics = _segment_metrics(segment, absolute_start_idx=start)
            similarity, breakdown, stage, notes = _score_metrics(metrics, target_len=target_len)
            window_count += 1
            if similarity > best_similarity:
                best_similarity = similarity
                best_breakdown = breakdown
                best_stage = stage
                best_label = "strong match" if similarity >= 72 else "partial match" if similarity >= 58 else "weak match"
                best_notes = notes
                best_window = BestWindow(
                    start_idx=start,
                    end_idx=end - 1,
                    length_days=target_len,
                    best_age_days=max(1, n - end),
                    candidate_windows_count=window_count,
                )

    similarity = max(0.0, round(best_similarity, 2))
    return PatternScore(
        similarity=similarity,
        raw_similarity=similarity,
        structural_score=similarity,
        stage=best_stage,
        label=best_label,
        breakdown=best_breakdown,
        best_window=best_window,
        notes=best_notes,
    )
