
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from statistics import mean

from .models import BestWindow, MatchBreakdown


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class ReferenceTemplate:
    name: str
    target_total_days: int
    target_hold_days: int
    total_days_tolerance: int
    hold_days_tolerance: int


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


REFERENCE_WINDOWS: dict[str, tuple[date, date]] = {
    "river": (date(2025, 9, 22), date(2025, 12, 30)),
    "siren-2": (date(2026, 2, 6), date(2026, 3, 21)),
}

REFERENCE_TEMPLATES = [
    ReferenceTemplate(name="river", target_total_days=100, target_hold_days=20, total_days_tolerance=30, hold_days_tolerance=10),
    ReferenceTemplate(name="siren", target_total_days=44, target_hold_days=34, total_days_tolerance=16, hold_days_tolerance=12),
]


def _find_top_cluster(norm: list[float], ath_idx: int, top_threshold: float, max_gap: int = 1) -> tuple[int, int]:
    start = ath_idx
    gaps = 0
    i = ath_idx - 1
    while i >= 0:
        if norm[i] >= top_threshold:
            start = i
            gaps = 0
        else:
            gaps += 1
            if gaps > max_gap:
                break
        i -= 1

    end = ath_idx
    gaps = 0
    i = ath_idx + 1
    while i < len(norm):
        if norm[i] >= top_threshold:
            end = i
            gaps = 0
        else:
            gaps += 1
            if gaps > max_gap:
                break
        i += 1
    return start, end


def _window_from_dates(timestamps_ms: list[int], start: date, end: date) -> tuple[int, int] | None:
    if not timestamps_ms:
        return None
    start_idx = None
    end_idx = None
    for idx, ts in enumerate(timestamps_ms):
        d = date.fromtimestamp(ts / 1000)
        if start_idx is None and d >= start:
            start_idx = idx
        if d <= end:
            end_idx = idx
    if start_idx is None or end_idx is None or end_idx - start_idx + 1 < 20:
        return None
    return start_idx, end_idx


def _iter_candidate_windows(prices: list[float], timestamps_ms: list[int] | None, coin_id: str | None) -> list[tuple[int, int]]:
    n = len(prices)
    windows: list[tuple[int, int]] = []

    if coin_id and timestamps_ms:
        ref = REFERENCE_WINDOWS.get(coin_id)
        if ref is not None:
            explicit = _window_from_dates(timestamps_ms, ref[0], ref[1])
            if explicit is not None:
                windows.append(explicit)

    for template in REFERENCE_TEMPLATES:
        lo = max(30, template.target_total_days - template.total_days_tolerance)
        hi = min(n, template.target_total_days + template.total_days_tolerance)
        for win_len in range(lo, hi + 1, 4):
            if win_len > n:
                continue
            for start in range(0, n - win_len + 1, 4):
                windows.append((start, start + win_len - 1))

    if not windows:
        windows.append((0, n - 1))

    # de-duplicate while preserving order
    dedup: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in windows:
        if item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def _score_window(window_prices: list[float], template: ReferenceTemplate) -> tuple[float, dict[str, float], str, list[str]]:
    n = len(window_prices)
    pmin, pmax = min(window_prices), max(window_prices)
    rng = max(pmax - pmin, 1e-9)
    norm = [(p - pmin) / rng for p in window_prices]
    ath_idx = max(range(n), key=lambda i: norm[i])
    top_threshold = 0.90
    cluster_start, cluster_end = _find_top_cluster(norm, ath_idx=ath_idx, top_threshold=top_threshold, max_gap=1)
    hold_days = cluster_end - cluster_start + 1

    # dump starts after the crown cluster; look for first persistent break lower
    crown_floor = min(norm[cluster_start:cluster_end + 1])
    dump_start = None
    break_level = max(0.50, crown_floor - 0.12)
    for i in range(cluster_end + 1, n):
        if norm[i] < break_level:
            lookahead = norm[i + 1:min(n, i + 4)]
            if not lookahead or max(lookahead) < top_threshold:
                dump_start = i
                break
    if dump_start is None:
        dump_start = n - 1

    post_dump_low = min(norm[dump_start:]) if dump_start < n else norm[-1]
    dump_depth = max(0.0, crown_floor - post_dump_low)
    shelf_zone = norm[dump_start:] if dump_start < n else [norm[-1]]
    shelf_mean = mean(shelf_zone) if shelf_zone else 0.0
    shelf_std = math.sqrt(mean([(x - shelf_mean) ** 2 for x in shelf_zone])) if shelf_zone else 1.0

    ath_touch_score = 1.0  # by definition cluster contains the ATH
    hold_score = max(
        0.0,
        1.0 - abs(hold_days - template.target_hold_days) / max(template.hold_days_tolerance, 1),
    )
    total_days_score = max(
        0.0,
        1.0 - abs(n - template.target_total_days) / max(template.total_days_tolerance, 1),
    )
    pre_ath_presence = ath_idx / max(n - 1, 1)
    pre_ath_score = _clamp((pre_ath_presence - 0.12) / 0.28)  # avoid ATH at the first few bars
    post_cluster_room = (n - cluster_end - 1) / max(n - 1, 1)
    post_cluster_score = _clamp((post_cluster_room - 0.15) / 0.35)
    drop_score = _clamp(dump_depth / 0.32)
    shelf_score = _clamp((1.0 - shelf_std / 0.18) * (1.0 - abs(shelf_mean - 0.35) / 0.40))
    right_spike_score = 0.55  # kept neutral in reference mode; user focus is on crown timing/holding
    reversion_score = _clamp(1.0 - abs(shelf_mean - 0.35) / 0.25)
    asymmetry_score = _clamp(1.0 - abs((cluster_start + 1) - max(1, n - dump_start)) / max(n * 0.45, 1))
    template_shape = _clamp(
        0.26 * ath_touch_score
        + 0.24 * hold_score
        + 0.14 * total_days_score
        + 0.10 * pre_ath_score
        + 0.10 * post_cluster_score
        + 0.10 * drop_score
        + 0.06 * shelf_score
    )
    raw = _clamp(
        0.28 * ath_touch_score
        + 0.26 * hold_score
        + 0.18 * total_days_score
        + 0.10 * drop_score
        + 0.08 * shelf_score
        + 0.05 * pre_ath_score
        + 0.05 * post_cluster_score
    )

    similarity = round(_clamp(0.60 * template_shape + 0.40 * raw) * 100, 2)
    notes = [
        f"Корона касается локального ATH и удерживает верхнюю зону около {hold_days} дневных свечей.",
        f"Длина всего окна около {n} свечей — ближе к эталону {template.name.upper()}.",
    ]
    if hold_score >= 0.75:
        notes.append("Длительность удержания короны близка к эталону.")
    else:
        notes.append("Длительность удержания короны близка лишь частично.")
    if drop_score >= 0.60:
        notes.append("После удержания начинается читаемый слив из короны.")
    else:
        notes.append("После удержания слив выражен слабо или запаздывает.")
    if shelf_score >= 0.55:
        notes.append("После слива есть относительно читаемая база/полка.")
    else:
        notes.append("Полка после слива шумная или неустойчивая.")

    stage = "completed" if dump_start < n - 1 else "active"
    metrics = {
        "crown": ath_touch_score,
        "drop": drop_score,
        "shelf": shelf_score,
        "right_spike": right_spike_score,
        "reversion": reversion_score,
        "asymmetry": asymmetry_score,
        "template_shape": template_shape,
        "hold_days": float(hold_days),
    }
    return similarity, metrics, stage, notes


def score_crown_shelf_right_spike(
    prices: list[float],
    timestamps_ms: list[int] | None = None,
    coin_id: str | None = None,
) -> PatternScore:
    if len(prices) < 30:
        breakdown = MatchBreakdown(
            crown=0.0, drop=0.0, shelf=0.0, right_spike=0.0, reversion=0.0, asymmetry=0.0, template_shape=0.0
        )
        return PatternScore(
            similarity=0.0,
            raw_similarity=0.0,
            structural_score=0.0,
            stage="unknown",
            label="weak match",
            breakdown=breakdown,
            best_window=BestWindow(
                start_idx=0,
                end_idx=max(0, len(prices) - 1),
                length_days=len(prices),
                best_age_days=len(prices),
                candidate_windows_count=1,
            ),
            notes=["Недостаточно исторических данных для сравнения с эталонами RIVER/SIREN."],
        )

    candidate_windows = _iter_candidate_windows(prices, timestamps_ms, coin_id)
    best = None

    for start_idx, end_idx in candidate_windows:
        window_prices = prices[start_idx:end_idx + 1]
        for template in REFERENCE_TEMPLATES:
            similarity, metrics, stage, notes = _score_window(window_prices, template)
            candidate = {
                "similarity": similarity,
                "metrics": metrics,
                "stage": stage,
                "notes": notes + [f"Режим сравнения: {template.name.upper()}-подобный."],
                "template_name": template.name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length_days": len(window_prices),
            }
            if best is None or candidate["similarity"] > best["similarity"]:
                best = candidate

    assert best is not None

    structural_score = round(best["metrics"]["template_shape"] * 100, 2)
    raw_similarity = round(
        (
            best["metrics"]["crown"] * 0.28
            + best["metrics"]["drop"] * 0.18
            + best["metrics"]["shelf"] * 0.12
            + best["metrics"]["right_spike"] * 0.05
            + best["metrics"]["reversion"] * 0.07
            + best["metrics"]["asymmetry"] * 0.05
            + best["metrics"]["template_shape"] * 0.25
        ) * 100,
        2,
    )

    similarity = best["similarity"]
    label = "strong match" if similarity >= 70 else "partial match" if similarity >= 58 else "weak match"

    breakdown = MatchBreakdown(
        crown=round(best["metrics"]["crown"], 4),
        drop=round(best["metrics"]["drop"], 4),
        shelf=round(best["metrics"]["shelf"], 4),
        right_spike=round(best["metrics"]["right_spike"], 4),
        reversion=round(best["metrics"]["reversion"], 4),
        asymmetry=round(best["metrics"]["asymmetry"], 4),
        template_shape=round(best["metrics"]["template_shape"], 4),
    )
    best_window = BestWindow(
        start_idx=best["start_idx"],
        end_idx=best["end_idx"],
        length_days=best["length_days"],
        best_age_days=best["length_days"],
        candidate_windows_count=len(candidate_windows),
    )

    return PatternScore(
        similarity=similarity,
        raw_similarity=raw_similarity,
        structural_score=structural_score,
        stage=best["stage"],
        label=label,
        breakdown=breakdown,
        best_window=best_window,
        notes=best["notes"],
    )
