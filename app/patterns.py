
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from .models import BestWindow, MatchBreakdown


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


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


def score_crown_shelf_right_spike(prices: list[float]) -> PatternScore:
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
            best_window=BestWindow(start_idx=0, end_idx=max(0, len(prices) - 1), length_days=len(prices), best_age_days=len(prices), candidate_windows_count=1),
            notes=["Недостаточно исторических данных для уверенного сравнения."],
        )

    n = len(prices)
    pmin, pmax = min(prices), max(prices)
    rng = max(pmax - pmin, 1e-9)
    norm = [(p - pmin) / rng for p in prices]

    crown_end = max(3, int(n * 0.22))
    shelf_start = max(crown_end + 1, int(n * 0.35))
    shelf_end = max(shelf_start + 3, int(n * 0.78))
    spike_start = max(shelf_end, int(n * 0.80))
    spike_end = n - 1

    crown_zone = norm[: crown_end + 1]
    shelf_zone = norm[shelf_start:shelf_end]
    spike_zone = norm[spike_start: spike_end + 1]

    crown_peak = max(crown_zone)
    crown_peak_idx = crown_zone.index(crown_peak)
    crown_score = _clamp(crown_peak / 0.85)

    after_crown = norm[crown_peak_idx:]
    dump_low = min(after_crown[: max(3, len(after_crown) // 2)])
    drop_ratio = crown_peak - dump_low
    drop_score = _clamp(drop_ratio / 0.45)

    shelf_mean = mean(shelf_zone) if shelf_zone else 0.0
    shelf_std = math.sqrt(mean([(x - shelf_mean) ** 2 for x in shelf_zone])) if shelf_zone else 1.0
    shelf_flatness = 1.0 - _clamp(shelf_std / 0.18)
    shelf_level_ok = 1.0 - _clamp(abs(shelf_mean - 0.35) / 0.35)
    shelf_score = _clamp(0.55 * shelf_flatness + 0.45 * shelf_level_ok)

    spike_peak = max(spike_zone) if spike_zone else 0.0
    spike_score = _clamp((spike_peak - shelf_mean) / 0.40)

    reversion_tail = norm[max(spike_start, spike_start + len(spike_zone)//2):]
    tail_mean = mean(reversion_tail) if reversion_tail else 0.0
    reversion_score = 1.0 - _clamp(abs(tail_mean - shelf_mean) / 0.25)

    left_len = crown_end + 1
    right_len = spike_end - spike_start + 1
    asymmetry_score = 1.0 - _clamp(abs(left_len - right_len) / max(n * 0.35, 1))

    template_score = _clamp(
        0.22 * crown_score
        + 0.18 * drop_score
        + 0.22 * shelf_score
        + 0.18 * spike_score
        + 0.12 * reversion_score
        + 0.08 * asymmetry_score
    )

    similarity = round(template_score * 100, 2)
    label = "strong match" if similarity >= 68 else "partial match" if similarity >= 55 else "weak match"
    stage = "completed" if reversion_score >= 0.6 else "active"

    notes = []
    notes.append("Левая корона читается достаточно хорошо." if crown_score >= 0.65 else "Левая корона присутствует, но выражена мягко.")
    notes.append("Снижение из левой зоны в полку читается хорошо." if drop_score >= 0.65 else "Падение из короны в полку выражено слабо.")
    notes.append("Есть длинная и достаточно ровная средняя полка." if shelf_score >= 0.65 else "Средняя полка слабая или слишком шумная.")
    notes.append("Правый шпиль выражен хорошо и визуально отделен от полки." if spike_score >= 0.65 else "Правый выступ есть, но он умеренный.")
    notes.append("После шпиля есть возврат в боковик." if reversion_score >= 0.60 else "После шпиля возврат в полку выражен слабо.")
    notes.append("Левая и правая части формы соразмерны." if asymmetry_score >= 0.60 else "Баланс левой и правой части формы средний.")
    notes.append("Общий силуэт хорошо похож на эталон." if template_score >= 0.68 else "Общий силуэт умеренно похож на эталон.")
    notes.append("Стадия: паттерн уже в основном отыгран." if stage == "completed" else "Стадия: паттерн еще активен и потенциально торгуем.")

    return PatternScore(
        similarity=similarity,
        raw_similarity=similarity,
        structural_score=similarity,
        stage=stage,
        label=label,
        breakdown=MatchBreakdown(
            crown=round(crown_score, 4),
            drop=round(drop_score, 4),
            shelf=round(shelf_score, 4),
            right_spike=round(spike_score, 4),
            reversion=round(reversion_score, 4),
            asymmetry=round(asymmetry_score, 4),
            template_shape=round(template_score, 4),
        ),
        best_window=BestWindow(
            start_idx=0,
            end_idx=n - 1,
            length_days=n,
            best_age_days=n,
            candidate_windows_count=max(1, n - 29),
        ),
        notes=notes,
    )
