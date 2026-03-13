from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _minmax_scale(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _smooth(xs: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(xs) <= 2:
        return xs[:]
    out: List[float] = []
    r = window // 2
    for i in range(len(xs)):
        lo = max(0, i - r)
        hi = min(len(xs), i + r + 1)
        out.append(_mean(xs[lo:hi]))
    return out


def _local_peaks(xs: List[float], min_prominence: float = 0.02) -> List[int]:
    peaks: List[int] = []
    n = len(xs)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if xs[i] > xs[i - 1] and xs[i] > xs[i + 1]:
            left = xs[i] - xs[i - 1]
            right = xs[i] - xs[i + 1]
            if max(left, right) >= min_prominence:
                peaks.append(i)
    return peaks


def _resample(xs: List[float], target: int = 120) -> List[float]:
    if not xs:
        return []
    if len(xs) == target:
        return xs[:]
    if len(xs) == 1:
        return [xs[0]] * target
    out: List[float] = []
    for i in range(target):
        pos = i * (len(xs) - 1) / (target - 1)
        left = int(math.floor(pos))
        right = min(len(xs) - 1, left + 1)
        t = pos - left
        out.append(xs[left] * (1 - t) + xs[right] * t)
    return out


@dataclass
class PatternBreakdown:
    crown: float
    drop: float
    shelf: float
    right_spike: float
    reversion: float
    asymmetry: float
    template_shape: float


@dataclass
class PatternResult:
    similarity: float
    label: str
    stage: str
    breakdown: PatternBreakdown
    notes: List[str]


class CrownShelfRightSpikeScorer:
    """
    Паттерн:
    crown -> shelf -> right_spike -> reversion

    Сделан мягче:
    - crown не бинарная
    - partial matches сохраняются
    - SIREN / RIVER должны проходить как минимум как partial/structural
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def score(self, prices: List[float]) -> PatternResult:
        xs = _resample(prices, 120)
        xs = _smooth(xs, 5)
        xs = _minmax_scale(xs)

        crown_zone = xs[:42]       # 0–35%
        shelf_zone = xs[42:90]     # 35–75%
        spike_zone = xs[90:108]    # 75–90%
        revert_zone = xs[108:]     # 90–100%

        crown_score, crown_notes = self._score_crown(crown_zone)
        drop_score, drop_notes = self._score_drop(crown_zone, shelf_zone)
        shelf_score, shelf_notes = self._score_shelf(shelf_zone)
        spike_score, spike_notes = self._score_right_spike(shelf_zone, spike_zone)
        reversion_score, reversion_notes = self._score_reversion(shelf_zone, spike_zone, revert_zone)
        asymmetry_score, asymmetry_notes = self._score_asymmetry(crown_zone, spike_zone)
        template_shape_score, template_notes = self._score_template_shape(xs)

        similarity = (
            0.18 * crown_score +
            0.14 * drop_score +
            0.24 * shelf_score +
            0.20 * spike_score +
            0.14 * reversion_score +
            0.05 * asymmetry_score +
            0.05 * template_shape_score
        ) * 100.0

        backbone = (shelf_score + spike_score + reversion_score) / 3.0

        if crown_score < 0.15 and backbone >= 0.55:
            label = "weak-crown variant"
        elif similarity >= 65:
            label = "strong match"
        elif similarity >= 45:
            label = "partial match"
        else:
            label = "weak match"

        stage, stage_notes = self._classify_stage(
            xs=xs,
            shelf_zone=shelf_zone,
            spike_zone=spike_zone,
            revert_zone=revert_zone,
            similarity=similarity,
            shelf_score=shelf_score,
            spike_score=spike_score,
            reversion_score=reversion_score,
        )

        notes = (
            crown_notes
            + drop_notes
            + shelf_notes
            + spike_notes
            + reversion_notes
            + asymmetry_notes
            + template_notes
            + stage_notes
        )

        return PatternResult(
            similarity=round(similarity, 2),
            label=label,
            stage=stage,
            breakdown=PatternBreakdown(
                crown=round(crown_score, 4),
                drop=round(drop_score, 4),
                shelf=round(shelf_score, 4),
                right_spike=round(spike_score, 4),
                reversion=round(reversion_score, 4),
                asymmetry=round(asymmetry_score, 4),
                template_shape=round(template_shape_score, 4),
            ),
            notes=notes,
        )

    def _score_crown(self, crown_zone: List[float]) -> Tuple[float, List[str]]:
        peaks = _local_peaks(crown_zone, min_prominence=0.02)
        top = max(crown_zone) if crown_zone else 0.0
        avg = _mean(crown_zone)
        vol = _stdev(crown_zone)

        peak_count_score = min(1.0, len(peaks) / 3.0)
        crest_score = max(0.0, min(1.0, (top - avg) / 0.20))
        texture_score = max(0.0, min(1.0, vol / 0.11))

        score = 0.45 * peak_count_score + 0.35 * crest_score + 0.20 * texture_score

        notes: List[str] = []
        if score >= 0.65:
            notes.append("Левая корона читается достаточно хорошо.")
        elif score >= 0.35:
            notes.append("Левая корона присутствует, но выражена мягко.")
        else:
            notes.append("Левая корона выражена слабо.")
        return score, notes

    def _score_drop(self, crown_zone: List[float], shelf_zone: List[float]) -> Tuple[float, List[str]]:
        if not crown_zone or not shelf_zone:
            return 0.0, ["Падение из короны в полку не удалось оценить."]
        crown_high = max(crown_zone)
        shelf_mid = _mean(shelf_zone)
        drop = max(0.0, crown_high - shelf_mid)
        score = max(0.0, min(1.0, drop / 0.35))

        if score >= 0.55:
            return score, ["Снижение из левой зоны в полку читается хорошо."]
        if score >= 0.25:
            return score, ["Снижение из левой зоны в полку умеренное."]
        return score, ["Падение из короны в полку выражено слабо."]

    def _score_shelf(self, shelf_zone: List[float]) -> Tuple[float, List[str]]:
        if not shelf_zone:
            return 0.0, ["Полку не удалось оценить."]
        vol = _stdev(shelf_zone)
        slope = abs(_mean(shelf_zone[-8:]) - _mean(shelf_zone[:8])) if len(shelf_zone) >= 16 else 0.0

        flatness = max(0.0, 1.0 - min(1.0, vol / 0.09))
        levelness = max(0.0, 1.0 - min(1.0, slope / 0.12))

        score = 0.65 * flatness + 0.35 * levelness

        if score >= 0.70:
            return score, ["Есть длинная и достаточно ровная средняя полка."]
        if score >= 0.40:
            return score, ["Полка присутствует, но не совсем плоская."]
        return score, ["Средняя полка слабая или слишком шумная."]

    def _score_right_spike(self, shelf_zone: List[float], spike_zone: List[float]) -> Tuple[float, List[str]]:
        if not shelf_zone or not spike_zone:
            return 0.0, ["Правый шпиль не удалось оценить."]
        shelf_mid = _mean(shelf_zone)
        spike_top = max(spike_zone)
        spike_gain = max(0.0, spike_top - shelf_mid)
        spike_narrowness = max(0.0, 1.0 - min(1.0, _stdev(spike_zone) / 0.18))
        raw_height = max(0.0, min(1.0, spike_gain / 0.30))

        score = 0.75 * raw_height + 0.25 * spike_narrowness

        if score >= 0.75:
            return score, ["Правый шпиль выражен хорошо и визуально отделен от полки."]
        if score >= 0.45:
            return score, ["Правый выступ есть, но он умеренный."]
        return score, ["Правый выступ слабый или размазан."]

    def _score_reversion(
        self,
        shelf_zone: List[float],
        spike_zone: List[float],
        revert_zone: List[float],
    ) -> Tuple[float, List[str]]:
        if not shelf_zone or not spike_zone or not revert_zone:
            return 0.0, ["Возврат в полку не удалось оценить."]

        shelf_mid = _mean(shelf_zone)
        revert_mid = _mean(revert_zone)
        dist_to_shelf = abs(revert_mid - shelf_mid)
        shelf_rejoin = max(0.0, 1.0 - min(1.0, dist_to_shelf / 0.12))

        spike_top = max(spike_zone)
        spike_excess = max(0.0, spike_top - shelf_mid)
        decay = max(0.0, spike_top - revert_mid)
        decay_score = 0.0 if spike_excess <= 0 else max(0.0, min(1.0, decay / max(0.12, spike_excess)))

        score = 0.65 * shelf_rejoin + 0.35 * decay_score

        if score >= 0.70:
            return score, ["После шпиля есть возврат в боковик."]
        if score >= 0.40:
            return score, ["После шпиля возврат в полку умеренный."]
        return score, ["После шпиля возврат в полку выражен слабо."]

    def _score_asymmetry(self, crown_zone: List[float], spike_zone: List[float]) -> Tuple[float, List[str]]:
        if not crown_zone or not spike_zone:
            return 0.0, ["Асимметрию формы не удалось оценить."]
        left_height = max(crown_zone) - min(crown_zone)
        right_height = max(spike_zone) - min(spike_zone)

        ratio = right_height / max(1e-9, left_height)
        score = 1.0 - min(1.0, abs(ratio - 1.0) / 1.5)

        if score >= 0.60:
            return score, ["Левая и правая части формы соразмерны."]
        return score, ["Баланс левой и правой части формы средний."]

    def _score_template_shape(self, xs: List[float]) -> Tuple[float, List[str]]:
        if not xs:
            return 0.0, ["Силуэт не удалось оценить."]

        crown = _mean(xs[:30])
        shelf = _mean(xs[45:85])
        spike = max(xs[92:108]) if len(xs) >= 108 else max(xs)
        revert = _mean(xs[108:]) if len(xs) > 108 else xs[-1]

        cond1 = max(0.0, min(1.0, (crown - shelf + 0.08) / 0.25))
        cond2 = max(0.0, min(1.0, (spike - shelf) / 0.30))
        cond3 = max(0.0, 1.0 - min(1.0, abs(revert - shelf) / 0.12))

        score = 0.35 * cond1 + 0.35 * cond2 + 0.30 * cond3

        if score >= 0.65:
            return score, ["Общий силуэт хорошо похож на эталон."]
        if score >= 0.40:
            return score, ["Общий силуэт умеренно похож на эталон."]
        return score, ["Силуэт заметно отклоняется от эталона."]

    def _classify_stage(
        self,
        xs: List[float],
        shelf_zone: List[float],
        spike_zone: List[float],
        revert_zone: List[float],
        similarity: float,
        shelf_score: float,
        spike_score: float,
        reversion_score: float,
    ) -> Tuple[str, List[str]]:
        notes: List[str] = []

        shelf_mid = _mean(shelf_zone) if shelf_zone else 0.0
        spike_top = max(spike_zone) if spike_zone else shelf_mid
        revert_mid = _mean(revert_zone) if revert_zone else shelf_mid
        last_tail = _mean(xs[-8:]) if len(xs) >= 8 else xs[-1]

        tail_distance = abs(last_tail - shelf_mid)

        if spike_score >= 0.45 and reversion_score >= 0.55 and tail_distance <= 0.10:
            notes.append("Стадия: паттерн уже в основном отыгран.")
            return "completed", notes

        if spike_score >= 0.45 and similarity >= 45 and tail_distance > 0.10:
            notes.append("Стадия: паттерн еще активен и потенциально торгуем.")
            return "active", notes

        if shelf_score >= 0.45 and spike_score < 0.30:
            notes.append("Стадия: паттерн еще формируется.")
            return "forming", notes

        if reversion_score >= 0.50:
            notes.append("Стадия ближе к завершенной.")
            return "completed", notes

        notes.append("Стадия ближе к активной.")
        return "active", notes


def score_crown_shelf_right_spike(close: Iterable[float]):
    scorer = CrownShelfRightSpikeScorer(debug=False)
    result = scorer.score(list(close))
    return result
