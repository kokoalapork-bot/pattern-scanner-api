import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

TARGET_LEN = 128


@dataclass
class PatternBreakdown:
    crown: float
    drop: float
    shelf: float
    right_spike: float
    reversion: float
    asymmetry: float
    template_shape: float


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalize_series(close: Iterable[float], target_len: int = TARGET_LEN) -> np.ndarray:
    arr = np.asarray(list(close), dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 30:
        raise ValueError("Need at least 30 data points to score a pattern")

    x = np.arange(len(arr))
    f = interp1d(x, arr, kind="linear")
    x_new = np.linspace(0, len(arr) - 1, target_len)
    y = f(x_new)

    win = 11 if target_len >= 11 else max(5, target_len // 2 * 2 + 1)
    if win >= 5 and win < len(y):
        y = savgol_filter(y, win, 3 if win >= 7 else 2)

    ymin, ymax = float(np.min(y)), float(np.max(y))
    if math.isclose(ymax, ymin):
        return np.zeros_like(y)

    return (y - ymin) / (ymax - ymin)


def crown_score(y: np.ndarray) -> float:
    left = y[:52]
    peaks, _ = find_peaks(left, prominence=0.04, distance=4)

    if len(peaks) < 3:
        return 0.0

    peak_vals = left[peaks]
    count_score = 1.0 if 3 <= len(peaks) <= 6 else clamp01(1 - abs(len(peaks) - 4) * 0.18)

    mean_val = float(np.mean(peak_vals))
    std_val = float(np.std(peak_vals))
    consistency = clamp01(1.0 - (std_val / (mean_val + 1e-9)))

    top_zone_score = float(np.mean(peak_vals > 0.72))
    uneven_top_bonus = clamp01(float(np.std(np.diff(peaks))) / 12.0)

    score = 0.38 * count_score + 0.30 * consistency + 0.22 * top_zone_score + 0.10 * uneven_top_bonus
    return clamp01(score)


def drop_score(y: np.ndarray) -> float:
    peak_idx = int(np.argmax(y[:60]))
    post = y[peak_idx:84]

    if len(post) < 8:
        return 0.0

    local_floor = float(np.min(post))
    drop = float(y[peak_idx] - local_floor)

    recovery = float(np.max(post[-8:]) - local_floor)
    recovery_penalty = clamp01(recovery / 0.25)

    score = clamp01(drop / 0.45) * (1 - 0.35 * recovery_penalty)
    return clamp01(score)


def shelf_score(y: np.ndarray) -> float:
    mid = y[48:100]
    if len(mid) < 20:
        return 0.0

    slope = abs(float(np.polyfit(np.arange(len(mid)), mid, 1)[0]))
    std = float(np.std(mid))
    mid_range = float(np.max(mid) - np.min(mid))

    flatness = clamp01(1 - std / 0.12)
    slope_score = clamp01(1 - slope / 0.0045)
    narrowness = clamp01(1 - mid_range / 0.28)

    score = 0.45 * flatness + 0.35 * slope_score + 0.20 * narrowness
    return clamp01(score)


def right_spike_score(y: np.ndarray) -> float:
    right = y[84:116]
    if len(right) < 12:
        return 0.0

    base = float(np.median(right))
    peak = float(np.max(right))
    spike = peak - base

    peaks, _ = find_peaks(right, prominence=0.04, distance=5)

    count_penalty = 0.0
    if len(peaks) == 0:
        count_penalty = 0.35
    elif len(peaks) > 1:
        count_penalty = min(0.5, (len(peaks) - 1) * 0.18)

    spike_strength = clamp01(spike / 0.22)
    return clamp01(spike_strength * (1 - count_penalty))


def reversion_score(y: np.ndarray) -> float:
    tail = y[116:128]

    if len(tail) < 8:
        return 0.0

    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    pre_tail_base = float(np.median(y[96:110]))

    returned_to_base = clamp01(1 - abs(tail_mean - pre_tail_base) / 0.18)
    flat_tail = clamp01(1 - tail_std / 0.10)
    no_second_breakout = clamp01(1 - max(0.0, tail_mean - pre_tail_base) / 0.15)

    score = 0.4 * returned_to_base + 0.35 * flat_tail + 0.25 * no_second_breakout
    return clamp01(score)


def asymmetry_score(y: np.ndarray) -> float:
    left = y[:64]
    right = y[64:]

    left_complexity = float(np.sum(np.abs(np.diff(left))))
    right_complexity = float(np.sum(np.abs(np.diff(right))))

    if left_complexity <= 1e-9:
        return 0.0

    ratio = right_complexity / left_complexity
    return clamp01(1 - max(0.0, ratio - 0.55) / 0.75)


def _template_curve() -> np.ndarray:
    anchors_x = np.array([0, 8, 16, 24, 30, 36, 44, 52, 60, 72, 88, 100, 108, 116, 127], dtype=float)
    anchors_y = np.array([0.18, 0.20, 0.55, 0.88, 0.78, 0.92, 0.60, 0.35, 0.32, 0.34, 0.33, 0.36, 0.55, 0.34, 0.32], dtype=float)

    f = interp1d(anchors_x, anchors_y, kind="linear")
    x = np.linspace(0, 127, 128)
    y = f(x)
    y = savgol_filter(y, 11, 3)

    ymin, ymax = float(np.min(y)), float(np.max(y))
    return (y - ymin) / (ymax - ymin)


_TEMPLATE = _template_curve()


def template_shape_score(y: np.ndarray) -> float:
    mad = float(np.mean(np.abs(y - _TEMPLATE)))
    return clamp01(1 - mad / 0.28)


def generate_notes(b: PatternBreakdown) -> list[str]:
    notes: list[str] = []

    if b.crown >= 0.75:
        notes.append("Сильная левая многозубая корона.")
    elif b.crown >= 0.55:
        notes.append("Корона читается, но не идеально собрана.")
    else:
        notes.append("Левая корона выражена слабо.")

    if b.shelf >= 0.72:
        notes.append("Есть длинная и достаточно ровная средняя полка.")
    elif b.shelf >= 0.52:
        notes.append("Полка присутствует, но не совсем плоская.")
    else:
        notes.append("Средняя полка слабая или слишком шумная.")

    if b.right_spike >= 0.72:
        notes.append("Правый шпиль выражен хорошо и визуально отделен от полки.")
    elif b.right_spike >= 0.5:
        notes.append("Правый выступ есть, но он умеренный.")
    else:
        notes.append("Правый выступ слабый или размазан.")

    if b.reversion >= 0.65:
        notes.append("После шпиля есть возврат в боковик.")
    else:
        notes.append("После шпиля возврат в полку выражен слабо.")

    if b.template_shape >= 0.72:
        notes.append("Общий силуэт близок к эталону паттерна.")
    elif b.template_shape >= 0.55:
        notes.append("Силуэт частично совпадает с эталоном.")
    else:
        notes.append("Силуэт заметно отклоняется от эталона.")

    return notes


def score_crown_shelf_right_spike(close: Iterable[float]) -> tuple[float, PatternBreakdown, list[str]]:
    y = normalize_series(close, target_len=TARGET_LEN)

    b = PatternBreakdown(
        crown=round(crown_score(y), 4),
        drop=round(drop_score(y), 4),
        shelf=round(shelf_score(y), 4),
        right_spike=round(right_spike_score(y), 4),
        reversion=round(reversion_score(y), 4),
        asymmetry=round(asymmetry_score(y), 4),
        template_shape=round(template_shape_score(y), 4),
    )

    score = (
        0.22 * b.crown +
        0.12 * b.drop +
        0.18 * b.shelf +
        0.16 * b.right_spike +
        0.08 * b.reversion +
        0.04 * b.asymmetry +
        0.20 * b.template_shape
    )

    score = round(clamp01(score) * 100, 2)
    notes = generate_notes(b)
    return score, b, notes
