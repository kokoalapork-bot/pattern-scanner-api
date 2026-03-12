import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

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
    return max(0.0, min(1.0, float(x)))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def moving_average(y: List[float], window: int = 7) -> List[float]:
    if window <= 1 or len(y) < window:
        return y[:]

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = [y[0]] * pad + y[:] + [y[-1]] * pad
    out: List[float] = []

    for i in range(len(y)):
        chunk = padded[i:i + window]
        out.append(sum(chunk) / window)

    return out


def linear_resample(values: List[float], target_len: int) -> List[float]:
    if len(values) == target_len:
        return values[:]
    if len(values) == 1:
        return [values[0]] * target_len

    out: List[float] = []
    last_index = len(values) - 1

    for i in range(target_len):
        pos = (i * last_index) / (target_len - 1)
        left = int(math.floor(pos))
        right = int(math.ceil(pos))

        if left == right:
            out.append(values[left])
        else:
            frac = pos - left
            v = values[left] * (1 - frac) + values[right] * frac
            out.append(v)

    return out


def normalize_series(close: Iterable[float], target_len: int = TARGET_LEN) -> List[float]:
    arr = [float(x) for x in close if x is not None and math.isfinite(float(x))]

    if len(arr) < 30:
        raise ValueError("Need at least 30 data points to score a pattern")

    y = linear_resample(arr, target_len)
    y = moving_average(y, window=7)

    ymin = min(y)
    ymax = max(y)

    if math.isclose(ymin, ymax):
        return [0.0 for _ in y]

    return [(v - ymin) / (ymax - ymin) for v in y]


def find_local_peaks(y: List[float], min_prominence: float = 0.04, min_distance: int = 4) -> List[int]:
    peaks: List[int] = []
    last_peak = -10_000

    for i in range(1, len(y) - 1):
        if not (y[i] > y[i - 1] and y[i] > y[i + 1]):
            continue

        left_start = max(0, i - min_distance)
        right_end = min(len(y), i + min_distance + 1)

        left_min = min(y[left_start:i + 1])
        right_min = min(y[i:right_end])
        prominence = y[i] - max(left_min, right_min)

        if prominence < min_prominence:
            continue

        if i - last_peak < min_distance:
            if peaks and y[i] > y[peaks[-1]]:
                peaks[-1] = i
                last_peak = i
            continue

        peaks.append(i)
        last_peak = i

    return peaks


def simple_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2
    y_mean = mean(values)

    num = 0.0
    den = 0.0
    for i, v in enumerate(values):
        dx = i - x_mean
        num += dx * (v - y_mean)
        den += dx * dx

    if den == 0:
        return 0.0
    return num / den


def crown_score(y: List[float]) -> float:
    left = y[:52]
    peaks = find_local_peaks(left, min_prominence=0.04, min_distance=4)

    if len(peaks) < 3:
        return 0.0

    peak_vals = [left[i] for i in peaks]
    count_score = 1.0 if 3 <= len(peaks) <= 6 else clamp01(1 - abs(len(peaks) - 4) * 0.18)

    mean_val = mean(peak_vals)
    std_val = std(peak_vals)
    consistency = clamp01(1.0 - (std_val / (mean_val + 1e-9)))

    top_zone_score = sum(1 for v in peak_vals if v > 0.72) / len(peak_vals)
    spacing = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    uneven_top_bonus = clamp01(std(spacing) / 12.0) if spacing else 0.0

    score = 0.38 * count_score + 0.30 * consistency + 0.22 * top_zone_score + 0.10 * uneven_top_bonus
    return clamp01(score)


def drop_score(y: List[float]) -> float:
    peak_idx = max(range(min(60, len(y))), key=lambda i: y[i])
    post = y[peak_idx:84]

    if len(post) < 8:
        return 0.0

    local_floor = min(post)
    drop = y[peak_idx] - local_floor

    recovery = max(post[-8:]) - local_floor
    recovery_penalty = clamp01(recovery / 0.25)

    score = clamp01(drop / 0.45) * (1 - 0.35 * recovery_penalty)
    return clamp01(score)


def shelf_score(y: List[float]) -> float:
    mid = y[48:100]
    if len(mid) < 20:
        return 0.0

    slope = abs(simple_slope(mid))
    s = std(mid)
    mid_range = max(mid) - min(mid)

    flatness = clamp01(1 - s / 0.12)
    slope_score = clamp01(1 - slope / 0.0045)
    narrowness = clamp01(1 - mid_range / 0.28)

    score = 0.45 * flatness + 0.35 * slope_score + 0.20 * narrowness
    return clamp01(score)


def right_spike_score(y: List[float]) -> float:
    right = y[84:116]
    if len(right) < 12:
        return 0.0

    base = median(right)
    peak = max(right)
    spike = peak - base

    peaks = find_local_peaks(right, min_prominence=0.04, min_distance=5)

    count_penalty = 0.0
    if len(peaks) == 0:
        count_penalty = 0.35
    elif len(peaks) > 1:
        count_penalty = min(0.5, (len(peaks) - 1) * 0.18)

    spike_strength = clamp01(spike / 0.22)
    return clamp01(spike_strength * (1 - count_penalty))


def reversion_score(y: List[float]) -> float:
    tail = y[116:128]
    if len(tail) < 8:
        return 0.0

    tail_mean = mean(tail)
    tail_std = std(tail)
    pre_tail_base = median(y[96:110])

    returned_to_base = clamp01(1 - abs(tail_mean - pre_tail_base) / 0.18)
    flat_tail = clamp01(1 - tail_std / 0.10)
    no_second_breakout = clamp01(1 - max(0.0, tail_mean - pre_tail_base) / 0.15)

    score = 0.4 * returned_to_base + 0.35 * flat_tail + 0.25 * no_second_breakout
    return clamp01(score)


def asymmetry_score(y: List[float]) -> float:
    left = y[:64]
    right = y[64:]

    left_complexity = sum(abs(left[i + 1] - left[i]) for i in range(len(left) - 1))
    right_complexity = sum(abs(right[i + 1] - right[i]) for i in range(len(right) - 1))

    if left_complexity <= 1e-9:
        return 0.0

    ratio = right_complexity / left_complexity
    return clamp01(1 - max(0.0, ratio - 0.55) / 0.75)


def template_curve() -> List[float]:
    anchors: List[Tuple[float, float]] = [
        (0, 0.18), (8, 0.20), (16, 0.55), (24, 0.88), (30, 0.78),
        (36, 0.92), (44, 0.60), (52, 0.35), (60, 0.32), (72, 0.34),
        (88, 0.33), (100, 0.36), (108, 0.55), (116, 0.34), (127, 0.32),
    ]

    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]

    out: List[float] = []
    for i in range(128):
        if i <= xs[0]:
            out.append(ys[0])
            continue
        if i >= xs[-1]:
            out.append(ys[-1])
            continue

        for j in range(len(xs) - 1):
            if xs[j] <= i <= xs[j + 1]:
                left_x, right_x = xs[j], xs[j + 1]
                left_y, right_y = ys[j], ys[j + 1]
                frac = (i - left_x) / (right_x - left_x)
                out.append(left_y * (1 - frac) + right_y * frac)
                break

    out = moving_average(out, window=7)
    ymin = min(out)
    ymax = max(out)
    return [(v - ymin) / (ymax - ymin) for v in out]


_TEMPLATE = template_curve()


def template_shape_score(y: List[float]) -> float:
    mad = mean([abs(a - b) for a, b in zip(y, _TEMPLATE)])
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
