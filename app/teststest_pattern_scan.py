from app.patterns import score_crown_shelf_right_spike
from app.services import find_best_window


def test_pattern_smoke():
    fake = [
        1.00, 1.05, 1.09, 1.06, 1.10, 1.04, 0.98, 0.93, 0.90, 0.89,
        0.90, 0.91, 0.90, 0.89, 0.90, 0.91, 0.92, 0.90, 0.96, 1.10,
        1.25, 1.08, 0.97, 0.92, 0.90, 0.91, 0.90, 0.90,
    ]
    result = score_crown_shelf_right_spike(fake)
    assert result.similarity >= 0
    assert result.label in {"strong match", "partial match", "weak-crown variant", "weak match"}
    assert result.stage in {"active", "completed", "forming"}


def test_best_window_search_returns_something():
    # окно с паттерном внутри более длинной истории
    history = (
        [0.5] * 20
        + [1.00, 1.05, 1.09, 1.06, 1.10, 1.04, 0.98, 0.93, 0.90, 0.89,
           0.90, 0.91, 0.90, 0.89, 0.90, 0.91, 0.92, 0.90, 0.96, 1.10,
           1.25, 1.08, 0.97, 0.92, 0.90, 0.91, 0.90, 0.90]
        + [0.52] * 20
    )
    best = find_best_window(history, 14, 60)
    assert best is not None
    assert "similarity" in best
    assert "best_window_start" in best
    assert "best_window_end" in best
    assert "candidate_windows_count" in best
    assert best["candidate_windows_count"] > 0


def test_no_hard_threshold_behavior():
    weakish = [1.0, 0.99, 1.01, 0.98, 0.97, 0.96, 0.95, 0.95, 0.96, 0.97, 0.99, 1.02] * 8
    best = find_best_window(weakish, 14, 90)
    # Даже слабый сигнал не должен выбрасываться как None, если окна есть
    assert best is not None
