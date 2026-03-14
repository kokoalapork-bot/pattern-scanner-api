from types import SimpleNamespace

from app import services


class DummyBreakdown:
    crown = 0.4
    drop = 0.5
    shelf = 0.7
    right_spike = 0.6
    reversion = 0.5
    asymmetry = 0.5
    template_shape = 0.6


def test_pre_breakout_stage_mode_prefers_setup_window(monkeypatch):
    # Synthetic series: long base then strong breakout near the end.
    closes = [1.0 + (0.01 if i % 7 == 0 else 0.0) for i in range(90)]
    closes += [1.05, 1.08, 1.12, 1.20, 1.35, 1.55, 1.78, 2.05, 2.35, 2.70]

    def fake_score(window_closes):
        # Legacy matcher is attracted to windows that already include late expansion.
        last = window_closes[-1]
        first = window_closes[0]
        growth = max(0.0, (last - first) / max(first, 1e-9))
        sim = min(95.0, 45.0 + growth * 120.0)
        return SimpleNamespace(
            similarity=sim,
            label="strong match",
            stage="active",
            breakdown=DummyBreakdown(),
            notes=["synthetic"],
        )

    monkeypatch.setattr(services, "score_crown_shelf_right_spike", fake_score)

    best_legacy = services.find_best_window(closes, min_age_days=30, max_age_days=90, stage_mode="legacy")
    best_pre = services.find_best_window(closes, min_age_days=30, max_age_days=90, stage_mode="pre_breakout_only")

    assert best_legacy is not None
    assert best_pre is not None

    # pre_breakout_only should avoid the latest post-breakout window.
    assert best_pre["best_window_end"] <= best_legacy["best_window_end"]
    assert best_pre["late_breakout_penalty"] <= best_legacy["late_breakout_penalty"]
    assert best_pre["post_breakout_extension_penalty"] <= best_legacy["post_breakout_extension_penalty"]
    assert best_pre["selected_window_stage"] == "pre_breakout"
