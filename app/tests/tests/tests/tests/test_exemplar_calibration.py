import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

from types import SimpleNamespace

from app.services import (
    EXEMPLAR_BREAKDOWNS,
    classify_final_label,
    compute_exemplar_metrics,
)


def test_siren_profile_is_consistent_with_reference_band():
    breakdown = SimpleNamespace(**EXEMPLAR_BREAKDOWNS["SIREN"])
    metrics = compute_exemplar_metrics(breakdown)
    assert metrics["reference_band_passed"] is True
    assert metrics["distance_to_siren_breakdown"] == 0.0


def test_river_profile_is_consistent_with_reference_band():
    breakdown = SimpleNamespace(**EXEMPLAR_BREAKDOWNS["RIVER"])
    metrics = compute_exemplar_metrics(breakdown)
    assert metrics["reference_band_passed"] is True
    assert metrics["distance_to_river_breakdown"] == 0.0


def test_guardrail_blocks_strong_match_when_far_from_exemplars():
    label = classify_final_label(
        base_label="strong match",
        structural_score=80.0,
        exemplar_consistency_score=20.0,
        reference_band_passed=False,
        universe_filter_status="included_for_scoring",
    )
    assert label in {"partial match", "weak match"}


def test_reject_when_structure_too_weak():
    label = classify_final_label(
        base_label="weak match",
        structural_score=20.0,
        exemplar_consistency_score=80.0,
        reference_band_passed=True,
        universe_filter_status="included_for_scoring",
    )
    assert label == "reject"
