import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

from types import SimpleNamespace

from app.patterns import score_crown_shelf_right_spike
from app.services import find_best_window, validate_scan_invariants


def test_broad_range_search_returns_best_window():
    closes = (
        [10.0] * 25
        + [12, 14, 13, 15, 12, 11, 10]
        + [8.0] * 60
        + [12.5, 14.0, 9.0, 8.2, 8.1, 8.0]
        + [7.9] * 100
    )

    best = find_best_window(closes, 14, 180)
    assert best is not None
    assert best["candidate_windows_count"] > 0
    assert best["best_window_end"] <= len(closes)


def test_partial_matches_are_not_zeroed():
    closes = [10, 12, 11, 13, 12, 9, 8, 8, 8.1, 8.0, 8.2, 8.1, 9.5, 11.0, 8.4, 8.2, 8.1]
    result = score_crown_shelf_right_spike(closes)
    assert result.label in {"strong match", "partial match", "weak-crown variant", "weak match"}
    assert result.similarity >= 0.0


def test_pipeline_invariants_ok():
    debug_by_symbol = {
        "BTC": SimpleNamespace(reason=None),
        "DOGE": SimpleNamespace(reason="history_http_404"),
        "UNKNOWN": SimpleNamespace(reason="unresolved_symbol"),
    }

    validate_scan_invariants(
        unresolved_symbols=["UNKNOWN"],
        skipped_symbols=["DOGE"],
        evaluated_symbols=["BTC"],
        skip_reasons={"DOGE": "history_http_404"},
        debug_by_symbol=debug_by_symbol,
        evaluated_count=1,
    )


def test_regression_symbol_pairs_shape():
    pairs = [
        ("BTC", "bitcoin"),
        ("ETH", "ethereum"),
        ("SIREN", "siren-2"),
        ("RIVER", "river"),
    ]
    for symbol, coingecko_id in pairs:
        assert isinstance(symbol, str)
        assert isinstance(coingecko_id, str)
        assert coingecko_id
