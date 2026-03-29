
from datetime import datetime, timedelta, timezone

from app.patterns import score_crown_shelf_right_spike


def _ts(days_offset: int) -> int:
    base = datetime(2025, 9, 1, tzinfo=timezone.utc)
    return int((base + timedelta(days=days_offset)).timestamp() * 1000)


def test_reference_coin_passes_on_reference_window():
    prices = []
    prices += [10 + i * 0.1 for i in range(20)]
    prices += [13.5, 14.0, 14.3, 14.5, 14.4, 14.6, 14.5, 14.55, 14.35, 14.4, 14.2, 14.25]
    prices += [14.1, 14.0, 13.8, 13.7, 13.5, 13.2, 12.8, 12.1, 11.5, 10.9, 10.2, 9.8]
    prices += [9.7, 9.6, 9.65, 9.7, 9.6, 9.7, 9.8, 9.7, 9.75, 9.7]
    prices += [10.1, 10.8, 11.5, 10.4, 9.9, 9.8, 9.7]
    timestamps = [_ts(i) for i in range(len(prices))]
    result = score_crown_shelf_right_spike(prices, timestamps=timestamps, coin_id="river")
    assert result.similarity >= 70


def test_strict_filters_reject_weak_general_shape():
    prices = [1, 1.2, 1.4, 1.7, 1.9, 2.1, 2.15, 2.2, 2.18, 2.22, 2.19, 2.21, 2.17,
              2.15, 2.1, 2.05, 1.95, 1.8, 1.65, 1.5, 1.42, 1.38, 1.35, 1.33, 1.32,
              1.33, 1.34, 1.33, 1.32, 1.34, 1.35, 1.42, 1.5, 1.62, 1.48, 1.4, 1.35, 1.33, 1.32]
    result = score_crown_shelf_right_spike(prices)
    assert result.similarity == 0
    assert result.stage == "filtered"
