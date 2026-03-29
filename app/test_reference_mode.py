
from app.patterns import score_crown_shelf_right_spike


def test_score_returns_valid_result():
    prices = [10, 12, 15, 17, 16, 13, 11, 10, 9, 9.5, 10, 10.2, 10.1, 10.0, 10.3, 14, 18, 12, 10.4, 10.2]
    prices = prices + [10.1] * 15
    result = score_crown_shelf_right_spike(prices)
    assert 0 <= result.similarity <= 100
    assert result.label in {"strong match", "partial match", "weak match"}
