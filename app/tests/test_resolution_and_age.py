
import pytest

from app.services import _age_days_from_history, _resolve_symbol


def test_age_days_from_history_uses_first_candle():
    history = {"prices": [[1735689600000, 1.0], [1735776000000, 1.1]]}  # 2025-01-01 UTC
    age = _age_days_from_history(history)
    assert age is not None
    assert age > 300


@pytest.mark.asyncio
async def test_resolve_symbol_prefers_better_ranked_exact_match():
    class DummyClient:
        async def search_symbol(self, query: str):
            return [
                {"id": "siren", "symbol": "SIREN", "market_cap_rank": 9999},
                {"id": "siren-2", "symbol": "SIREN", "market_cap_rank": 58},
            ]

    resolved = await _resolve_symbol(DummyClient(), "SIREN")
    assert resolved == "siren-2"
