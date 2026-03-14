import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

import pytest

from app.models import ScanRequest
from app.services import (
    build_automatic_market_universe,
    classify_universe_filter_from_market,
)


class DummyClient:
    pass


@pytest.mark.asyncio
async def test_automatic_market_universe_uses_coingecko_id_as_primary_key():
    markets = [
        {
            "id": "stakestone",
            "symbol": "sto",
            "name": "StakeStone",
            "market_cap": 120_000_000,
            "total_volume": 8_000_000,
        },
        {
            "id": "river",
            "symbol": "river",
            "name": "River",
            "market_cap": 50_000_000,
            "total_volume": 5_000_000,
        },
    ]

    req = ScanRequest(
        pattern_name="crown_shelf_right_spike",
        min_age_days=14,
        max_age_days=450,
        top_k=10,
        max_coins_to_evaluate=10,
        vs_currency="usd",
        include_notes=True,
    )

    (
        candidates,
        debug_by_asset,
        skip_reasons,
        asset_sources,
        universe_total_count,
        market_batch_size,
        market_batch_ids,
    ) = await build_automatic_market_universe(
        client=DummyClient(),
        req=req,
        markets=markets,
    )

    assert universe_total_count == 2
    assert market_batch_size == 2
    assert set(market_batch_ids) == {"stakestone", "river"}

    ids = {c["id"] for c in candidates}
    assert "stakestone" in ids
    assert "river" in ids

    assert asset_sources["stakestone"]["input_coingecko_id"] == "stakestone"
    assert asset_sources["stakestone"]["source_type"] == "market_universe"

    stake_key = "id:stakestone"
    assert debug_by_asset[stake_key].coingecko_id == "stakestone"
    assert debug_by_asset[stake_key].input_symbol == "STO"


def test_stakestone_is_not_lost_due_to_symbol_resolution():
    coin = {
        "id": "stakestone",
        "symbol": "sto",
        "name": "StakeStone",
        "market_cap": 120_000_000,
        "total_volume": 8_000_000,
    }
    status, reason = classify_universe_filter_from_market(coin)
    assert status == "included_for_scoring"
    assert reason == "included_for_scoring"


def test_btc_eth_and_stables_are_still_filtered_before_scoring():
    btc = {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "market_cap": 1_000_000_000_000,
        "total_volume": 1_000_000_000,
    }
    usdt = {
        "id": "tether",
        "symbol": "usdt",
        "name": "Tether USD",
        "market_cap": 100_000_000_000,
        "total_volume": 1_000_000_000,
    }

    assert classify_universe_filter_from_market(btc)[0] != "included_for_scoring"
    assert classify_universe_filter_from_market(usdt)[0] != "included_for_scoring"
