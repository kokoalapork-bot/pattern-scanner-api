import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

from app.services import (
    build_asset_sources,
    make_asset_key_from_id,
    resolve_requested_assets,
)


def test_symbol_path_still_resolves():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
        {"id": "siren-2", "symbol": "siren", "name": "Siren", "market_cap": 10},
    ]

    (
        candidates,
        pending_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=["RIVER", "SIREN"], coingecko_ids=None)

    assert resolved_symbols == ["RIVER", "SIREN"]
    assert unresolved_symbols == []
    assert resolved_coingecko_ids == []
    assert pending_ids == []
    assert len(candidates) == 2


def test_direct_coingecko_ids_path_from_markets():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
        {"id": "siren-2", "symbol": "siren", "name": "Siren", "market_cap": 10},
    ]

    (
        candidates,
        pending_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["river", "siren-2"])

    assert resolved_symbols == []
    assert unresolved_symbols == []
    assert resolved_coingecko_ids == ["river", "siren-2"]
    assert pending_ids == []
    assert len(candidates) == 2


def test_mixed_input_dedupes_by_final_id():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 100},
    ]

    (
        candidates,
        pending_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=["RIVER"], coingecko_ids=["river"])

    assert len(candidates) == 1
    assert candidates[0]["id"] == "river"

    sources = build_asset_sources(debug_by_symbol)
    assert sources["river"]["source_type"] == "mixed"


def test_missing_from_markets_is_not_invalid_immediately():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
    ]

    (
        candidates,
        pending_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["stakestone"])

    assert candidates == []
    assert unresolved_symbols == []
    assert resolved_coingecko_ids == []
    assert pending_ids == ["stakestone"]

    asset_key = make_asset_key_from_id("stakestone")
    assert asset_key in debug_by_symbol
    assert debug_by_symbol[asset_key].reason is None
    assert debug_by_symbol[asset_key].status == "pending_lookup"


def test_stakestone_case_is_not_rejected_by_symbol_logic():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
    ]

    (
        candidates,
        pending_ids,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["stakestone"])

    assert "stakestone" in pending_ids
    assert unresolved_symbols == []
    assert resolved_symbols == []
