import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

from app.services import (
    build_asset_sources,
    dedupe_candidates_by_id,
    make_asset_key_from_id,
    make_asset_key_from_symbol,
    resolve_requested_assets,
)


def test_symbol_path_still_resolves():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
        {"id": "siren-2", "symbol": "siren", "name": "Siren", "market_cap": 10},
    ]

    (
        candidates,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        invalid_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=["RIVER", "SIREN"], coingecko_ids=None)

    assert resolved_symbols == ["RIVER", "SIREN"]
    assert unresolved_symbols == []
    assert resolved_coingecko_ids == []
    assert invalid_coingecko_ids == []
    assert len(candidates) == 2


def test_direct_coingecko_ids_path():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
        {"id": "siren-2", "symbol": "siren", "name": "Siren", "market_cap": 10},
    ]

    (
        candidates,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        invalid_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["river", "siren-2"])

    assert resolved_symbols == []
    assert unresolved_symbols == []
    assert resolved_coingecko_ids == ["river", "siren-2"]
    assert invalid_coingecko_ids == []
    assert len(candidates) == 2


def test_mixed_input_dedupes_by_final_id():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 100},
    ]

    (
        candidates,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        invalid_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=["RIVER"], coingecko_ids=["river"])

    assert len(candidates) == 1
    assert candidates[0]["id"] == "river"

    sources = build_asset_sources(["RIVER"], ["river"], debug_by_symbol)
    assert sources["river"]["source_type"] == "mixed"


def test_invalid_coingecko_id_is_not_unresolved_symbol():
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10},
    ]

    (
        candidates,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        invalid_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["this-does-not-exist-xyz"])

    assert candidates == []
    assert unresolved_symbols == []
    assert invalid_coingecko_ids == ["this-does-not-exist-xyz"]
    asset_key = make_asset_key_from_id("this-does-not-exist-xyz")
    assert skip_reasons[asset_key] == "invalid_coingecko_id"
    assert debug_by_symbol[asset_key].reason == "invalid_coingecko_id"


def test_stakestone_case_goes_by_id():
    markets = [
        {"id": "stakestone", "symbol": "sto", "name": "StakeStone", "market_cap": 10},
    ]

    (
        candidates,
        resolved_symbols,
        unresolved_symbols,
        resolved_coingecko_ids,
        invalid_coingecko_ids,
        skip_reasons,
        debug_by_symbol,
    ) = resolve_requested_assets(markets, symbols=None, coingecko_ids=["stakestone"])

    assert len(candidates) == 1
    assert candidates[0]["id"] == "stakestone"
    assert resolved_coingecko_ids == ["stakestone"]
    assert invalid_coingecko_ids == []
