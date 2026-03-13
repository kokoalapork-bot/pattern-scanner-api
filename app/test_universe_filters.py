import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

from app.data_sources import (
    classify_behavioral_universe_filter,
    looks_like_major,
    looks_like_stable,
)
from app.services import classify_universe_filter_from_market


def test_btc_is_excluded_as_major():
    coin = {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "market_cap": 1_000_000_000_000,
        "total_volume": 1_000_000_000,
    }
    status, reason = classify_universe_filter_from_market(coin)
    assert status == "excluded_major"
    assert reason == "excluded_btc"


def test_eth_is_excluded_as_major():
    coin = {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "market_cap": 500_000_000_000,
        "total_volume": 1_000_000_000,
    }
    status, reason = classify_universe_filter_from_market(coin)
    assert status == "excluded_major"
    assert reason == "excluded_eth"


def test_large_cap_alt_is_excluded():
    coin = {
        "id": "solana",
        "symbol": "sol",
        "name": "Solana",
        "market_cap": 50_000_000_000,
        "total_volume": 1_000_000_000,
    }
    status, reason = classify_universe_filter_from_market(coin)
    assert status == "excluded_large_cap"
    assert reason == "excluded_large_cap"


def test_obvious_stablecoin_is_excluded():
    coin = {
        "id": "tether",
        "symbol": "usdt",
        "name": "Tether USD",
        "market_cap": 100_000_000_000,
        "total_volume": 1_000_000_000,
    }
    status, reason = classify_universe_filter_from_market(coin)
    assert status == "excluded_stablecoin"
    assert reason == "excluded_stablecoin_denylist"


def test_low_volatility_pseudo_stable_is_excluded():
    closes = [1.00, 1.01, 0.99, 1.00, 1.01, 1.00, 0.99, 1.00]
    status, reason = classify_behavioral_universe_filter(closes)
    assert status == "excluded_stablecoin"
    assert reason == "excluded_stablecoin_behavior"


def test_broad_low_vol_asset_is_excluded():
    closes = [10.00, 10.02, 10.01, 10.00, 10.01, 10.00, 10.02, 10.01]
    status, reason = classify_behavioral_universe_filter(closes)
    assert status == "excluded_low_volatility"
    assert reason == "excluded_low_volatility"
