import asyncio
import os

os.environ.setdefault("COINGECKO_AUTH_MODE", "demo")
os.environ.setdefault("COINGECKO_API_KEY", "test-demo-key")
os.environ.setdefault("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")
from types import SimpleNamespace

from app.data_sources import MarketDataFetchResult
from app.models import ScanRequest
from app.patterns import PatternBreakdown
from app.services import scan_pattern


class DummyClient:
    def __init__(self, markets, fetch_map):
        self._markets = markets
        self._fetch_map = fetch_map
        self.auth = SimpleNamespace(
            mode="demo",
            base_url="https://api.coingecko.com/api/v3",
            api_key_present=True,
            header_name="x-cg-demo-api-key",
        )

    async def close(self):
        return None

    async def get_markets(self, vs_currency="usd", pages=1, per_page=250):
        return self._markets

    def normalize_history_days(self, requested_days: int):
        return requested_days, False

    async def fetch_market_history(self, coingecko_id: str, vs_currency="usd", days=450, interval="daily"):
        return self._fetch_map[coingecko_id]


def test_scan_compact_response_mode(monkeypatch):
    from app import services

    prices = [[1_700_000_000_000 + i * 86_400_000, 1.0 + (i % 10) * 0.01] for i in range(80)]
    markets = [
        {"id": "river", "symbol": "river", "name": "River", "market_cap": 10_000_000, "total_volume": 1_000_000},
    ]
    fetch_map = {
        "river": MarketDataFetchResult(ok=True, chart={"prices": prices}),
    }

    monkeypatch.setattr(services, "CoinGeckoClient", lambda: DummyClient(markets, fetch_map))
    monkeypatch.setattr(
        services,
        "find_best_window",
        lambda closes, min_age_days, max_age_days: {
            "similarity": 91.0,
            "raw_similarity": 88.0,
            "structural_score": 82.0,
            "base_label": "strong match",
            "stage": "forming",
            "breakdown": PatternBreakdown(0.4, 0.6, 0.7, 0.65, 0.52, 0.57, 0.66),
            "notes": ["x"],
            "best_window_len": 30,
            "best_window_start": 0,
            "best_window_end": 30,
            "best_age_days": 30,
            "exemplar_consistency_score": 79.0,
            "distance_to_siren_breakdown": 0.11,
            "distance_to_river_breakdown": 0.14,
            "reference_band_passed": True,
            "left_structure_ok": True,
            "pre_breakout_tail_ok": True,
            "stage_ok": True,
            "pre_breakout_base_score": 81.0,
            "candidate_windows_count": 3,
        },
    )
    monkeypatch.setattr(services, "classify_final_label", lambda **kwargs: "strong match")
    monkeypatch.setattr(services, "classify_behavioral_universe_filter", lambda closes: ("included_for_scoring", "included_for_scoring"))

    req = ScanRequest(
        pattern_name="crown_shelf_right_spike",
        min_age_days=14,
        max_age_days=30,
        top_k=1,
        max_coins_to_evaluate=1,
        vs_currency="usd",
        include_notes=False,
        compact_response=True,
    )

    resp = asyncio.run(scan_pattern(req))
    payload = resp.model_dump()

    assert req.compact_response is True
    assert "debug_by_symbol" not in payload
    assert "pre_filter_candidates" not in payload
    assert "skip_reasons" not in payload

    assert set(payload.keys()) == {
        "pattern_name",
        "evaluated_count",
        "returned_count",
        "market_offset",
        "market_batch_size",
        "market_batch_ids",
        "results",
    }

    assert len(payload["results"]) == 1
    assert set(payload["results"][0].keys()) == {
        "coingecko_id",
        "symbol",
        "name",
        "similarity",
        "label",
    }
