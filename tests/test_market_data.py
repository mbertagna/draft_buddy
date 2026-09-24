"""Tests for market-data ingestion and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from draft_buddy.data.market_data import (
    FantasyProsGateway,
    MarketDataUnavailable,
    MarketRankingRequest,
    assert_rankings_payload_complete,
    consensus_adp_from_sources,
    rankings_payload_to_dataframe,
)


def _ranking_payload() -> dict:
    """Return a minimal FantasyPros rankings payload."""
    return {
        "total_experts": 43,
        "last_updated": "8/29",
        "players": [
            {
                "player_id": 19788,
                "player_name": "Ja'Marr Chase",
                "player_team_id": "CIN",
                "player_position_id": "WR",
                "rank_ecr": "1",
                "rank_min": "1",
                "rank_max": "4",
                "rank_ave": "1.32",
                "rank_std": "0.91",
                "tier": "1",
            }
        ],
    }


def _fake_response(payload: dict, status_code: int = 200):
    """Build a minimal requests-like response object."""
    response = SimpleNamespace()
    response.status_code = status_code
    response.json = lambda: payload

    def raise_for_status() -> None:
        if status_code >= 400:
            raise requests.HTTPError(response=response)

    response.raise_for_status = raise_for_status
    return response


class FakeSession:
    """In-memory HTTP session for deterministic gateway tests."""

    def __init__(self, responses: list) -> None:
        """Store queued responses or exceptions."""
        self.responses = list(responses)
        self.calls = []

    def get(self, *args, **kwargs):
        """Return or raise the next queued response."""
        self.calls.append((args, kwargs))
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def test_assert_rankings_payload_complete_rejects_public_api_cap() -> None:
    """Verify a free-tier 10-player ADP payload is treated as unavailable."""
    payload = {
        "count": 366,
        "limit": 10,
        "public_api_limited": True,
        "players": [{"player_name": "Jahmyr Gibbs"}] * 10,
    }

    with pytest.raises(MarketDataUnavailable, match="truncated"):
        assert_rankings_payload_complete(payload)


def test_fantasypros_gateway_does_not_cache_truncated_payload(tmp_path: Path) -> None:
    """Verify truncated live responses are not stored as the latest ranking cache."""
    payload = {
        "count": 366,
        "limit": 10,
        "public_api_limited": True,
        "players": [{"player_name": "Jahmyr Gibbs", "player_position_id": "RB"}] * 10,
    }
    session = FakeSession([_fake_response(payload)])
    gateway = FantasyProsGateway(cache_dir=str(tmp_path), api_key="key", session=session)
    request = MarketRankingRequest(season=2026, scoring="HALF")
    try:
        gateway.fetch_consensus_rankings(request)
    except MarketDataUnavailable:
        pass

    assert not (tmp_path / "2026_half_adp_all_0_latest.json").exists()
    """Verify FantasyPros ADP rows normalize with provenance and dispersion."""
    frame = rankings_payload_to_dataframe(_ranking_payload(), "ADP", "2026-08-01T00:00:00+00:00")

    row = frame.iloc[0]
    assert row["source"] == "fantasypros_adp"
    assert row["adp"] == pytest.approx(1.32)
    assert row["rank_std"] == pytest.approx(0.91)
    assert row["total_experts"] == 43


def test_rankings_payload_to_dataframe_keeps_ecr_separate_from_adp() -> None:
    """Verify non-ADP rankings populate ECR instead of ADP."""
    frame = rankings_payload_to_dataframe(_ranking_payload(), "draft", "2026-08-01T00:00:00+00:00")

    row = frame.iloc[0]
    assert pd.isna(row["adp"]) and row["ecr"] == pytest.approx(1.0)


def test_fantasypros_gateway_writes_snapshot_and_latest_cache(tmp_path: Path) -> None:
    """Verify successful API responses are normalized and cached."""
    session = FakeSession([_fake_response(_ranking_payload())])
    gateway = FantasyProsGateway(cache_dir=str(tmp_path), api_key="key", session=session)
    request = MarketRankingRequest(season=2026, scoring="PPR")

    frame = gateway.fetch_consensus_rankings(request)

    assert frame.iloc[0]["player_name"] == "Ja'Marr Chase"
    assert (tmp_path / "2026_ppr_adp_all_0_latest.json").exists()
    assert list(tmp_path.glob("2026_ppr_adp_all_0_*.json"))
    assert session.calls[0][1]["headers"] == {"x-api-key": "key"}


def test_fantasypros_gateway_uses_stale_cache_when_key_missing(tmp_path: Path) -> None:
    """Verify cache fallback works when the API key is unavailable."""
    request = MarketRankingRequest(season=2026, scoring="HALF", position="RB")
    cache_path = tmp_path / "2026_half_adp_rb_0_latest.json"
    cache_path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-08-01T00:00:00+00:00",
                "request": {},
                "payload": _ranking_payload(),
            }
        ),
        encoding="utf-8",
    )
    gateway = FantasyProsGateway(cache_dir=str(tmp_path), api_key="")

    frame = gateway.fetch_consensus_rankings(request)

    assert frame.iloc[0]["source"] == "fantasypros_adp"


def test_fantasypros_gateway_can_require_fresh_api_response(tmp_path: Path) -> None:
    """Verify callers can opt out of stale-cache fallback."""
    gateway = FantasyProsGateway(cache_dir=str(tmp_path), api_key="")
    request = MarketRankingRequest(season=2026, scoring="PPR")

    with pytest.raises(MarketDataUnavailable, match="FANTASYPROS_API_KEY"):
        gateway.fetch_consensus_rankings(request, allow_stale_cache=False)


def test_fantasypros_gateway_retries_transient_failures(tmp_path: Path) -> None:
    """Verify transient HTTP failures are retried before succeeding."""
    session = FakeSession([requests.ConnectionError("temporary"), _fake_response(_ranking_payload())])
    gateway = FantasyProsGateway(
        cache_dir=str(tmp_path),
        api_key="key",
        session=session,
        max_retries=1,
    )

    frame = gateway.fetch_consensus_rankings(MarketRankingRequest(season=2026, scoring="PPR"))

    assert len(session.calls) == 2
    assert frame.iloc[0]["adp"] == pytest.approx(1.32)


def test_consensus_adp_from_sources_uses_median_and_source_count() -> None:
    """Verify multi-source ADP consensus preserves dispersion and provenance."""
    fantasypros = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "position": "RB",
                "team": "BUF",
                "adp": 20.0,
                "source": "fantasypros_adp",
            }
        ]
    )
    sleeper = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "position": "RB",
                "team": "BUF",
                "adp": 24.0,
                "source": "sleeper_projection_adp",
            }
        ]
    )

    consensus = consensus_adp_from_sources([fantasypros, sleeper])

    row = consensus.iloc[0]
    assert row["consensus_adp"] == pytest.approx(22.0)
    assert row["market_source_count"] == 2
    assert row["market_adp_dispersion"] == pytest.approx(2.0)
