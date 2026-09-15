"""Tests for static player valuation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from draft_buddy.data.player_valuation import (
    add_static_player_value_metrics,
    agreement_score_from_z,
    estimate_data_confidence,
    estimate_effective_sample_size,
)


def _valuation_frame() -> pd.DataFrame:
    """Return a small market/history valuation fixture."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "History Star",
                "position": "RB",
                "projected_points": 20.0,
                "adp": 2.0,
                "games_played_frac": 1.0,
            },
            {
                "player_id": 2,
                "name": "Market Premium",
                "position": "RB",
                "projected_points": 11.0,
                "adp": 1.0,
                "games_played_frac": 0.3,
            },
            {
                "player_id": 3,
                "name": "Rookie",
                "position": "RB",
                "projected_points": 10.0,
                "adp": 3.0,
                "games_played_frac": "R",
            },
            {
                "player_id": 4,
                "name": "Consensus WR",
                "position": "WR",
                "projected_points": 15.0,
                "adp": 20.0,
                "games_played_frac": 1.0,
            },
            {
                "player_id": 5,
                "name": "Second WR",
                "position": "WR",
                "projected_points": 14.0,
                "adp": 30.0,
                "games_played_frac": 1.0,
            },
        ]
    )


def test_add_static_player_value_metrics_labels_market_premium_player() -> None:
    """Verify early ADP with weak history remains visible as market-backed upside."""
    valued = add_static_player_value_metrics(_valuation_frame())
    player = valued[valued["name"] == "Market Premium"].iloc[0]

    assert player["market_premium_points"] > 0.0
    assert player["player_archetype"] == "market_backed_upside"
    assert player["projection_basis"] == "market_supported"


def test_add_static_player_value_metrics_identifies_proven_consensus() -> None:
    """Verify strong history near market expectation gets a consensus label."""
    valued = add_static_player_value_metrics(_valuation_frame())
    player = valued[valued["name"] == "Consensus WR"].iloc[0]

    assert player["agreement_score"] >= 80.0
    assert player["data_confidence"] == pytest.approx(1.0)


def test_add_static_player_value_metrics_keeps_rookie_agreement_separate() -> None:
    """Verify ADP-derived rookie rows are flagged as market-imputed."""
    valued = add_static_player_value_metrics(_valuation_frame())
    rookie = valued[valued["name"] == "Rookie"].iloc[0]

    assert rookie["projection_basis"] == "rookie_market_imputed"
    assert rookie["effective_sample_size"] == pytest.approx(0.0)


def test_projection_intervals_widen_for_sparse_history() -> None:
    """Verify sparse history receives a wider projection interval."""
    valued = add_static_player_value_metrics(_valuation_frame())
    sparse = valued[valued["name"] == "Market Premium"].iloc[0]
    proven = valued[valued["name"] == "Consensus WR"].iloc[0]
    sparse_width = sparse["projected_points_p80"] - sparse["projected_points_p20"]
    proven_width = proven["projected_points_p80"] - proven["projected_points_p20"]

    assert sparse_width > proven_width


def test_agreement_score_from_z_declines_with_disagreement() -> None:
    """Verify standardized edge maps to an intuitive agreement score."""
    assert agreement_score_from_z(0.0) == pytest.approx(100.0)
    assert agreement_score_from_z(2.0) < agreement_score_from_z(1.0)


def test_data_confidence_handles_rookies_and_missing_values() -> None:
    """Verify data confidence is bounded for sparse and rookie rows."""
    assert estimate_data_confidence("R") == pytest.approx(0.35)
    assert estimate_data_confidence(np.nan) == pytest.approx(0.35)
    assert estimate_data_confidence(0.01) == pytest.approx(0.15)


def test_effective_sample_size_uses_lookback_game_estimate() -> None:
    """Verify effective sample size scales from games-played fraction."""
    assert estimate_effective_sample_size(0.5) == pytest.approx(17.0)
    assert estimate_effective_sample_size("R") == pytest.approx(0.0)
