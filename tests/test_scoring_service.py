"""Tests for scoring service behavior."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.scoring.engine import ScoringEngine
from draft_buddy.data.scoring.service import ScoringService, _weekly_projections_from_draft_players


def test_weekly_projections_zero_out_bye_week() -> None:
    """Verify bye weeks are projected as zero points."""
    dataframe = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": 10.0, "bye_week": 2}])

    assert _weekly_projections_from_draft_players(dataframe)[1][2] == 0


def test_apply_scoring_returns_empty_frame_with_total_pts_column() -> None:
    """Verify empty input produces an empty frame with total_pts."""
    result = ScoringService().apply_scoring(pd.DataFrame())

    assert "total_pts" in result.columns and result.empty


def test_calculate_games_played_frac_returns_empty_frame_for_empty_input() -> None:
    """Verify empty history yields an empty fraction frame."""
    result = ScoringService().calculate_games_played_frac(pd.DataFrame())

    assert list(result.columns) == ["player_id", "games_played_frac"]


def test_merge_roster_with_legacy_assigns_new_rookie_ids() -> None:
    """Verify rookies receive generated ids above the legacy max id."""
    draft_pool_df = pd.DataFrame(
        [
            {"player_display_name": "Vet", "position": "QB"},
            {"player_display_name": "Rookie", "position": "QB"},
        ]
    )
    legacy_stats_df = pd.DataFrame(
        [{"player_id": 7, "player_display_name": "Vet", "position": "QB", "total_pts": 10.0, "games_played_frac": 1.0}]
    )
    result = ScoringService().merge_roster_with_legacy(draft_pool_df, legacy_stats_df)

    assert int(result.loc[result["player_display_name"] == "Rookie", "player_id"].iloc[0]) == 8


def test_apply_rookie_metadata_marks_rookies_with_r() -> None:
    """Verify rookie metadata replaces games_played_frac with R."""
    dataframe = pd.DataFrame([{"games_played_frac": 0.5, "is_rookie_original": True}])
    result = ScoringService().apply_rookie_metadata(dataframe)

    assert result.iloc[0]["games_played_frac"] == "R"


def test_finalize_draft_players_sorts_by_total_points_descending() -> None:
    """Verify finalized draft players are sorted by total points descending."""
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "player_display_name": "A", "position": "QB", "recent_team": "BUF", "total_pts": 1.0, "games_played_frac": 1.0, "bye_week": 7},
            {"player_id": 2, "player_display_name": "B", "position": "QB", "recent_team": "BUF", "total_pts": 5.0, "games_played_frac": 1.0, "bye_week": 7},
        ]
    )
    result = ScoringService().finalize_draft_players(dataframe)

    assert list(result["player_id"]) == [2, 1]


def test_prepare_offense_kicking_features_creates_total_fumbles_lost() -> None:
    """Verify composite total_fumbles_lost is derived from component columns."""
    result = ScoringEngine.prepare_offense_kicking_features(
        pd.DataFrame([{"sack_fumbles_lost": 1, "rushing_fumbles_lost": 2, "receiving_fumbles_lost": 3}])
    )

    assert int(result.iloc[0]["total_fumbles_lost"]) == 6


def test_apply_scoring_adds_linear_stat_points() -> None:
    """Verify apply_scoring sums configured linear scoring rules."""
    result = ScoringEngine.apply_scoring(
        pd.DataFrame([{"passing_yards": 250, "passing_tds": 2}]),
        {"passing_yards": 0.04, "passing_tds": 4},
    )

    assert float(result.iloc[0]["total_pts"]) == 18.0
