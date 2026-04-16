"""Behavioral tests for the scoring service."""

from unittest.mock import patch

import pandas as pd

from draft_buddy.logic.scoring.service import ScoringService


def test_apply_scoring_returns_empty_frame_with_total_pts_for_none_input():
    """Verify None input returns an empty frame with the scoring column present."""
    scored = ScoringService().apply_scoring(None)

    assert list(scored.columns) == ["total_pts"]


def test_calculate_games_played_frac_aggregates_across_multiple_seasons_and_teams():
    """Verify games-played fraction uses summed team games over a player's career."""
    historical = pd.DataFrame(
        [
            {"player_id": 1, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 1, "season": 2023, "recent_team": "BUF", "week": 2},
            {"player_id": 1, "season": 2024, "recent_team": "KC", "week": 1},
            {"player_id": 2, "season": 2024, "recent_team": "KC", "week": 1},
            {"player_id": 2, "season": 2024, "recent_team": "KC", "week": 2},
        ]
    )

    fractions = ScoringService().calculate_games_played_frac(historical)

    assert round(float(fractions.loc[fractions["player_id"] == 1, "games_played_frac"].iloc[0]), 2) == 0.75


def test_aggregate_legacy_stats_supports_mean_measure_of_center():
    """Verify non-median aggregation uses the mean."""
    historical = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 10.0, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 1, "total_pts": 16.0, "season": 2024, "recent_team": "BUF", "week": 1},
        ]
    )

    legacy = ScoringService().aggregate_legacy_stats(historical, measure_of_center="mean")

    assert float(legacy["total_pts"].iloc[0]) == 13.0


def test_aggregate_legacy_stats_skips_metadata_when_metadata_merge_raises():
    """Verify metadata enrichment failure still returns the aggregated stats."""
    service = ScoringService()
    historical = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 10.0, "season": 2023, "recent_team": "BUF", "week": 1},
        ]
    )

    with patch.object(pd.DataFrame, "sort_values", side_effect=RuntimeError("boom")):
        legacy = service.aggregate_legacy_stats(historical)

    assert list(legacy.columns) == ["player_id", "total_pts", "games_played_frac"]


def test_merge_roster_with_legacy_merges_on_name_and_position_when_metadata_exists():
    """Verify roster merge matches by name and position when metadata is available."""
    draft_pool = pd.DataFrame(
        [{"player_display_name": "Player A", "position": "WR", "recent_team": "BUF"}]
    )
    legacy = pd.DataFrame(
        [{"player_id": 7, "player_display_name": "Player A", "position": "WR", "total_pts": 99.0, "games_played_frac": 1.0}]
    )

    merged = ScoringService().merge_roster_with_legacy(draft_pool, legacy)

    assert int(merged["player_id"].iloc[0]) == 7


def test_merge_roster_with_legacy_falls_back_to_player_id_when_name_metadata_missing():
    """Verify roster merge falls back to player_id when name/position metadata is unavailable."""
    draft_pool = pd.DataFrame(
        [{"player_id": 11, "player_display_name": "Player A", "position": "WR"}]
    )
    legacy = pd.DataFrame([{"player_id": 11, "total_pts": 88.0, "games_played_frac": 1.0}])

    merged = ScoringService().merge_roster_with_legacy(draft_pool, legacy)

    assert float(merged["total_pts"].iloc[0]) == 88.0


def test_merge_roster_with_legacy_initializes_missing_player_id_column_when_needed():
    """Verify roster merge can handle draft pools without player IDs."""
    draft_pool = pd.DataFrame([{"player_display_name": "Rookie", "position": "RB"}])
    legacy = pd.DataFrame([{"player_id": 1, "total_pts": 10.0, "games_played_frac": 1.0}])

    merged = ScoringService().merge_roster_with_legacy(draft_pool, legacy)

    assert "player_id" in merged.columns


def test_merge_roster_with_legacy_assigns_new_rookie_ids_and_marks_original_rookies():
    """Verify rookies receive new IDs and remain marked as original rookies."""
    draft_pool = pd.DataFrame(
        [{"player_display_name": "Rookie", "position": "RB"}]
    )
    legacy = pd.DataFrame(
        [{"player_id": 12, "player_display_name": "Veteran", "position": "RB", "total_pts": 70.0, "games_played_frac": 1.0}]
    )

    merged = ScoringService().merge_roster_with_legacy(draft_pool, legacy)

    assert int(merged["player_id"].iloc[0]) == 13 and bool(merged["is_rookie_original"].iloc[0]) is True


def test_apply_rookie_metadata_replaces_games_played_with_rookie_marker():
    """Verify rookie metadata writes the R marker and drops the source flag."""
    df = pd.DataFrame([{"games_played_frac": 0.5, "is_rookie_original": True}])

    updated = ScoringService().apply_rookie_metadata(df)

    assert updated["games_played_frac"].iloc[0] == "R" and "is_rookie_original" not in updated.columns


def test_generate_weekly_projections_zeroes_out_the_bye_week():
    """Verify weekly projections set the bye week to zero points."""
    df = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": 20.0, "bye_week": 7}])

    projections = ScoringService().generate_weekly_projections(df)

    assert projections[1][7] == 0 and projections[1][6] == 20.0
