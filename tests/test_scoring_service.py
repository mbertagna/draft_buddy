"""Tests for scoring service behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from draft_buddy.data.scoring.engine import ScoringEngine
from draft_buddy.data.scoring.service import ScoringService, _weekly_projections_from_draft_players


def test_weekly_projections_zero_out_bye_week() -> None:
    """Verify bye weeks are projected as zero points."""
    dataframe = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": 10.0, "bye_week": 2}])

    assert _weekly_projections_from_draft_players(dataframe)[1][2] == 0


def test_weekly_projections_treat_nan_total_points_as_zero() -> None:
    """Verify missing total points become zero-valued weekly projections."""
    dataframe = pd.DataFrame([{"player_id": 1, "position": "QB", "total_pts": float("nan"), "bye_week": None}])

    assert _weekly_projections_from_draft_players(dataframe)[1][1] == 0


def test_apply_scoring_returns_empty_frame_with_total_pts_column() -> None:
    """Verify empty input produces an empty frame with total_pts."""
    result = ScoringService().apply_scoring(pd.DataFrame())

    assert "total_pts" in result.columns and result.empty


def test_calculate_games_played_frac_returns_empty_frame_for_empty_input() -> None:
    """Verify empty history yields an empty fraction frame."""
    result = ScoringService().calculate_games_played_frac(pd.DataFrame())

    assert list(result.columns) == ["player_id", "games_played_frac"]


def test_calculate_games_played_frac_divides_player_games_by_team_games() -> None:
    """Verify games played fraction uses per-team season game totals."""
    historical = pd.DataFrame(
        [
            {"player_id": 1, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 1, "season": 2023, "recent_team": "BUF", "week": 2},
            {"player_id": 2, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 3, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 3, "season": 2023, "recent_team": "BUF", "week": 2},
        ]
    )

    result = ScoringService().calculate_games_played_frac(historical)

    assert float(result.loc[result["player_id"] == 1, "games_played_frac"].iloc[0]) == 1.0


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


def test_merge_roster_with_legacy_falls_back_to_player_id_merge() -> None:
    """Verify legacy merge falls back to player_id when name metadata is missing."""
    draft_pool_df = pd.DataFrame(
        [{"player_id": 11, "player_display_name": "Vet", "position": "QB"}]
    )
    legacy_stats_df = pd.DataFrame(
        [{"player_id": 11, "total_pts": 10.0, "games_played_frac": 0.8}]
    )

    result = ScoringService().merge_roster_with_legacy(draft_pool_df, legacy_stats_df)

    assert float(result.iloc[0]["total_pts"]) == 10.0 and bool(result.iloc[0]["is_rookie_original"]) is False


def test_merge_roster_with_legacy_assigns_rookie_ids_when_draft_pool_has_no_player_id() -> None:
    """Verify rookies receive generated ids even when the draft pool lacks player_id."""
    draft_pool_df = pd.DataFrame([{"player_display_name": "New Player", "position": "RB"}])
    legacy_stats_df = pd.DataFrame(columns=["player_id", "total_pts", "games_played_frac"])

    result = ScoringService().merge_roster_with_legacy(draft_pool_df, legacy_stats_df)

    assert int(result.iloc[0]["player_id"]) == 1 and bool(result.iloc[0]["is_rookie_original"]) is True


def test_aggregate_legacy_stats_uses_mean_when_requested() -> None:
    """Verify legacy aggregation switches to mean when requested."""
    historical = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 10.0, "season": 2023, "player_display_name": "A", "position": "QB", "recent_team": "BUF", "week": 1},
            {"player_id": 1, "total_pts": 16.0, "season": 2024, "player_display_name": "A", "position": "QB", "recent_team": "BUF", "week": 2},
        ]
    )

    result = ScoringService().aggregate_legacy_stats(historical, measure_of_center="mean")

    assert float(result.iloc[0]["total_pts"]) == 13.0


def test_aggregate_legacy_stats_tolerates_missing_metadata_columns() -> None:
    """Verify aggregation still returns totals when metadata extraction fails."""
    historical = pd.DataFrame(
        [
            {"player_id": 1, "total_pts": 10.0, "season": 2023, "recent_team": "BUF", "week": 1},
            {"player_id": 1, "total_pts": 12.0, "season": 2024, "recent_team": "BUF", "week": 2},
        ]
    )

    result = ScoringService().aggregate_legacy_stats(historical)

    assert list(result.columns) == ["player_id", "total_pts", "games_played_frac"]


def test_apply_rookie_metadata_marks_rookies_with_r() -> None:
    """Verify rookie metadata replaces games_played_frac with R."""
    dataframe = pd.DataFrame([{"games_played_frac": 0.5, "is_rookie_original": True}])
    result = ScoringService().apply_rookie_metadata(dataframe)

    assert result.iloc[0]["games_played_frac"] == "R"


def test_apply_rookie_metadata_drops_marker_even_without_games_fraction() -> None:
    """Verify rookie marker is removed even when games_played_frac is absent."""
    dataframe = pd.DataFrame([{"player_id": 1, "is_rookie_original": False}])

    result = ScoringService().apply_rookie_metadata(dataframe)

    assert "is_rookie_original" not in result.columns


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


def test_finalize_draft_players_ignores_missing_optional_columns() -> None:
    """Verify finalize keeps only columns that actually exist."""
    dataframe = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "A", "position": "QB", "total_pts": 1.0}]
    )

    result = ScoringService().finalize_draft_players(dataframe)

    assert list(result.columns) == ["player_id", "player_display_name", "position", "total_pts"]


def test_merge_draft_year_with_legacy_joins_by_player_id() -> None:
    """Verify draft-year merge preserves draft metadata while adding legacy fields."""
    draft_year = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "A", "position": "QB", "recent_team": "BUF"}]
    )
    legacy = pd.DataFrame([{"player_id": 1, "total_pts": 20.0, "games_played_frac": 0.9}])

    result = ScoringService().merge_draft_year_with_legacy(draft_year, legacy)

    assert float(result.iloc[0]["total_pts"]) == 20.0


def test_generate_weekly_projections_delegates_to_weekly_projection_builder() -> None:
    """Verify service weekly projections reuse the shared helper behavior."""
    dataframe = pd.DataFrame([{"player_id": 3, "position": "WR", "total_pts": 8.0, "bye_week": 4}])

    result = ScoringService().generate_weekly_projections(dataframe)

    assert result[3][4] == 0 and result[3][5] == 8.0


def test_prepare_offense_kicking_features_creates_total_fumbles_lost() -> None:
    """Verify composite total_fumbles_lost is derived from component columns."""
    result = ScoringEngine.prepare_offense_kicking_features(
        pd.DataFrame([{"sack_fumbles_lost": 1, "rushing_fumbles_lost": 2, "receiving_fumbles_lost": 3}])
    )

    assert int(result.iloc[0]["total_fumbles_lost"]) == 6


def test_prepare_offense_kicking_features_derives_fg_buckets_and_yards_from_list() -> None:
    """Verify kicker list strings create FG buckets and total made yards."""
    result = ScoringEngine.prepare_offense_kicking_features(
        pd.DataFrame([{"fg_made_list": "18;27;36;48;55;61", "pat_made": 2, "pat_missed": 1}])
    )

    assert (
        int(result.iloc[0]["fg_made_0_39"]) == 3
        and int(result.iloc[0]["fg_made_40_49"]) == 1
        and int(result.iloc[0]["fg_made_50_59"]) == 1
        and int(result.iloc[0]["fg_made_60_"]) == 1
        and int(result.iloc[0]["fg_made_yards"]) == 245
        and int(result.iloc[0]["xp_made"]) == 2
    )


def test_apply_scoring_adds_linear_stat_points() -> None:
    """Verify apply_scoring sums configured linear scoring rules."""
    result = ScoringEngine.apply_scoring(
        pd.DataFrame([{"passing_yards": 250, "passing_tds": 2}]),
        {"passing_yards": 0.04, "passing_tds": 4},
    )

    assert float(result.iloc[0]["total_pts"]) == 18.0


def test_apply_scoring_prefers_touchdown_synonyms_without_double_counting() -> None:
    """Verify normalized rule aliases do not double-count touchdown columns."""
    result = ScoringEngine.apply_scoring(
        pd.DataFrame([{"passing_touchdowns": 2, "passing_tds": 2}]),
        {"passing_touchdowns": 4, "passing_tds": 6},
    )

    assert float(result.iloc[0]["total_pts"]) == 8.0


def test_apply_scoring_prefers_fg_yards_over_bucket_rules() -> None:
    """Verify yardage-based FG scoring suppresses tiered bucket double counting."""
    result = ScoringEngine.apply_scoring(
        pd.DataFrame([{"fg_made_list": "50"}]),
        {"fg_made_yards": 0.1, "fg_made_50_59": 5},
    )

    assert float(result.iloc[0]["total_pts"]) == 5.0


def test_apply_scoring_prefers_specific_fumble_components_over_total() -> None:
    """Verify specific fumble penalties suppress the aggregate total_fumbles_lost rule."""
    result = ScoringEngine.apply_scoring(
        pd.DataFrame([{"sack_fumbles_lost": 1, "rushing_fumbles_lost": 1}]),
        {"total_fumbles_lost": -2, "sack_fumbles_lost": -1, "rushing_fumbles_lost": -1},
    )

    assert float(result.iloc[0]["total_pts"]) == -2.0


def test_apply_team_def_scoring_uses_points_allowed_tiers_and_specific_return_touchdowns() -> None:
    """Verify defense scoring applies tier bonuses and avoids aggregate return TD double counting."""
    result = ScoringEngine.apply_team_def_scoring(
        pd.DataFrame(
            [
                {
                    "points_allowed": 0,
                    "sacks": 2,
                    "kick_return_touchdowns": 1,
                    "punt_return_touchdowns": 1,
                    "st_def_td": 2,
                }
            ]
        ),
        {
            "sacks": 1,
            "kick_return_touchdowns": 6,
            "punt_return_touchdowns": 6,
            "st_def_td": 6,
            "def_points_allowed_0": 10,
        },
    )

    assert float(result.iloc[0]["def_total_pts"]) == 24.0


def test_compute_total_made_fg_yards_approximates_from_bucket_columns() -> None:
    """Verify FG yardage falls back to bucket midpoints when lists are absent."""
    dataframe = pd.DataFrame(
        [{"fg_made_0_19": 1, "fg_made_20_29": 1, "fg_made_30_39": 0, "fg_made_40_49": 0, "fg_made_50_59": 1, "fg_made_60_": 0}]
    )

    result = ScoringEngine._compute_total_made_fg_yards(dataframe)

    assert float(result.iloc[0]) == 90.0


def test_example_scoring_rows_match_reported_totals() -> None:
    """Verify the example helpers return totals that match the scored rows."""
    offense_df, offense_total = ScoringEngine.example_offense_scoring_row()
    kicking_df, kicking_total = ScoringEngine.example_kicking_scoring_row()

    assert offense_total == pytest.approx(float(offense_df.iloc[0]["total_pts"])) and kicking_total == pytest.approx(
        float(kicking_df.iloc[0]["total_pts"])
    )
