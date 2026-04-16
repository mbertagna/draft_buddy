"""Tests for simulator evaluator math and orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft_buddy.simulator.evaluator import (
    _get_playoffs_tree,
    _get_records_from_matchups_results_df,
    _optimal_lineup_points,
    _precompute_manager_weekly_points,
    _solve_matchups_single_thread,
    format_playoff_tree_string,
    generate_and_resolve_playoffs,
    generate_round_robin_schedule,
    simulate_season_fast,
)


@pytest.fixture
def two_team_matchups_df():
    """Return one-week two-team schedule."""
    return pd.DataFrame(
        [
            {
                "Week": 1,
                "Matchup": 1,
                "Away Team": None,
                "Away Manager(s)": "Team A",
                "Away Score": 0.0,
                "Home Score": 0.0,
                "Home Team": None,
                "Home Manager(s)": "Team B",
            }
        ]
    )


@pytest.fixture
def two_team_precomputed_weekly_points():
    """Return precomputed weekly points for two teams."""
    return {
        "Team A": [{"QB": [18.0], "RB": [12.0, 10.0], "WR": [11.0, 9.0], "TE": [7.0]}],
        "Team B": [{"QB": [20.0], "RB": [14.0, 13.0], "WR": [8.0, 6.0], "TE": [5.0]}],
    }


@pytest.fixture
def ten_team_names():
    """Return 10 deterministic manager names."""
    return [f"Team {index}" for index in range(1, 11)]


@pytest.fixture
def regular_records_six_team_seed_order():
    """Return six teams already sorted by seed."""
    return [
        ("Seed1", {"W": 10, "L": 3, "T": 0, "pts": 1400.0}),
        ("Seed2", {"W": 9, "L": 4, "T": 0, "pts": 1380.0}),
        ("Seed3", {"W": 8, "L": 5, "T": 0, "pts": 1320.0}),
        ("Seed4", {"W": 7, "L": 6, "T": 0, "pts": 1300.0}),
        ("Seed5", {"W": 6, "L": 7, "T": 0, "pts": 1260.0}),
        ("Seed6", {"W": 5, "L": 8, "T": 0, "pts": 1200.0}),
    ]


@pytest.fixture
def playoff_precomputed_points():
    """Return deterministic multi-week precomputed points by seed."""
    def weekly(value):
        return [{"QB": [value], "RB": [0.0, 0.0], "WR": [0.0, 0.0], "TE": [0.0]}]

    return {
        "Seed1": weekly(80.0) * 20,
        "Seed2": weekly(70.0) * 20,
        "Seed3": weekly(60.0) * 20,
        "Seed4": weekly(20.0) * 20,
        "Seed5": weekly(30.0) * 20,
        "Seed6": weekly(10.0) * 20,
    }


def test_optimal_lineup_points_sums_required_starters_and_single_best_flex():
    """Verify lineup sum uses starters and one top flex candidate."""
    points = _optimal_lineup_points(
        {
            "QB": [30.0],
            "RB": [20.0, 18.0, 15.0],
            "WR": [17.0, 16.0, 14.0],
            "TE": [12.0, 10.0],
        }
    )

    assert points == 128.0


def test_optimal_lineup_points_uses_highest_remaining_flex_across_positions():
    """Verify flex picks the highest leftover across RB/WR/TE."""
    points = _optimal_lineup_points(
        {
            "QB": [25.0],
            "RB": [20.0, 19.0, 8.0],
            "WR": [18.0, 17.0, 16.0, 5.0],
            "TE": [12.0, 11.0],
        }
    )

    assert points == 127.0


def test_optimal_lineup_points_handles_missing_required_position_without_index_error():
    """Verify missing required positions simply contribute zero."""
    points = _optimal_lineup_points({"QB": [], "RB": [10.0], "WR": [9.0], "TE": []})

    assert points == 19.0


def test_solve_matchups_single_thread_sets_away_score_column(
    two_team_matchups_df, two_team_precomputed_weekly_points
):
    """Verify away score is calculated for a resolved matchup."""
    result = _solve_matchups_single_thread(two_team_precomputed_weekly_points, two_team_matchups_df)

    assert float(result.iloc[0]["Away Score"]) == 67.0


def test_solve_matchups_single_thread_sets_home_score_column(
    two_team_matchups_df, two_team_precomputed_weekly_points
):
    """Verify home score is calculated for a resolved matchup."""
    result = _solve_matchups_single_thread(two_team_precomputed_weekly_points, two_team_matchups_df)

    assert float(result.iloc[0]["Home Score"]) == 66.0


def test_solve_matchups_single_thread_returns_zero_when_week_is_out_of_range():
    """Verify score defaults to zero when requested week is unavailable."""
    matchups = pd.DataFrame([{"Week": 3, "Away Manager(s)": "Team A", "Home Manager(s)": "Team B"}])
    points = {"Team A": [{"QB": [10.0], "RB": [], "WR": [], "TE": []}], "Team B": []}
    result = _solve_matchups_single_thread(points, matchups)

    assert float(result.iloc[0]["Away Score"]) == 0.0


def test_solve_matchups_single_thread_keeps_row_unchanged_when_manager_is_missing():
    """Verify NaN manager rows bypass score computation."""
    matchups = pd.DataFrame(
        [{"Week": 1, "Away Manager(s)": np.nan, "Home Manager(s)": "Team B", "Away Score": 0.0, "Home Score": 0.0}]
    )
    result = _solve_matchups_single_thread({}, matchups)

    assert pd.isna(result.iloc[0]["Away Manager(s)"])


def test_get_records_from_matchups_results_df_aggregates_wins_losses_ties_and_points():
    """Verify records include expected aggregate metrics."""
    dataframe = pd.DataFrame(
        [
            {"Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 100.0, "Home Score": 90.0},
            {"Away Manager(s)": "A", "Home Manager(s)": "C", "Away Score": 80.0, "Home Score": 80.0},
            {"Away Manager(s)": "B", "Home Manager(s)": "C", "Away Score": 70.0, "Home Score": 95.0},
        ]
    )
    records = dict(_get_records_from_matchups_results_df(dataframe))

    assert records["A"] == {"W": 1, "L": 0, "T": 1, "pts": 180.0}


def test_get_records_from_matchups_results_df_sorts_by_wins_then_ties_then_points():
    """Verify standings tie-break by total points after W/T."""
    dataframe = pd.DataFrame(
        [
            {"Away Manager(s)": "A", "Home Manager(s)": "C", "Away Score": 100.0, "Home Score": 90.0},
            {"Away Manager(s)": "B", "Home Manager(s)": "C", "Away Score": 95.0, "Home Score": 85.0},
            {"Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 80.0, "Home Score": 80.0},
        ]
    )
    ordered = _get_records_from_matchups_results_df(dataframe)

    assert ordered[0][0] == "A"


def test_generate_round_robin_schedule_has_expected_matchup_count_for_10_teams_14_weeks(ten_team_names):
    """Verify 10-team 14-week schedule has 70 total matchups."""
    schedule = generate_round_robin_schedule(ten_team_names, num_weeks=14)

    assert len(schedule) == 70


def test_generate_round_robin_schedule_handles_odd_team_count_without_bye_rows():
    """Verify generated rows exclude explicit __BYE__ matchups."""
    schedule = generate_round_robin_schedule([f"T{index}" for index in range(1, 10)], num_weeks=14)
    managers = set(schedule["Away Manager(s)"].astype(str)).union(set(schedule["Home Manager(s)"].astype(str)))

    assert "__BYE__" not in managers


def test_generate_round_robin_schedule_returns_empty_dataframe_for_single_team():
    """Verify one-team input yields an empty schedule."""
    schedule = generate_round_robin_schedule(["Solo"], num_weeks=14)

    assert schedule.empty is True


def test_generate_round_robin_schedule_stops_after_requested_weeks_when_less_than_full_round_robin():
    """Verify schedule generation respects num_weeks cap."""
    schedule = generate_round_robin_schedule(["A", "B", "C", "D", "E", "F"], num_weeks=2)

    assert int(schedule["Week"].max()) == 2


def test_generate_and_resolve_playoffs_first_round_excludes_top_two_seeds_as_byes(
    playoff_precomputed_points, regular_records_six_team_seed_order
):
    """Verify seed1 and seed2 skip first playoff round."""
    playoffs = generate_and_resolve_playoffs(
        playoff_precomputed_points, regular_records_six_team_seed_order, 6, start_week=15
    )
    first_round_teams = set(playoffs[playoffs["Week"] == 15]["Home Manager(s)"]).union(
        set(playoffs[playoffs["Week"] == 15]["Away Manager(s)"])
    )

    assert "Seed1" not in first_round_teams


def test_generate_and_resolve_playoffs_first_round_excludes_seed2_as_bye(
    playoff_precomputed_points, regular_records_six_team_seed_order
):
    """Verify seed2 also skips first playoff round."""
    playoffs = generate_and_resolve_playoffs(
        playoff_precomputed_points, regular_records_six_team_seed_order, 6, start_week=15
    )
    first_round_teams = set(playoffs[playoffs["Week"] == 15]["Home Manager(s)"]).union(
        set(playoffs[playoffs["Week"] == 15]["Away Manager(s)"])
    )

    assert "Seed2" not in first_round_teams


def test_generate_and_resolve_playoffs_bubbles_highest_projected_seed_to_champion(
    playoff_precomputed_points, regular_records_six_team_seed_order
):
    """Verify highest scoring seed wins championship path."""
    playoffs = generate_and_resolve_playoffs(
        playoff_precomputed_points, regular_records_six_team_seed_order, 6, start_week=15
    )
    _, winner = _get_playoffs_tree(playoffs)

    assert winner == "Seed1"


def test_generate_and_resolve_playoffs_returns_empty_for_single_seed():
    """Verify playoff generation returns empty table when only one seed exists."""
    playoffs = generate_and_resolve_playoffs({}, [("Only", {"W": 1, "L": 0, "T": 0, "pts": 10.0})], 1, 15)

    assert playoffs.empty is True


def test_format_playoff_tree_string_includes_diagonal_connectors_when_lines_enabled():
    """Verify tree formatter renders slash connectors with line mode."""
    bracket = format_playoff_tree_string(["Champion", "Semi A", "Semi B"], lines=True)

    assert "/" in bracket


def test_get_playoffs_tree_returns_expected_winner_from_last_row_scores():
    """Verify winner extraction uses final matchup scores."""
    playoffs = pd.DataFrame(
        [
            {
                "Week": 15,
                "Matchup": 1,
                "Away Manager(s)": "Seed4",
                "Away Score": 100.0,
                "Home Score": 110.0,
                "Home Manager(s)": "Seed1",
            },
            {
                "Week": 16,
                "Matchup": 1,
                "Away Manager(s)": "Seed2",
                "Away Score": 99.0,
                "Home Score": 105.0,
                "Home Manager(s)": "Seed1",
            },
        ]
    )
    _, winner = _get_playoffs_tree(playoffs)

    assert winner == "Seed1"


def test_precompute_manager_weekly_points_sorts_points_descending_within_position():
    """Verify precompute sorts weekly positional lists descending."""
    weekly_projections = {
        1: {"pos": "RB", "pts": [5.0]},
        2: {"pos": "RB", "pts": [12.0]},
    }
    rosters = {"A": [1, 2]}
    precomputed = _precompute_manager_weekly_points(weekly_projections, rosters, max_week=1)

    assert precomputed["A"][0]["RB"] == [12.0, 5.0]


def test_precompute_manager_weekly_points_ignores_invalid_positions():
    """Verify unsupported positions are excluded from precomputed buckets."""
    weekly_projections = {1: {"pos": "K", "pts": [7.0]}}
    precomputed = _precompute_manager_weekly_points(weekly_projections, {"A": [1]}, max_week=1)

    assert precomputed["A"][0]["QB"] == []


def test_precompute_manager_weekly_points_converts_bad_point_values_to_zero():
    """Verify unparsable weekly point values become zero."""
    weekly_projections = {1: {"pos": "WR", "pts": ["bad-value"]}}
    precomputed = _precompute_manager_weekly_points(weekly_projections, {"A": [1]}, max_week=1)

    assert precomputed["A"][0]["WR"] == [0.0]


def test_precompute_manager_weekly_points_skips_missing_player_projection_records():
    """Verify unknown player ids in rosters are ignored."""
    precomputed = _precompute_manager_weekly_points({}, {"A": [999]}, max_week=1)

    assert precomputed["A"][0]["QB"] == []


def test_precompute_manager_weekly_points_pads_missing_future_weeks_with_zero():
    """Verify missing projection weeks default to zero points."""
    weekly_projections = {1: {"pos": "WR", "pts": [6.0]}}
    precomputed = _precompute_manager_weekly_points(weekly_projections, {"A": [1]}, max_week=2)

    assert precomputed["A"][1]["WR"] == [0.0]


def test_simulate_season_fast_runs_full_pipeline_and_returns_expected_winner():
    """Verify full season simulation returns deterministic champion."""
    weekly_projections = {
        1: {"pos": "QB", "pts": [30.0] * 18},
        2: {"pos": "QB", "pts": [20.0] * 18},
        3: {"pos": "QB", "pts": [10.0] * 18},
        4: {"pos": "QB", "pts": [5.0] * 18},
    }
    rosters = {"A": [1], "B": [2], "C": [3], "D": [4]}
    matchups = pd.DataFrame(
        [
            {"Week": 1, "Away Manager(s)": "A", "Home Manager(s)": "B"},
            {"Week": 1, "Away Manager(s)": "C", "Home Manager(s)": "D"},
            {"Week": 2, "Away Manager(s)": "A", "Home Manager(s)": "C"},
            {"Week": 2, "Away Manager(s)": "B", "Home Manager(s)": "D"},
            {"Week": 3, "Away Manager(s)": "A", "Home Manager(s)": "D"},
            {"Week": 3, "Away Manager(s)": "B", "Home Manager(s)": "C"},
        ]
    )
    _, _, _, _, winner = simulate_season_fast(
        weekly_projections, matchups, rosters, season=2025, save_data=False, num_playoff_teams=2
    )

    assert winner == "A"


def test_simulate_season_fast_writes_csv_outputs_when_save_data_enabled(tmp_path):
    """Verify save_data option writes regular and playoff CSV files."""
    weekly_projections = {1: {"pos": "QB", "pts": [10.0] * 18}, 2: {"pos": "QB", "pts": [9.0] * 18}}
    rosters = {"A": [1], "B": [2]}
    matchups = pd.DataFrame([{"Week": 1, "Away Manager(s)": "A", "Home Manager(s)": "B"}])
    prefix = str(Path(tmp_path) / "season_out")
    simulate_season_fast(
        weekly_projections,
        matchups,
        rosters,
        season=2025,
        output_file_prefix=prefix,
        save_data=True,
        num_playoff_teams=2,
    )

    assert Path(f"{prefix}_regular.csv").exists()


def test_simulate_season_fast_writes_playoff_csv_when_save_data_enabled(tmp_path):
    """Verify save_data option writes playoff CSV file."""
    weekly_projections = {1: {"pos": "QB", "pts": [10.0] * 18}, 2: {"pos": "QB", "pts": [9.0] * 18}}
    rosters = {"A": [1], "B": [2]}
    matchups = pd.DataFrame([{"Week": 1, "Away Manager(s)": "A", "Home Manager(s)": "B"}])
    prefix = str(Path(tmp_path) / "season_out")
    simulate_season_fast(
        weekly_projections,
        matchups,
        rosters,
        season=2025,
        output_file_prefix=prefix,
        save_data=True,
        num_playoff_teams=2,
    )

    assert Path(f"{prefix}_playoff.csv").exists()
