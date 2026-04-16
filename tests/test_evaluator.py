"""Tests for simulator evaluator helpers."""

from __future__ import annotations

import pandas as pd

from draft_buddy.simulator.evaluator import (
    _get_records_from_matchups_results_df,
    _optimal_lineup_points,
    _precompute_manager_weekly_points,
    _solve_matchups_single_thread,
    format_playoff_tree_string,
    generate_and_resolve_playoffs,
    generate_round_robin_schedule,
    simulate_season_fast,
)


def test_optimal_lineup_points_uses_best_remaining_flex() -> None:
    """Verify the flex spot uses the best leftover RB/WR/TE score."""
    points = _optimal_lineup_points({"QB": [10.0], "RB": [9.0, 8.0, 7.0], "WR": [6.0, 5.0], "TE": [4.0]})

    assert points == 49.0


def test_solve_matchups_single_thread_skips_rows_with_missing_manager() -> None:
    """Verify missing-manager rows are preserved without score computation."""
    matchups = pd.DataFrame([{"Week": 1, "Away Manager(s)": None, "Home Manager(s)": "Team B", "Away Score": 0.0, "Home Score": 0.0}])
    result = _solve_matchups_single_thread({}, matchups)

    assert pd.isna(result.iloc[0]["Away Manager(s)"])


def test_get_records_from_matchups_results_df_tracks_ties() -> None:
    """Verify tied matchups increment tie counts for both teams."""
    dataframe = pd.DataFrame([{"Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 10.0, "Home Score": 10.0}])
    records = dict(_get_records_from_matchups_results_df(dataframe))

    assert records["A"]["T"] == 1 and records["B"]["T"] == 1


def test_generate_round_robin_schedule_returns_empty_for_single_team() -> None:
    """Verify one-team schedules produce an empty dataframe."""
    assert generate_round_robin_schedule(["Solo"], 3).empty


def test_generate_and_resolve_playoffs_returns_empty_frame_for_single_seed() -> None:
    """Verify playoff generation with one seed returns an empty result."""
    assert generate_and_resolve_playoffs({}, [("Solo", {})], 1, 15).empty


def test_format_playoff_tree_string_contains_branch_lines() -> None:
    """Verify bracket rendering includes slash markers when line mode is enabled."""
    assert "/" in format_playoff_tree_string(["Champion", "Finalist", "Semifinalist"], lines=True)


def test_precompute_manager_weekly_points_ignores_non_skill_positions() -> None:
    """Verify positions outside QB/RB/WR/TE are ignored."""
    projections = {1: {"pos": "K", "pts": [3.0]}}
    rosters = {"Team": [1]}
    weekly_points = _precompute_manager_weekly_points(projections, rosters, 1)

    assert weekly_points["Team"][0]["QB"] == []


def test_simulate_season_fast_returns_regular_results_and_winner() -> None:
    """Verify season simulation returns resolved tables and a winner."""
    weekly_projections = {
        1: {"pos": "QB", "pts": [10.0] * 18},
        2: {"pos": "RB", "pts": [9.0] * 18},
        3: {"pos": "QB", "pts": [8.0] * 18},
        4: {"pos": "RB", "pts": [7.0] * 18},
    }
    matchups = pd.DataFrame([{"Week": 1, "Matchup": 1, "Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 0.0, "Home Score": 0.0}])
    rosters = {"A": [1, 2], "B": [3, 4]}
    regular_results, regular_records, playoff_results, playoff_tree, winner = simulate_season_fast(
        weekly_projections,
        matchups,
        rosters,
        season=2025,
        num_playoff_teams=2,
    )

    assert not regular_results.empty and regular_records and not playoff_results.empty and playoff_tree and winner in {"A", "B"}
