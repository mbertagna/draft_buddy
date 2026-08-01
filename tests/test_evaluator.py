"""Tests for simulator evaluator helpers."""

from __future__ import annotations

from pathlib import Path

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


def test_optimal_lineup_points_uses_best_remaining_flex() -> None:
    """Verify the flex spot uses the best leftover RB/WR/TE score."""
    points = _optimal_lineup_points({"QB": [10.0], "RB": [9.0, 8.0, 7.0], "WR": [6.0, 5.0], "TE": [4.0]})

    assert points == 49.0


def test_optimal_lineup_points_handles_missing_required_positions() -> None:
    """Verify missing starter slots contribute zero instead of failing."""
    points = _optimal_lineup_points({"RB": [9.0], "WR": [8.0], "TE": []})

    assert points == 17.0


def test_solve_matchups_single_thread_skips_rows_with_missing_manager() -> None:
    """Verify missing-manager rows are preserved without score computation."""
    matchups = pd.DataFrame([{"Week": 1, "Away Manager(s)": None, "Home Manager(s)": "Team B", "Away Score": 0.0, "Home Score": 0.0}])
    result = _solve_matchups_single_thread({}, matchups)

    assert pd.isna(result.iloc[0]["Away Manager(s)"])


def test_solve_matchups_single_thread_uses_zero_when_week_is_out_of_range() -> None:
    """Verify teams without enough weekly projections receive zero for that week."""
    precomputed = {
        "A": [{"QB": [10.0], "RB": [], "WR": [], "TE": []}],
        "B": [{"QB": [9.0], "RB": [], "WR": [], "TE": []}],
    }
    matchups = pd.DataFrame(
        [{"Week": 2, "Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 1.0, "Home Score": 1.0}]
    )

    result = _solve_matchups_single_thread(precomputed, matchups)

    assert float(result.iloc[0]["Away Score"]) == 0.0 and float(result.iloc[0]["Home Score"]) == 0.0


def test_get_records_from_matchups_results_df_tracks_ties() -> None:
    """Verify tied matchups increment tie counts for both teams."""
    dataframe = pd.DataFrame([{"Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 10.0, "Home Score": 10.0}])
    records = dict(_get_records_from_matchups_results_df(dataframe))

    assert records["A"]["T"] == 1 and records["B"]["T"] == 1


def test_get_records_from_matchups_results_df_sorts_by_wins_then_points() -> None:
    """Verify record ordering uses wins and total points as tiebreakers."""
    dataframe = pd.DataFrame(
        [
            {"Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 11.0, "Home Score": 9.0},
            {"Away Manager(s)": "C", "Home Manager(s)": "D", "Away Score": 8.0, "Home Score": 6.0},
            {"Away Manager(s)": "A", "Home Manager(s)": "C", "Away Score": 7.0, "Home Score": 5.0},
            {"Away Manager(s)": "B", "Home Manager(s)": "D", "Away Score": 12.0, "Home Score": 10.0},
        ]
    )

    records = _get_records_from_matchups_results_df(dataframe)

    assert [name for name, _record in records[:2]] == ["A", "B"]


def test_generate_round_robin_schedule_returns_empty_for_single_team() -> None:
    """Verify one-team schedules produce an empty dataframe."""
    assert generate_round_robin_schedule(["Solo"], 3).empty


def test_generate_round_robin_schedule_skips_bye_matchups_for_odd_team_count(monkeypatch) -> None:
    """Verify odd-team schedules omit BYE placeholder rows."""
    monkeypatch.setattr("random.shuffle", lambda values: None)
    monkeypatch.setattr("random.random", lambda: 0.0)

    schedule = generate_round_robin_schedule(["A", "B", "C"], 3)

    assert "__BYE__" not in set(schedule["Away Manager(s)"]) | set(schedule["Home Manager(s)"])


def test_generate_round_robin_schedule_adds_extra_weeks_after_unique_pairs(monkeypatch) -> None:
    """Verify extra requested weeks are filled from the remaining candidate pairs."""
    monkeypatch.setattr("random.shuffle", lambda values: None)
    monkeypatch.setattr("random.random", lambda: 0.0)

    schedule = generate_round_robin_schedule(["A", "B", "C", "D"], 4)

    assert sorted(schedule["Week"].unique().tolist()) == [1, 2, 3, 4]


def test_generate_and_resolve_playoffs_returns_empty_frame_for_single_seed() -> None:
    """Verify playoff generation with one seed returns an empty result."""
    assert generate_and_resolve_playoffs({}, [("Solo", {})], 1, 15).empty


def test_generate_and_resolve_playoffs_advances_higher_seed_on_tie() -> None:
    """Verify tied playoff games advance the home or higher-seeded side."""
    precomputed = {
        "A": [{"QB": [10.0], "RB": [8.0, 7.0], "WR": [6.0, 5.0], "TE": [4.0]}] * 3,
        "B": [{"QB": [10.0], "RB": [8.0, 7.0], "WR": [6.0, 5.0], "TE": [4.0]}] * 3,
    }

    playoffs = generate_and_resolve_playoffs(precomputed, [("A", {}), ("B", {})], 2, 15)

    assert playoffs.iloc[-1]["Home Manager(s)"] == "A" and playoffs.iloc[-1]["Home Score"] == playoffs.iloc[-1]["Away Score"]


def test_format_playoff_tree_string_contains_branch_lines() -> None:
    """Verify bracket rendering includes slash markers when line mode is enabled."""
    assert "/" in format_playoff_tree_string(["Champion", "Finalist", "Semifinalist"], lines=True)


def test_format_playoff_tree_string_without_lines_omits_branch_markers() -> None:
    """Verify line-free rendering omits slash markers."""
    assert "/" not in format_playoff_tree_string(["Champion", "Finalist", "Semifinalist"], lines=False)


def test_get_playoffs_tree_returns_winner_from_final_row() -> None:
    """Verify bracket conversion returns a printable tree and the final winner."""
    playoffs = pd.DataFrame(
        [
            {"Week": 15, "Matchup": 1, "Away Manager(s)": "B", "Away Score": 90.0, "Home Score": 100.0, "Home Manager(s)": "A"},
            {"Week": 16, "Matchup": 1, "Away Manager(s)": "C", "Away Score": 105.0, "Home Score": 99.0, "Home Manager(s)": "A"},
        ]
    )

    tree, winner = _get_playoffs_tree(playoffs)

    assert winner == "C" and "C (105.0)" in tree


def test_precompute_manager_weekly_points_ignores_non_skill_positions() -> None:
    """Verify positions outside QB/RB/WR/TE are ignored."""
    projections = {1: {"pos": "K", "pts": [3.0]}}
    rosters = {"Team": [1]}
    weekly_points = _precompute_manager_weekly_points(projections, rosters, 1)

    assert weekly_points["Team"][0]["QB"] == []


def test_precompute_manager_weekly_points_coerces_bad_projection_values_to_zero() -> None:
    """Verify invalid weekly values are treated as zero during precomputation."""
    projections = {1: {"pos": "RB", "pts": ["bad", 4.0]}, 2: {"pos": "RB", "pts": [5.0, 6.0]}}
    rosters = {"Team": [1, 2, 999]}

    weekly_points = _precompute_manager_weekly_points(projections, rosters, 2)

    assert weekly_points["Team"][0]["RB"] == [5.0, 0.0]


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


def test_simulate_season_fast_saves_regular_and_playoff_csvs(tmp_path: Path) -> None:
    """Verify optional CSV persistence writes both output files."""
    weekly_projections = {
        1: {"pos": "QB", "pts": [10.0] * 18},
        2: {"pos": "RB", "pts": [9.0] * 18},
        3: {"pos": "WR", "pts": [8.0] * 18},
        4: {"pos": "TE", "pts": [7.0] * 18},
        5: {"pos": "QB", "pts": [6.0] * 18},
        6: {"pos": "RB", "pts": [5.0] * 18},
        7: {"pos": "WR", "pts": [4.0] * 18},
        8: {"pos": "TE", "pts": [3.0] * 18},
    }
    matchups = pd.DataFrame(
        [{"Week": 1, "Matchup": 1, "Away Manager(s)": "A", "Home Manager(s)": "B", "Away Score": 0.0, "Home Score": 0.0}]
    )
    rosters = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}
    output_prefix = tmp_path / "season"

    simulate_season_fast(
        weekly_projections,
        matchups,
        rosters,
        season=2025,
        output_file_prefix=str(output_prefix),
        save_data=True,
        num_playoff_teams=2,
    )

    assert (tmp_path / "season_regular.csv").exists() and (tmp_path / "season_playoff.csv").exists()
