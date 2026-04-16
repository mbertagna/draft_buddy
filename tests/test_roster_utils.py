"""Tests for roster slot allocation and scoring."""

from draft_buddy.core.roster_utils import calculate_roster_scores, categorize_roster_by_slots
from draft_buddy.domain.entities import Player


def test_categorize_roster_assigns_standard_lineup_to_starters_without_bench_spillover():
    """Verify a complete standard lineup fills starters with no bench players."""
    team_roster = [
        Player(1, "QB", "QB", 20.0),
        Player(2, "RB A", "RB", 18.0),
        Player(3, "RB B", "RB", 16.0),
        Player(4, "WR A", "WR", 17.0),
        Player(5, "WR B", "WR", 15.0),
        Player(6, "TE", "TE", 12.0),
    ]
    starters, bench, _ = categorize_roster_by_slots(
        team_roster,
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0},
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1},
    )

    assert len(bench) == 0


def test_categorize_roster_places_third_best_rb_in_flex_when_two_rb_slots_exist():
    """Verify optimal FLEX routing for three RBs."""
    team_roster = [
        Player(10, "RB Top", "RB", 25.0),
        Player(11, "RB Mid", "RB", 20.0),
        Player(12, "RB Flex", "RB", 15.0),
    ]
    starters, _, flex_players = categorize_roster_by_slots(
        team_roster,
        {"QB": 0, "RB": 2, "WR": 0, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
    )
    rb_starter_names = {player.name for player in starters["RB"]}

    assert flex_players[0].name == "RB Flex"


def test_categorize_roster_keeps_two_highest_rbs_in_rb_starter_slots():
    """Verify RB starter slots keep the two highest projected RB players."""
    team_roster = [
        Player(10, "RB Top", "RB", 25.0),
        Player(11, "RB Mid", "RB", 20.0),
        Player(12, "RB Flex", "RB", 15.0),
    ]
    starters, _, _ = categorize_roster_by_slots(
        team_roster,
        {"QB": 0, "RB": 2, "WR": 0, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
    )
    rb_starter_names = {player.name for player in starters["RB"]}

    assert rb_starter_names == {"RB Top", "RB Mid"}


def test_categorize_roster_spills_only_one_wr_to_bench_when_four_wrs_exist():
    """Verify WR allocation produces WR, FLEX, and one bench player."""
    team_roster = [
        Player(20, "WR Top", "WR", 24.0),
        Player(21, "WR Two", "WR", 21.0),
        Player(22, "WR Flex", "WR", 18.0),
        Player(23, "WR Bench", "WR", 10.0),
    ]
    starters, bench, _ = categorize_roster_by_slots(
        team_roster,
        {"QB": 0, "RB": 0, "WR": 2, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 1, "TE": 0},
    )

    assert len(bench) == 1


def test_categorize_roster_assigns_two_wrs_to_wr_slots_when_four_wrs_exist():
    """Verify two WR players fill WR starter slots."""
    team_roster = [
        Player(20, "WR Top", "WR", 24.0),
        Player(21, "WR Two", "WR", 21.0),
        Player(22, "WR Flex", "WR", 18.0),
        Player(23, "WR Bench", "WR", 10.0),
    ]
    starters, _, _ = categorize_roster_by_slots(
        team_roster,
        {"QB": 0, "RB": 0, "WR": 2, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 1, "TE": 0},
    )

    assert len(starters["WR"]) == 2


def test_categorize_roster_assigns_one_wr_to_flex_when_four_wrs_exist():
    """Verify one WR player is routed into FLEX."""
    team_roster = [
        Player(20, "WR Top", "WR", 24.0),
        Player(21, "WR Two", "WR", 21.0),
        Player(22, "WR Flex", "WR", 18.0),
        Player(23, "WR Bench", "WR", 10.0),
    ]
    starters, _, _ = categorize_roster_by_slots(
        team_roster,
        {"QB": 0, "RB": 0, "WR": 2, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 1, "TE": 0},
    )

    assert len(starters["FLEX"]) == 1


def test_calculate_roster_scores_sums_starters_and_bench_totals_correctly():
    """Verify score totals are calculated for starters and bench."""
    team_roster = [
        Player(30, "QB", "QB", 20.0),
        Player(31, "RB A", "RB", 18.0),
        Player(32, "RB B", "RB", 16.0),
        Player(33, "WR A", "WR", 17.0),
        Player(34, "WR B", "WR", 15.0),
        Player(35, "TE", "TE", 12.0),
        Player(36, "Bench WR", "WR", 8.0),
    ]
    scores = calculate_roster_scores(
        team_roster,
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0},
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1},
    )

    assert scores["starters_total_points"] == 98.0


def test_calculate_roster_scores_sums_bench_totals_correctly():
    """Verify bench total points include non-starters only."""
    team_roster = [
        Player(30, "QB", "QB", 20.0),
        Player(31, "RB A", "RB", 18.0),
        Player(32, "RB B", "RB", 16.0),
        Player(33, "WR A", "WR", 17.0),
        Player(34, "WR B", "WR", 15.0),
        Player(35, "TE", "TE", 12.0),
        Player(36, "Bench WR", "WR", 8.0),
    ]
    scores = calculate_roster_scores(
        team_roster,
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 0},
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1},
    )

    assert scores["bench_total_points"] == 8.0


def test_calculate_roster_scores_counts_flex_points_in_starter_total():
    """Verify FLEX player points are included in starter totals."""
    team_roster = [
        Player(40, "RB A", "RB", 20.0),
        Player(41, "RB B", "RB", 18.0),
        Player(42, "RB FLEX", "RB", 15.0),
    ]
    scores = calculate_roster_scores(
        team_roster,
        {"QB": 0, "RB": 2, "WR": 0, "TE": 0, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
    )

    assert scores["starters_total_points"] == 53.0


def test_calculate_roster_scores_keeps_over_limit_player_on_bench():
    """Verify over-limit bench player remains in bench scoring totals."""
    team_roster = [
        Player(50, "QB Starter", "QB", 20.0),
        Player(51, "QB Bench", "QB", 10.0),
        Player(52, "QB Extra", "QB", 8.0),
    ]
    scores = calculate_roster_scores(
        team_roster,
        {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        {"QB": 1, "RB": 0, "WR": 0, "TE": 0},
    )

    assert scores["bench_total_points"] == 18.0
