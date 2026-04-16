"""Tests for QB pass-catcher stacking calculations."""

from draft_buddy.core.stacking import calculate_stack_count
from draft_buddy.domain.entities import Player


def test_calculate_stack_count_returns_zero_for_empty_roster():
    """Verify no players yields zero stacks."""
    assert calculate_stack_count([]) == 0


def test_calculate_stack_count_returns_zero_without_team_match():
    """Verify no QB/WR-TE team match yields zero stacks."""
    roster = [
        Player(1, "QB BUF", "QB", 20.0, team="BUF"),
        Player(2, "WR MIA", "WR", 15.0, team="MIA"),
    ]

    assert calculate_stack_count(roster) == 0


def test_calculate_stack_count_returns_one_for_single_qb_wr_same_team():
    """Verify one QB-WR stack counts as one."""
    roster = [
        Player(3, "QB KC", "QB", 22.0, team="KC"),
        Player(4, "WR KC", "WR", 18.0, team="KC"),
    ]

    assert calculate_stack_count(roster) == 1


def test_calculate_stack_count_returns_two_for_qb_wr_te_same_team():
    """Verify one QB with WR and TE from same team yields two stacks."""
    roster = [
        Player(5, "QB CIN", "QB", 21.0, team="CIN"),
        Player(6, "WR CIN", "WR", 17.0, team="CIN"),
        Player(7, "TE CIN", "TE", 13.0, team="CIN"),
    ]

    assert calculate_stack_count(roster) == 2


def test_calculate_stack_count_ignores_players_without_team():
    """Verify players with missing team metadata are ignored."""
    roster = [
        Player(8, "QB Missing Team", "QB", 19.0, team=None),
        Player(9, "WR Missing Team", "WR", 14.0, team=None),
    ]

    assert calculate_stack_count(roster) == 0


def test_calculate_stack_count_ignores_non_stack_positions_on_same_team():
    """Verify same-team RB does not contribute to stack totals."""
    roster = [
        Player(10, "QB PHI", "QB", 24.0, team="PHI"),
        Player(11, "RB PHI", "RB", 17.0, team="PHI"),
    ]

    assert calculate_stack_count(roster) == 0
