"""Tests for draft assistant GP and ignore filters."""

from __future__ import annotations

from draft_buddy.core.entities import Player
from draft_buddy.web.draft_advisor_filter import (
    exclude_ignored_players,
    exclude_incomplete_players,
    passes_gp_filter,
)


def test_rookie_always_passes_gp_filter() -> None:
    """Verify rookies bypass the GP minimum."""
    assert passes_gp_filter("R", 0.75) is True


def test_empty_gp_min_passes_all_players() -> None:
    """Verify no threshold allows every finite GP value."""
    assert passes_gp_filter(0.2, None) is True


def test_gp_below_threshold_is_filtered() -> None:
    """Verify players below the threshold are excluded."""
    assert passes_gp_filter(0.5, 0.7) is False


def test_gp_at_threshold_passes() -> None:
    """Verify players at the threshold are included."""
    assert passes_gp_filter(0.7, 0.7) is True


def test_non_finite_gp_frac_is_filtered() -> None:
    """Verify invalid GP values are excluded when a threshold is set."""
    assert passes_gp_filter(None, 0.7) is False
    assert passes_gp_filter("bad", 0.7) is False


def test_exclude_ignored_players_removes_matching_ids() -> None:
    """Verify blinded player ids are dropped from the available pool."""
    players = [
        Player(player_id=1, name="A", position="RB", projected_points=100.0),
        Player(player_id=2, name="B", position="WR", projected_points=90.0),
        Player(player_id=3, name="C", position="TE", projected_points=80.0),
    ]

    remaining = exclude_ignored_players(players, [2, 99])

    assert [player.player_id for player in remaining] == [1, 3]


def test_exclude_ignored_players_noop_when_empty() -> None:
    """Verify an empty ignore list leaves the pool unchanged."""
    players = [
        Player(player_id=1, name="A", position="RB", projected_points=100.0),
    ]

    remaining = exclude_ignored_players(players, [])

    assert [player.player_id for player in remaining] == [1]


def test_exclude_incomplete_players_drops_placeholders() -> None:
    """Verify sleeper-only and non-skill players are excluded from advisor pools."""
    players = [
        Player(player_id=1, name="A", position="RB", projected_points=100.0),
        Player(
            player_id=2,
            name="B",
            position="RB",
            projected_points=0.0,
            data_completeness="sleeper_only",
        ),
        Player(
            player_id=3,
            name="DET",
            position="DEF",
            projected_points=0.0,
            data_completeness="sleeper_only",
        ),
    ]

    remaining = exclude_incomplete_players(players)

    assert [player.player_id for player in remaining] == [1]
