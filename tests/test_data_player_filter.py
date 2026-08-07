"""Tests for inactive-player exclusion filtering."""

from __future__ import annotations

from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.player_filter import exclude_inactive_players


def test_exclude_inactive_players_removes_matching_roster_status(player_factory) -> None:
    """Verify players with an excluded roster status are removed."""
    active = player_factory(1, "RB")
    inactive = player_factory(2, "RB")
    object.__setattr__(inactive, "sleeper_status", "Inactive")
    catalog = PlayerCatalog([active, inactive])

    filtered = exclude_inactive_players(catalog, roster_statuses=["Inactive"], injury_statuses=[])

    assert [player.player_id for player in filtered] == [1]


def test_exclude_inactive_players_removes_matching_injury_status(player_factory) -> None:
    """Verify players with an excluded injury status are removed."""
    healthy = player_factory(1, "WR")
    injured = player_factory(2, "WR")
    object.__setattr__(injured, "sleeper_injury_status", "IR")
    catalog = PlayerCatalog([healthy, injured])

    filtered = exclude_inactive_players(catalog, roster_statuses=[], injury_statuses=["IR"])

    assert [player.player_id for player in filtered] == [1]


def test_exclude_inactive_players_keeps_players_missing_status_data(player_factory) -> None:
    """Verify players without status data are never excluded."""
    player = player_factory(1, "TE")
    catalog = PlayerCatalog([player])

    filtered = exclude_inactive_players(
        catalog, roster_statuses=["Inactive"], injury_statuses=["IR"]
    )

    assert [player.player_id for player in filtered] == [1]
