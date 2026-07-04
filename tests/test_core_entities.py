"""Tests for canonical core entities."""

from __future__ import annotations

import pytest

from draft_buddy.core import Pick, Player, TeamRoster


def test_player_catalog_preserves_input_order(player_catalog) -> None:
    """Verify catalog iteration order is stable."""
    catalog_ids = [player.player_id for player in player_catalog]

    assert catalog_ids == list(player_catalog.player_ids)


def test_player_catalog_require_raises_for_missing_player(player_catalog) -> None:
    """Verify missing player ids raise a descriptive error."""
    with pytest.raises(KeyError, match="Unknown player id"):
        player_catalog.require(9999)


def test_team_roster_round_trips_through_dict() -> None:
    """Verify typed roster serialization preserves ids and counts."""
    roster = TeamRoster(player_ids=[1, 2, 3], qb_count=1, rb_count=1, wr_count=1, te_count=0, flex_count=1)
    restored = TeamRoster.from_dict(roster.to_dict())

    assert restored.to_dict() == roster.to_dict()


def test_player_to_dict_defaults_sleeper_fields_to_none() -> None:
    """Verify players without Sleeper data serialize null sleeper_* fields."""
    player = Player(player_id=1, name="A", position="QB", projected_points=100.0)

    assert player.to_dict()["sleeper_status"] is None


def test_player_to_dict_includes_populated_sleeper_fields() -> None:
    """Verify Sleeper-derived fields serialize when populated."""
    player = Player(
        player_id=1,
        name="A",
        position="QB",
        projected_points=100.0,
        sleeper_id="123",
        sleeper_status="Active",
        sleeper_injury_status="Questionable",
        sleeper_depth_chart_position="QB",
    )

    assert player.to_dict()["sleeper_injury_status"] == "Questionable"


def test_pick_round_trips_through_dict() -> None:
    """Verify typed pick serialization is lossless."""
    pick = Pick(
        pick_number=3,
        team_id=2,
        player_id=7,
        is_manual_pick=True,
        previous_pick_index=2,
        previous_override_team_id=4,
    )

    assert Pick.from_dict(pick.to_dict()) == pick
