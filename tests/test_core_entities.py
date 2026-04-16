"""Tests for canonical core entities."""

from __future__ import annotations

import pytest

from draft_buddy.core import Pick, TeamRoster


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
