"""Tests for domain-level Player serialization behavior."""

import numpy as np

from draft_buddy.domain.entities import Player


def test_player_to_dict_serializes_flat_standard_fields():
    """Verify standard player fields serialize as expected."""
    player = Player(
        player_id=101,
        name="Sample Player",
        position="RB",
        projected_points=180.5,
        games_played_frac=0.9,
        adp=24.0,
        bye_week=9,
        team="GB",
    )

    assert player.to_dict() == {
        "player_id": 101,
        "name": "Sample Player",
        "position": "RB",
        "projected_points": 180.5,
        "games_played_frac": 0.9,
        "adp": 24.0,
        "bye_week": 9,
        "team": "GB",
    }


def test_player_to_dict_converts_infinite_adp_to_none():
    """Verify infinite ADP serializes to None."""
    player = Player(player_id=102, name="No ADP", position="WR", projected_points=140.0, adp=np.inf)

    assert player.to_dict()["adp"] is None


def test_player_to_dict_preserves_none_bye_week():
    """Verify missing bye week remains None in serialized output."""
    player = Player(player_id=103, name="No Bye", position="TE", projected_points=120.0, bye_week=None)

    assert player.to_dict()["bye_week"] is None
