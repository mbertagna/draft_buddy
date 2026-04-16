"""Tests for ID-based draft state behavior."""

from __future__ import annotations

import json

from draft_buddy.core import Pick


def test_draft_state_tracks_ids_and_roster_counts(draft_state, player_catalog) -> None:
    """Verify state stores player ids and typed counts instead of player objects."""
    draft_state.add_player_to_roster(1, player_catalog.require(2))
    draft_state.add_player_to_roster(1, player_catalog.require(3))
    draft_state.add_player_to_roster(1, player_catalog.require(6))
    roster = draft_state.roster_for_team(1)

    assert roster.player_ids == [2, 3, 6] and roster.rb_count == 1 and roster.wr_count == 1 and roster.flex_count == 1


def test_draft_state_serialization_omits_embedded_player_payloads(draft_state, player_catalog) -> None:
    """Verify serialized draft state contains ids and scalar fields only."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.append_pick(Pick(pick_number=1, team_id=1, player_id=1))
    payload = json.dumps(draft_state.to_dict())

    assert "\"name\"" not in payload and "\"position\"" not in payload and "\"projected_points\"" not in payload


def test_draft_state_load_from_dict_restores_typed_history(draft_state) -> None:
    """Verify deserialization restores typed picks and rosters."""
    payload = {
        "available_player_ids": [3, 4],
        "team_rosters": {"1": {"player_ids": [1, 2], "qb_count": 1, "rb_count": 1, "wr_count": 0, "te_count": 0, "flex_count": 0}},
        "draft_order": [1, 2, 3, 4],
        "current_pick_index": 2,
        "current_pick_number": 3,
        "draft_history": [{"pick_number": 1, "team_id": 1, "player_id": 1}],
        "override_team_id": 4,
        "agent_team_id": 2,
    }
    draft_state.load_from_dict(payload)

    assert draft_state.draft_history[0].player_id == 1 and draft_state.roster_for_team(1).player_ids == [1, 2] and draft_state.override_team_id == 4
