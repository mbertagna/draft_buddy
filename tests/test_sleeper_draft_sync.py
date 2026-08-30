"""Tests for Sleeper draft-order translation and incremental pick sync."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from draft_buddy.core.draft_invariants import assert_invariants
from draft_buddy.data.sleeper_client import SleeperHttpGateway
from draft_buddy.data.sleeper_draft_sync import SleeperDraftSyncService
from draft_buddy.data.sleeper_player_resolver import PlayerResolver, encode_sleeper_player_id

FIXTURES = Path(__file__).parent / "fixtures" / "sleeper"
ROSTER_ID_TO_TEAM_ID = {10: 1, 20: 2, 30: 3, 40: 4}


class FakeDraftGateway:
    """In-memory Sleeper gateway for sync tests."""

    def __init__(self, draft_meta: dict, picks: list[dict]) -> None:
        self.draft_meta = draft_meta
        self.picks = picks

    def fetch_draft(self, draft_id: str) -> dict:
        """Return stored draft metadata."""
        _ = draft_id
        return self.draft_meta

    def fetch_draft_picks(self, draft_id: str) -> pd.DataFrame:
        """Return stored picks shaped like the HTTP gateway."""
        _ = draft_id
        return SleeperHttpGateway._to_draft_picks_dataframe(self.picks)

    def fetch_all_players(self) -> pd.DataFrame:
        """Unused in these tests."""
        return pd.DataFrame()

    def fetch_league_rosters(self, league_id: str) -> pd.DataFrame:
        """Unused in these tests."""
        _ = league_id
        return pd.DataFrame(columns=["roster_id", "sleeper_id"])


def _load_json(name: str):
    """Load a Sleeper fixture file."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_build_draft_order_snakes_mapped_team_ids() -> None:
    """Verify slot-to-roster mapping becomes a snake list of team ids."""
    draft_meta = _load_json("draft_snake_12team.json")
    service = SleeperDraftSyncService(FakeDraftGateway(draft_meta, []))

    draft_order = service.build_draft_order(
        draft_meta, ROSTER_ID_TO_TEAM_ID, num_teams=4, rounds=2
    )

    assert draft_order == [1, 2, 3, 4, 4, 3, 2, 1]


def test_build_draft_order_rejects_non_snake_type() -> None:
    """Verify auction drafts fail fast."""
    service = SleeperDraftSyncService(FakeDraftGateway({"type": "auction"}, []))

    with pytest.raises(ValueError, match="Unsupported Sleeper draft type"):
        service.build_draft_order({"type": "auction"}, ROSTER_ID_TO_TEAM_ID, 4, 2)


def test_sync_new_picks_applies_only_new_pick_numbers(draft_controller) -> None:
    """Verify incremental sync does not re-apply earlier picks."""
    picks = _load_json("picks_after_5.json")
    service = SleeperDraftSyncService(FakeDraftGateway({}, picks))
    resolver = PlayerResolver(draft_controller.player_catalog)

    high_water, errors = service.sync_new_picks(
        "draft123", draft_controller, resolver, ROSTER_ID_TO_TEAM_ID, last_synced_pick_no=0
    )
    assert high_water == 5
    assert errors == []
    first_count = len(draft_controller.state.draft_history)

    high_water, errors = service.sync_new_picks(
        "draft123", draft_controller, resolver, ROSTER_ID_TO_TEAM_ID, last_synced_pick_no=5
    )
    assert high_water == 5
    assert errors == []
    assert len(draft_controller.state.draft_history) == first_count


def test_sync_new_picks_materializes_off_catalog_skill_player(draft_controller) -> None:
    """Verify off-catalog skill picks become sleeper_only catalog entries."""
    picks = _load_json("pick_with_off_catalog_player.json")
    service = SleeperDraftSyncService(FakeDraftGateway({}, picks))
    resolver = PlayerResolver(draft_controller.player_catalog)

    high_water, errors = service.sync_new_picks(
        "draft123", draft_controller, resolver, ROSTER_ID_TO_TEAM_ID, last_synced_pick_no=0
    )

    assert high_water == 1
    assert errors == []
    player = draft_controller.player_catalog.require(9999)
    assert player.data_completeness == "sleeper_only"
    assert player.name == "Off Catalog"
    assert 9999 in draft_controller.state.roster_for_team(1).player_ids


def test_sync_new_picks_places_dst_as_display_only(draft_controller) -> None:
    """Verify DST string ids do not consume skill roster slots."""
    picks = _load_json("picks_after_10.json")
    service = SleeperDraftSyncService(FakeDraftGateway({}, picks))
    resolver = PlayerResolver(draft_controller.player_catalog)

    high_water, errors = service.sync_new_picks(
        "draft123", draft_controller, resolver, ROSTER_ID_TO_TEAM_ID, last_synced_pick_no=0
    )

    dst_id = encode_sleeper_player_id("DET")
    assert high_water == 10
    assert errors == []
    assert dst_id in draft_controller.state.display_only_player_ids
    assert dst_id not in draft_controller.state.roster_for_team(1).player_ids
    assert_invariants(draft_controller.state)


def test_sync_new_picks_uses_draft_slot_when_roster_id_is_null(draft_controller) -> None:
    """Verify mock-draft picks with a null roster_id still land on a team."""
    picks = [
        {
            "pick_no": 1,
            "player_id": "1001",
            "roster_id": None,
            "draft_slot": 1,
            "round": 1,
            "metadata": {
                "first_name": "Bijan",
                "last_name": "Robinson",
                "position": "RB",
                "team": "ATL",
                "injury_status": "",
            },
        }
    ]
    service = SleeperDraftSyncService(FakeDraftGateway({}, picks))
    resolver = PlayerResolver(draft_controller.player_catalog)

    high_water, errors = service.sync_new_picks(
        "draft123",
        draft_controller,
        resolver,
        ROSTER_ID_TO_TEAM_ID,
        last_synced_pick_no=0,
        slot_to_roster_id={"1": 10},
    )

    assert errors == []
    assert high_water == 1
    assert 1001 in draft_controller.state.roster_for_team(1).player_ids
