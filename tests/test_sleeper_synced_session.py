"""Tests for config-driven Sleeper synced sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from draft_buddy.data.sleeper_client import SleeperHttpGateway
from draft_buddy.web.session import DraftSessionManager, SyncReadOnlyError
from draft_buddy.web.sleeper_synced_session import (
    SleeperSyncedSession,
    validate_sleeper_sync_config,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sleeper"
ROSTER_ID_TO_TEAM_ID = {10: 1, 20: 2, 30: 3, 40: 4}


class FakeDraftGateway:
    """In-memory Sleeper gateway for synced-session tests."""

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
        """Return an empty directory so resolver uses pick metadata."""
        return pd.DataFrame()

    def fetch_league_rosters(self, league_id: str) -> pd.DataFrame:
        """Unused in these tests."""
        _ = league_id
        return pd.DataFrame(columns=["roster_id", "sleeper_id"])


def _enable_sync(config) -> None:
    """Configure a 4-team test league for Sleeper sync."""
    config.draft.SLEEPER_SYNC_ENABLED = True
    config.draft.SLEEPER_DRAFT_ID = "draft123"
    config.draft.SLEEPER_LEAGUE_ID = "league123"
    config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID = dict(ROSTER_ID_TO_TEAM_ID)
    config.draft.SLEEPER_SYNC_POLL_SECONDS = 3
    config.draft.NUM_TEAMS = 4


def test_validate_sleeper_sync_config_requires_draft_id(config) -> None:
    """Verify a missing draft id fails loud before the session boots."""
    config.draft.SLEEPER_SYNC_ENABLED = True
    config.draft.SLEEPER_DRAFT_ID = ""
    config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID = {1: 1, 2: 2, 3: 3, 4: 4}
    config.draft.NUM_TEAMS = 4

    with pytest.raises(ValueError, match="SLEEPER_DRAFT_ID"):
        validate_sleeper_sync_config(config)


def test_sleeper_synced_session_replays_picks_into_ui_state(config, player_catalog) -> None:
    """Verify bootstrap mirrors fixture picks and marks the session read-only."""
    _ = player_catalog
    _enable_sync(config)
    draft_meta = json.loads((FIXTURES / "draft_snake_12team.json").read_text(encoding="utf-8"))
    picks = json.loads((FIXTURES / "picks_after_5.json").read_text(encoding="utf-8"))
    gateway = FakeDraftGateway(draft_meta, picks)

    session = SleeperSyncedSession(config, sleeper_gateway=gateway)
    ui_state = session.get_ui_state()

    assert ui_state["sleeper_sync"] is True
    assert ui_state["last_synced_pick_no"] == 5
    assert ui_state["sleeper_draft_id"] == "draft123"
    assert len(session.draft_history) == 5


def test_sleeper_synced_session_rejects_local_picks(config, player_catalog) -> None:
    """Verify the synced session has no local pick mutation surface."""
    _ = player_catalog
    _enable_sync(config)
    draft_meta = json.loads((FIXTURES / "draft_snake_12team.json").read_text(encoding="utf-8"))
    gateway = FakeDraftGateway(draft_meta, [])
    session = SleeperSyncedSession(config, sleeper_gateway=gateway)

    with pytest.raises(SyncReadOnlyError, match="Sleeper sync"):
        session.draft_player(1)


def test_session_manager_builds_synced_session_when_enabled(config, player_catalog) -> None:
    """Verify the manager factory selects SleeperSyncedSession from config."""
    _ = player_catalog
    _enable_sync(config)
    draft_meta = json.loads((FIXTURES / "draft_snake_12team.json").read_text(encoding="utf-8"))
    gateway = FakeDraftGateway(draft_meta, [])
    manager = DraftSessionManager(config, sleeper_gateway=gateway)

    session = manager.get_or_create("unused")

    assert isinstance(session, SleeperSyncedSession)
    assert session.get_ui_state()["sleeper_sync"] is True


def test_sync_from_sleeper_skips_fetch_within_poll_interval(config, player_catalog) -> None:
    """Verify extra UI polls do not multiply Sleeper GETs inside the interval."""
    _ = player_catalog
    _enable_sync(config)
    draft_meta = json.loads((FIXTURES / "draft_snake_12team.json").read_text(encoding="utf-8"))
    gateway = FakeDraftGateway(draft_meta, [])
    session = SleeperSyncedSession(config, sleeper_gateway=gateway)
    calls = {"count": 0}
    original = gateway.fetch_draft_picks

    def counting_fetch(draft_id: str) -> pd.DataFrame:
        calls["count"] += 1
        return original(draft_id)

    gateway.fetch_draft_picks = counting_fetch
    session.sync_from_sleeper()

    assert calls["count"] == 0


def test_sleeper_synced_session_reset_replays_the_same_board(config, player_catalog) -> None:
    """Verify a restart replay from pick 0 rebuilds the same visual board."""
    _ = player_catalog
    _enable_sync(config)
    draft_meta = json.loads((FIXTURES / "draft_snake_12team.json").read_text(encoding="utf-8"))
    picks = json.loads((FIXTURES / "picks_after_5.json").read_text(encoding="utf-8"))
    gateway = FakeDraftGateway(draft_meta, picks)
    session = SleeperSyncedSession(config, sleeper_gateway=gateway)
    first_board = session.get_ui_state()["visual_board"]

    session.reset()

    assert session.get_ui_state()["visual_board"] == first_board
