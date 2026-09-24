"""Tests for in-season snapshot adapters, store, and name resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from draft_buddy.inseason.espn_adapter import EspnSnapshotAdapter
from draft_buddy.inseason.exceptions import NameResolutionError
from draft_buddy.inseason.sleeper_adapter import SleeperSnapshotAdapter, current_week_from_nfl_state
from draft_buddy.inseason.state import LeagueSnapshot, TeamRecord, TeamSnapshot
from draft_buddy.inseason.store import InSeasonStore


class FakeSleeperGateway:
    """In-memory Sleeper gateway for snapshot tests."""

    def fetch_nfl_state(self) -> dict:
        """Return a regular-season week."""
        return {"week": 4, "season_type": "regular"}

    def fetch_league_roster_payloads(self, league_id: str) -> list:
        """Return two mapped rosters."""
        _ = league_id
        return [
            {
                "roster_id": 10,
                "players": ["1", "2"],
                "reserve": ["3"],
                "settings": {"wins": 2, "losses": 1, "ties": 0, "fpts": 300.0},
            },
            {
                "roster_id": 20,
                "players": ["5", "6"],
                "reserve": [],
                "settings": {"wins": 1, "losses": 2, "ties": 0, "fpts": 250.0},
            },
        ]

    def fetch_league_matchups(self, league_id: str, week: int) -> list:
        """Return one pairing for every remaining week."""
        _ = league_id
        if week > 5:
            return []
        return [{"roster_id": 10, "matchup_id": 1}, {"roster_id": 20, "matchup_id": 1}]


def test_current_week_from_nfl_state_uses_regular_week() -> None:
    """Verify regular-season Sleeper state maps onto the NFL week."""
    week = current_week_from_nfl_state({"week": 4, "season_type": "regular"}, last_regular_week=14)

    assert week == 4


def test_sleeper_adapter_maps_roster_ids_to_team_ids(player_catalog) -> None:
    """Verify Sleeper roster ids are translated with the configured map."""
    adapter = SleeperSnapshotAdapter(FakeSleeperGateway())

    snapshot = adapter.build_snapshot(
        league_id="redraft_nbfl_12",
        sleeper_league_id="league-1",
        roster_id_to_team_id={10: 1, 20: 2},
        catalog=player_catalog,
        team_names={1: "Alpha", 2: "Bravo"},
        last_regular_week=5,
    )

    assert snapshot.teams[1].player_ids == [1, 2]


def test_sleeper_adapter_records_ir_reserve_ids(player_catalog) -> None:
    """Verify Sleeper reserve lists become IR player ids."""
    adapter = SleeperSnapshotAdapter(FakeSleeperGateway())

    snapshot = adapter.build_snapshot(
        league_id="redraft_nbfl_12",
        sleeper_league_id="league-1",
        roster_id_to_team_id={10: 1, 20: 2},
        catalog=player_catalog,
        team_names={1: "Alpha", 2: "Bravo"},
        last_regular_week=5,
    )

    assert snapshot.teams[1].ir_player_ids == [3]


def test_sleeper_adapter_keeps_live_weeks_when_a_later_week_is_empty(player_catalog) -> None:
    """Verify an empty Sleeper week fills only that week."""
    adapter = SleeperSnapshotAdapter(FakeSleeperGateway())

    snapshot = adapter.build_snapshot(
        league_id="redraft_nbfl_12",
        sleeper_league_id="league-1",
        roster_id_to_team_id={10: 1, 20: 2},
        catalog=player_catalog,
        team_names={1: "Alpha", 2: "Bravo"},
        last_regular_week=6,
        current_week=4,
    )
    by_week = {matchup.week: matchup.source for matchup in snapshot.remaining_matchups}

    assert by_week[4] == "sleeper" and by_week[6] == "round_robin"


def test_espn_adapter_uses_pasted_matchup_for_that_week(player_catalog) -> None:
    """Verify a pasted week-4 pairing replaces the generated opponent."""
    adapter = EspnSnapshotAdapter()
    payload = {
        "current_week": 4,
        "teams": [
            {"team_id": 1, "player_ids": [1]},
            {"team_id": 2, "player_ids": [2]},
        ],
        "matchups": [{"week": 4, "home": "1", "away": "2"}],
    }

    snapshot = adapter.build_snapshot(
        league_id="red_league_10",
        payload=payload,
        catalog=player_catalog,
        team_names={1: "Alpha", 2: "Bravo"},
        last_regular_week=5,
        team_ids=[1, 2],
    )
    week_four = next(matchup for matchup in snapshot.remaining_matchups if matchup.week == 4)

    assert week_four.source == "file" and week_four.home_team_id == 1 and week_four.away_team_id == 2


def test_espn_adapter_resolves_unique_player_names(player_catalog) -> None:
    """Verify a unique pasted name maps onto the catalog id."""
    adapter = EspnSnapshotAdapter()
    payload = {
        "current_week": 3,
        "teams": [{"team_id": 1, "wins": 2, "losses": 0, "player_names": ["QB One"]}],
    }

    snapshot = adapter.build_snapshot(
        league_id="red_league_10",
        payload=payload,
        catalog=player_catalog,
        team_names={1: "SKOL!"},
        last_regular_week=4,
        team_ids=[1, 2],
    )

    assert snapshot.teams[1].player_ids == [1]


def test_espn_adapter_resolves_team_display_name(player_catalog) -> None:
    """Verify a pasted manager name maps onto the configured team id."""
    adapter = EspnSnapshotAdapter()
    payload = {
        "current_week": 3,
        "teams": [{"team": "SKOL!", "wins": 2, "losses": 0, "player_names": ["QB One"]}],
    }

    snapshot = adapter.build_snapshot(
        league_id="red_league_10",
        payload=payload,
        catalog=player_catalog,
        team_names={1: "SKOL!"},
        last_regular_week=4,
        team_ids=[1],
    )

    assert snapshot.teams[1].player_ids == [1]


def test_espn_adapter_rejects_unknown_team_name(player_catalog) -> None:
    """Verify unknown manager names fail instead of guessing."""
    adapter = EspnSnapshotAdapter()
    payload = {
        "current_week": 3,
        "teams": [{"team": "Ghost Franchise", "player_names": ["QB One"]}],
    }

    with pytest.raises(NameResolutionError, match="unmatched"):
        adapter.build_snapshot(
            league_id="red_league_10",
            payload=payload,
            catalog=player_catalog,
            team_names={1: "SKOL!"},
            last_regular_week=4,
            team_ids=[1],
        )


def test_espn_adapter_rejects_unmatched_names(player_catalog) -> None:
    """Verify unknown ESPN names fail instead of guessing."""
    adapter = EspnSnapshotAdapter()
    payload = {
        "current_week": 3,
        "teams": [{"team_id": 1, "player_names": ["Nobody McGhost"]}],
    }

    with pytest.raises(NameResolutionError, match="unmatched"):
        adapter.build_snapshot(
            league_id="red_league_10",
            payload=payload,
            catalog=player_catalog,
            team_names={1: "SKOL!"},
            last_regular_week=4,
            team_ids=[1],
        )


def test_inseason_store_round_trips_snapshot(tmp_path: Path) -> None:
    """Verify snapshots persist and reload with the same team ids."""
    store = InSeasonStore(str(tmp_path / "state.json"), str(tmp_path / "proj.json"))
    snapshot = LeagueSnapshot(
        league_id="red_league_10",
        platform="espn",
        current_week=4,
        teams={
            1: TeamSnapshot(
                team_id=1,
                display_name="SKOL!",
                player_ids=[1, 2],
                record=TeamRecord(wins=2, losses=1),
            )
        },
    )
    store.save_snapshot(snapshot)

    loaded = store.load_snapshot()

    assert loaded.teams[1].player_ids == [1, 2]
