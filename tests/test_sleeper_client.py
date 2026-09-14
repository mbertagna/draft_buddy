"""Tests for Sleeper HTTP client caching and data-shaping behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from draft_buddy.data.sleeper_client import SleeperHttpGateway


def _fake_response(payload):
    """Build a minimal object mimicking a requests.Response."""
    response = SimpleNamespace()
    response.json = lambda: payload
    response.raise_for_status = lambda: None
    return response


def test_fetch_all_players_downloads_and_caches_when_missing(monkeypatch, tmp_path: Path) -> None:
    """Verify a missing cache triggers a download and writes the cache file."""
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    raw_players = {
        "4984": {
            "full_name": "Josh Allen",
            "position": "QB",
            "team": "BUF",
            "status": "Active",
            "years_exp": 7,
        }
    }
    monkeypatch.setattr("requests.get", lambda _url: _fake_response(raw_players))

    players_df = gateway.fetch_all_players()

    cache_path = tmp_path / "sleeper_players.json"
    assert cache_path.exists()
    assert players_df.iloc[0]["sleeper_id"] == "4984"
    assert players_df.iloc[0]["full_name"] == "Josh Allen"
    assert players_df.iloc[0]["years_exp"] == 7


def test_fetch_all_players_uses_fresh_cache_without_downloading(monkeypatch, tmp_path: Path) -> None:
    """Verify a fresh cache file is used instead of issuing a new request."""
    cache_path = tmp_path / "sleeper_players.json"
    cache_path.write_text(json.dumps({"1": {"full_name": "Cached Player", "position": "RB"}}), encoding="utf-8")
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    called = {"downloaded": False}
    monkeypatch.setattr("requests.get", lambda _url: called.__setitem__("downloaded", True))

    players_df = gateway.fetch_all_players()

    assert called["downloaded"] is False
    assert players_df.iloc[0]["full_name"] == "Cached Player"


def test_fetch_all_players_redownloads_when_cache_is_stale(monkeypatch, tmp_path: Path) -> None:
    """Verify a stale cache file is refreshed via a new download."""
    cache_path = tmp_path / "sleeper_players.json"
    cache_path.write_text(json.dumps({"1": {"full_name": "Old Player", "position": "RB"}}), encoding="utf-8")
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path), cache_max_age_seconds=0)
    raw_players = {"2": {"full_name": "New Player", "position": "WR"}}
    monkeypatch.setattr("requests.get", lambda _url: _fake_response(raw_players))

    players_df = gateway.fetch_all_players()

    assert players_df.iloc[0]["full_name"] == "New Player"


def test_fetch_league_rosters_flattens_roster_player_ids(monkeypatch, tmp_path: Path) -> None:
    """Verify league rosters are flattened into one row per rostered player."""
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    rosters_payload = [
        {"roster_id": 1, "players": ["100", "101"]},
        {"roster_id": 2, "players": ["200"]},
    ]
    monkeypatch.setattr("requests.get", lambda _url: _fake_response(rosters_payload))

    rosters_df = gateway.fetch_league_rosters("league123")

    assert list(rosters_df["sleeper_id"]) == ["100", "101", "200"]
    assert list(rosters_df["roster_id"]) == [1, 1, 2]


def test_fetch_draft_returns_raw_metadata(monkeypatch, tmp_path: Path) -> None:
    """Verify draft metadata is returned without reshaping."""
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    payload = {
        "type": "snake",
        "status": "drafting",
        "slot_to_roster_id": {"1": 10, "2": 20},
        "settings": {"rounds": 16},
    }
    monkeypatch.setattr("requests.get", lambda _url: _fake_response(payload))

    draft = gateway.fetch_draft("draft123")

    assert draft["type"] == "snake"
    assert draft["slot_to_roster_id"]["1"] == 10


def test_fetch_draft_picks_preserves_string_player_ids(monkeypatch, tmp_path: Path) -> None:
    """Verify picks flatten metadata and keep DST string ids."""
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    payload = [
        {
            "pick_no": 1,
            "player_id": "1001",
            "roster_id": 10,
            "draft_slot": 1,
            "round": 1,
            "metadata": {
                "first_name": "Josh",
                "last_name": "Allen",
                "position": "QB",
                "team": "BUF",
            },
        },
        {
            "pick_no": 2,
            "player_id": "DET",
            "roster_id": 20,
            "draft_slot": 2,
            "round": 1,
            "metadata": {
                "first_name": "Detroit",
                "last_name": "Defense",
                "position": "DEF",
                "team": "DET",
            },
        },
    ]
    monkeypatch.setattr("requests.get", lambda _url: _fake_response(payload))

    picks_df = gateway.fetch_draft_picks("draft123")

    assert list(picks_df["player_id"]) == ["1001", "DET"]
    assert list(picks_df["position"]) == ["QB", "DEF"]
    assert picks_df.iloc[0]["first_name"] == "Josh"


def test_fetch_draft_picks_empty_payload_returns_empty_frame(monkeypatch, tmp_path: Path) -> None:
    """Verify an empty picks list still has the expected columns."""
    gateway = SleeperHttpGateway(cache_dir=str(tmp_path))
    monkeypatch.setattr("requests.get", lambda _url: _fake_response([]))

    picks_df = gateway.fetch_draft_picks("draft123")

    assert list(picks_df.columns) == [
        "pick_no",
        "player_id",
        "roster_id",
        "draft_slot",
        "round",
        "first_name",
        "last_name",
        "position",
        "team",
        "injury_status",
    ]
    assert picks_df.empty
