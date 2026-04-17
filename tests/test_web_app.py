"""Tests for FastAPI route behavior."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np

from draft_buddy.web.app import create_app


class FakeSessionManager:
    """Simple session manager returning one fake session."""

    def __init__(self, session):
        self.session = session
        self.created = False

    def get_or_create(self, session_id: str):
        _ = session_id
        return self.session

    def create_new(self, session_id: str):
        _ = session_id
        self.created = True
        return self.session


def test_create_new_draft_sets_session_cookie(config, fake_session) -> None:
    """Verify creating a draft sets the session cookie."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/new")

    assert response.status_code == 200 and "draft_session_id" in response.cookies


def test_draft_state_reuses_existing_session_cookie(config, fake_session) -> None:
    """Verify existing session cookies are preserved on draft state reads."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    client.cookies.set("draft_session_id", "existing-session")

    response = client.get("/api/draft/state")

    assert response.status_code == 200 and response.cookies.get("draft_session_id") is None


def test_draft_pick_requires_player_id(config, fake_session) -> None:
    """Verify draft_pick rejects payloads without player_id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/pick", json={})

    assert response.status_code == 400


def test_draft_pick_maps_value_error_to_400(config, fake_session) -> None:
    """Verify draft mutation route translates ValueError into HTTP 400."""
    fake_session.draft_player = lambda _player_id: (_ for _ in ()).throw(ValueError("bad pick"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/pick", json={"player_id": 1})

    assert response.status_code == 400 and response.json()["detail"] == "bad pick"


def test_undo_pick_maps_value_error_to_400(config, fake_session) -> None:
    """Verify undo route translates ValueError into HTTP 400."""
    fake_session.undo_last_pick = lambda: (_ for _ in ()).throw(ValueError("cannot undo"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/undo")

    assert response.status_code == 400 and response.json()["detail"] == "cannot undo"


def test_override_team_requires_team_id(config, fake_session) -> None:
    """Verify override route rejects payloads without team_id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/override_team", json={})

    assert response.status_code == 400 and response.json()["detail"] == "Team ID is required"


def test_override_team_maps_value_error_to_400(config, fake_session) -> None:
    """Verify override-team validation errors become HTTP 400 responses."""
    fake_session.set_current_team_picking = lambda _team_id: (_ for _ in ()).throw(ValueError("bad team"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/override_team", json={"team_id": 9})

    assert response.status_code == 400 and response.json()["detail"] == "bad team"


def test_simulate_pick_maps_value_error_to_400(config, fake_session) -> None:
    """Verify simulate-pick validation errors become HTTP 400 responses."""
    fake_session.simulate_single_pick = lambda: (_ for _ in ()).throw(ValueError("stop"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/simulate_pick")

    assert response.status_code == 400 and response.json()["detail"] == "stop"


def test_simulate_rest_maps_value_error_to_400(config, fake_session) -> None:
    """Verify simulate-rest validation errors become HTTP 400 responses."""
    fake_session.simulate_scheduled_picks_remaining = lambda: (_ for _ in ()).throw(ValueError("halt"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/simulate_rest")

    assert response.status_code == 400 and response.json()["detail"] == "halt"


def test_dashboard_returns_500_when_frontend_file_missing(config, fake_session, monkeypatch) -> None:
    """Verify dashboard route returns a plain-text 500 when the frontend is absent."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/")

    assert response.status_code == 500


def test_hello_world_returns_api_message(config, fake_session) -> None:
    """Verify the health route returns the static greeting."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/hello")

    assert response.status_code == 200 and "Hello from DRAFT BUDDY backend!" in response.json()["message"]


def test_ai_suggestion_for_team_parses_ignore_ids(config, fake_session) -> None:
    """Verify ignore query parsing passes numeric ids to the session method."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/api/draft/ai_suggestion_for_team?team_id=2&ignore=1, x, 3")

    assert response.status_code == 200 and response.json()["ignore"] == [1, 3]


def test_draft_summary_ignores_missing_catalog_players(config, fake_session) -> None:
    """Verify draft summary skips picks for players missing from the catalog."""
    fake_session.draft_history.append(SimpleNamespace(pick_number=2, team_id=1, player_id=9999))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/draft/summary")

    assert response.status_code == 200 and response.json()["total_picks"] == 1


def test_export_csv_formats_missing_player_adp_and_bye_week(config, fake_session) -> None:
    """Verify CSV export uses N/A for infinite ADP and missing bye week."""
    player = fake_session.player_catalog.get(1)
    fake_session.player_catalog = fake_session.player_catalog.with_updated_player(
        player.__class__(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            projected_points=player.projected_points,
            games_played_frac=player.games_played_frac,
            adp=float("inf"),
            bye_week=None,
            team=player.team,
        )
    )
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/api/draft/export_csv")

    assert response.status_code == 200 and "N/A" in response.text


def test_export_csv_skips_missing_players_in_history(config, fake_session) -> None:
    """Verify CSV export omits history rows that no longer resolve to a player."""
    fake_session.draft_history.append(SimpleNamespace(pick_number=2, team_id=2, player_id=9999))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/draft/export_csv")

    assert response.status_code == 200 and "9999" not in response.text


def test_simulate_season_maps_missing_file_to_400(config, fake_session, monkeypatch) -> None:
    """Verify missing schedule files return HTTP 400."""
    web_app_module = importlib.import_module("draft_buddy.web.app")

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("schedule missing")

    monkeypatch.setattr(web_app_module.SeasonSimulationService, "simulate_season", _raise_file_not_found)
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/simulate_season")

    assert response.status_code == 400 and response.json()["detail"] == "schedule missing"


def test_simulate_season_maps_generic_error_to_500(config, fake_session, monkeypatch) -> None:
    """Verify simulator failures return HTTP 500."""
    web_app_module = importlib.import_module("draft_buddy.web.app")

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(web_app_module.SeasonSimulationService, "simulate_season", _raise_runtime_error)
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/simulate_season")

    assert response.status_code == 500 and "Season simulation failed: boom" in response.json()["detail"]


def test_get_players_filters_by_search(config, fake_session) -> None:
    """Verify player list filtering applies the search term."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/api/players?search=QB%20One")

    assert response.status_code == 200 and len(response.json()) == 1


def test_get_players_filters_by_position_and_sorts_by_adp(config, fake_session) -> None:
    """Verify player table filtering and ADP sorting use query parameters."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/players?position=QB&sort_by=adp&sort_dir=asc")
    payload = response.json()

    assert response.status_code == 200 and [player["player_id"] for player in payload] == [1, 5, 9, 13]


def test_get_players_uses_default_vorp_sort_for_unknown_key(config, fake_session) -> None:
    """Verify unknown sort keys fall back to VORP ordering."""
    fake_session.get_positional_baselines = lambda: {"QB": 250.0, "RB": 200.0, "WR": 180.0, "TE": 120.0}
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/players?sort_by=unknown&sort_dir=desc")
    payload = response.json()

    assert response.status_code == 200 and payload[0]["player_id"] == 4


def test_get_players_formats_nan_bye_week_as_na(config, fake_session) -> None:
    """Verify player payload uses N/A when bye_week is NaN."""
    player = fake_session.player_catalog.get(1)
    fake_session.player_catalog = fake_session.player_catalog.with_updated_player(
        player.__class__(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            projected_points=player.projected_points,
            games_played_frac=player.games_played_frac,
            adp=player.adp,
            bye_week=float("nan"),
            team=player.team,
        )
    )
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/players?search=QB%20One")

    assert response.status_code == 200 and response.json()[0]["bye_week"] == "N/A"
