"""Tests for FastAPI route behavior."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

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


def test_dashboard_returns_500_when_frontend_file_missing(config, fake_session, monkeypatch) -> None:
    """Verify dashboard route returns a plain-text 500 when the frontend is absent."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/")

    assert response.status_code == 500


def test_ai_suggestion_for_team_parses_ignore_ids(config, fake_session) -> None:
    """Verify ignore query parsing passes numeric ids to the session method."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/api/draft/ai_suggestion_for_team?team_id=2&ignore=1, x, 3")

    assert response.status_code == 200 and response.json()["ignore"] == [1, 3]


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
