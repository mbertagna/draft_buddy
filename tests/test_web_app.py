"""Tests for FastAPI route behavior."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np

from draft_buddy.web.app import create_app
from draft_buddy.data.insights.loader import LoadedPlayerInsights
from draft_buddy.data.insights.schemas import (
    Confidence,
    DepthRole,
    PlayerInsight,
    PlayerInsightsFile,
    PlayingTimeTier,
    RecoveryStatus,
    RiskLevel,
)


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

    def run_locked(self, session_id: str, mutation):
        """Run a mutation against the fake session and persist."""
        _ = session_id
        mutation(self.session)
        self.session.save_state()
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

    assert response.status_code == 400 and response.json()["message"] == "bad pick"


def test_undo_pick_maps_value_error_to_400(config, fake_session) -> None:
    """Verify undo route translates ValueError into HTTP 400."""
    fake_session.undo_last_pick = lambda: (_ for _ in ()).throw(ValueError("cannot undo"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/undo")

    assert response.status_code == 400 and response.json()["message"] == "cannot undo"


def test_shelve_players_requires_player_ids(config, fake_session) -> None:
    """Verify shelve rejects payloads without player_ids."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/shelve", json={})

    assert response.status_code == 400


def test_shelve_by_adp_requires_max_adp(config, fake_session) -> None:
    """Verify shelve_by_adp rejects payloads without max_adp."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/shelve_by_adp", json={})

    assert response.status_code == 400


def test_shelve_inactive_returns_ui_state(config, fake_session) -> None:
    """Verify shelve_inactive mutation returns the session UI state."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/shelve_inactive", json={})

    assert response.status_code == 200 and response.json() == {"ok": True}


def test_shelve_players_returns_ui_state(config, fake_session) -> None:
    """Verify shelve mutation returns the session UI state."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/shelve", json={"player_ids": [3]})

    assert response.status_code == 200 and response.json() == {"ok": True}


def test_transfer_requires_player_id(config, fake_session) -> None:
    """Verify transfer rejects payloads without player_id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/transfer", json={"to_team_id": 2})

    assert response.status_code == 400 and response.json()["message"] == "Player ID is required"


def test_transfer_requires_destination_team_id(config, fake_session) -> None:
    """Verify transfer rejects payloads without destination team id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/draft/transfer", json={"player_id": 1})

    assert response.status_code == 400
    assert response.json()["message"] == "Destination team ID is required"


def test_transfer_maps_value_error_to_400(config, fake_session) -> None:
    """Verify transfer validation errors become HTTP 400 responses."""
    fake_session.transfer_player = lambda _player_id, _to_team_id, to_round=None: (
        _ for _ in ()
    ).throw(ValueError("bad transfer"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/transfer", json={"player_id": 1, "to_team_id": 2})

    assert response.status_code == 400 and response.json()["message"] == "bad transfer"


def test_swap_requires_both_player_ids(config, fake_session) -> None:
    """Verify swap rejects payloads missing either player id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/swap", json={"player_id_1": 1})

    assert response.status_code == 400
    assert response.json()["message"] == "Both player_id_1 and player_id_2 are required"


def test_swap_maps_value_error_to_400(config, fake_session) -> None:
    """Verify swap validation errors become HTTP 400 responses."""
    fake_session.swap_players = lambda _player_id_1, _player_id_2: (
        _ for _ in ()
    ).throw(ValueError("bad swap"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post(
        "/api/draft/swap", json={"player_id_1": 1, "player_id_2": 2}
    )

    assert response.status_code == 400 and response.json()["message"] == "bad swap"


def test_transfer_accepts_optional_to_round(config, fake_session) -> None:
    """Verify transfer forwards an optional destination round."""
    calls: list[tuple] = []

    def _transfer(player_id, to_team_id, to_round=None):
        calls.append((player_id, to_team_id, to_round))

    fake_session.transfer_player = _transfer
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post(
        "/api/draft/transfer",
        json={"player_id": 1, "to_team_id": 2, "to_round": 3},
    )

    assert response.status_code == 200 and calls == [(1, 2, 3)]


def test_override_team_requires_team_id(config, fake_session) -> None:
    """Verify override route rejects payloads without team_id."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/override_team", json={})

    assert response.status_code == 400 and response.json()["message"] == "Team ID is required"


def test_override_team_maps_value_error_to_400(config, fake_session) -> None:
    """Verify override-team validation errors become HTTP 400 responses."""
    fake_session.set_current_team_picking = lambda _team_id: (_ for _ in ()).throw(ValueError("bad team"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/override_team", json={"team_id": 9})

    assert response.status_code == 400 and response.json()["message"] == "bad team"


def test_simulate_pick_maps_value_error_to_400(config, fake_session) -> None:
    """Verify simulate-pick validation errors become HTTP 400 responses."""
    fake_session.simulate_single_pick = lambda use_policy=False: (_ for _ in ()).throw(ValueError("stop"))
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/simulate_pick")

    assert response.status_code == 400 and response.json()["message"] == "stop"


def test_simulate_rest_maps_value_error_to_400(config, fake_session) -> None:
    """Verify simulate-rest validation errors become HTTP 400 responses."""
    fake_session.simulate_scheduled_picks_remaining = lambda use_policy=False: (_ for _ in ()).throw(
        ValueError("halt")
    )
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.post("/api/draft/simulate_rest")

    assert response.status_code == 400 and response.json()["message"] == "halt"


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

    assert response.status_code == 400 and response.json()["message"] == "schedule missing"


def test_simulate_season_maps_generic_error_to_500(config, fake_session, monkeypatch) -> None:
    """Verify simulator failures return HTTP 500."""
    web_app_module = importlib.import_module("draft_buddy.web.app")

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(web_app_module.SeasonSimulationService, "simulate_season", _raise_runtime_error)
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.post("/api/simulate_season")

    assert response.status_code == 500 and "Season simulation failed: boom" in response.json()["message"]


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


def test_get_players_includes_sleeper_fields(config, fake_session) -> None:
    """Verify player payload surfaces Sleeper-derived stats."""
    player = fake_session.player_catalog.get(1)
    fake_session.player_catalog = fake_session.player_catalog.with_updated_player(
        player.__class__(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            projected_points=player.projected_points,
            games_played_frac=player.games_played_frac,
            adp=player.adp,
            bye_week=player.bye_week,
            team=player.team,
            sleeper_status="Active",
            sleeper_injury_status="Questionable",
            sleeper_depth_chart_position="QB",
        )
    )
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))

    response = client.get("/api/players?search=QB%20One")
    payload = response.json()[0]

    assert payload["sleeper_injury_status"] == "Questionable" and payload["sleeper_depth_chart_position"] == "QB"


def _sample_loaded_insights(player_id: int = 1) -> LoadedPlayerInsights:
    """Return a minimal loaded insights fixture for web API tests."""
    insight = PlayerInsight(
        outlook_phrase="Starter role expected",
        summary="Should lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.NA,
        overall_confidence=Confidence.HIGH,
    )
    meta = PlayerInsightsFile(
        draft_year=2026,
        generated_at=datetime(2026, 7, 4, 17, 7, 47, tzinfo=timezone.utc),
        model="gemini-2.0-flash",
        players={str(player_id): insight},
    )
    return LoadedPlayerInsights(
        players={player_id: insight},
        meta=meta,
        source_path="/tmp/player_insights_2026_20260704T170747Z.json",
    )


def test_insights_meta_unavailable_when_not_loaded(config, fake_session) -> None:
    """Verify insights meta reports unavailable when no export is loaded."""
    client = TestClient(create_app(config=config, session_manager=FakeSessionManager(fake_session)))
    response = client.get("/api/insights/meta")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["enriched_player_count"] == 0


def test_insights_meta_returns_loaded_export_metadata(config, fake_session) -> None:
    """Verify insights meta surfaces loaded export metadata."""
    loaded = _sample_loaded_insights()
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(fake_session),
            loaded_insights=loaded,
        )
    )
    response = client.get("/api/insights/meta")
    payload = response.json()

    assert payload["available"] is True
    assert payload["draft_year"] == 2026
    assert payload["enriched_player_count"] == 1
    assert payload["source_file"] == "player_insights_2026_20260704T170747Z.json"


def test_get_players_includes_insight_when_enriched(config, fake_session) -> None:
    """Verify enriched players include insight payloads on /api/players."""
    loaded = _sample_loaded_insights(player_id=1)
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(fake_session),
            loaded_insights=loaded,
        )
    )
    response = client.get("/api/players?search=QB%20One")
    payload = response.json()[0]

    assert payload["insight"] is not None
    assert payload["insight"]["outlook_phrase"] == "Starter role expected"


def test_get_players_sets_insight_null_when_not_enriched(config, fake_session) -> None:
    """Verify unenriched players return insight null on /api/players."""
    loaded = _sample_loaded_insights(player_id=999)
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(fake_session),
            loaded_insights=loaded,
        )
    )
    response = client.get("/api/players?search=QB%20One")
    payload = response.json()[0]

    assert payload["insight"] is None
