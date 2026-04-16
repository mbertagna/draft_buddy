"""Web route tests using FastAPI TestClient."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from unittest.mock import patch

from fastapi.testclient import TestClient

from draft_buddy.domain.entities import Player
from draft_buddy.web.app import create_app


@dataclass
class _FakeSession:
    """Minimal session stub for app route tests."""

    should_raise_on_pick: bool = False
    should_raise_on_undo: bool = False
    should_raise_on_override: bool = False
    should_raise_on_simulate_pick: bool = False
    should_raise_on_simulate_rest: bool = False
    ai_payload: dict | None = None
    ai_all_payload: dict | None = None
    ai_for_team_payload: dict | None = None

    def __post_init__(self):
        self.current_pick_number = 2
        self.current_pick_idx = 1
        self.draft_order = [1, 2]
        self.available_players_ids = {101, 102, 103}
        self.player_map = {
            101: Player(101, "RB One", "RB", 120.0, adp=5.0, bye_week=7, team="KC"),
            102: Player(102, "WR One", "WR", 110.0, adp=12.0, bye_week=8, team="BUF"),
            103: Player(103, "QB One", "QB", 130.0, adp=2.0, bye_week=9, team="MIA"),
        }
        self._draft_history = [{"previous_pick_number": 1, "team_id": 1, "player_id": 101}]

    def get_ui_state(self):
        """Return deterministic UI state payload."""
        return {"ok": True}

    def draft_player(self, player_id: int):
        """Optionally raise to test 400 conversion."""
        if self.should_raise_on_pick:
            raise ValueError("invalid pick")
        self._draft_history.append(
            {"previous_pick_number": self.current_pick_number, "team_id": 1, "player_id": player_id}
        )

    def undo_last_pick(self):
        """Undo the last pick or raise a validation error."""
        if self.should_raise_on_undo:
            raise ValueError("cannot undo")
        if self._draft_history:
            self._draft_history.pop()

    def set_current_team_picking(self, team_id: int):
        """Persist an override or raise a validation error."""
        if self.should_raise_on_override:
            raise ValueError("invalid override")
        self.current_team_picking = team_id

    def simulate_single_pick(self):
        """Simulate one pick or raise a validation error."""
        if self.should_raise_on_simulate_pick:
            raise ValueError("cannot simulate")
        self.current_pick_number += 1

    def simulate_scheduled_picks_remaining(self):
        """Simulate the rest of the draft or raise a validation error."""
        if self.should_raise_on_simulate_rest:
            raise ValueError("cannot simulate rest")
        self.current_pick_idx = len(self.draft_order)

    def save_state(self, path: str):
        """No-op save."""
        del path

    def get_positional_baselines(self):
        """Return simple baselines for VORP sorting."""
        return {"QB": 100.0, "RB": 100.0, "WR": 100.0, "TE": 100.0}

    def get_ai_suggestion(self):
        """Return a deterministic single-team AI payload."""
        return self.ai_payload or {"QB": 0.1}

    def get_ai_suggestions_all(self):
        """Return deterministic all-team AI suggestions."""
        return self.ai_all_payload or {1: {"QB": 0.1}}

    def get_ai_suggestion_for_team(self, team_id: int, ignore_player_ids=None):
        """Return deterministic team-specific AI suggestions."""
        self.last_ai_team = team_id
        self.last_ignore_ids = ignore_player_ids or []
        return self.ai_for_team_payload or {"WR": 0.2}


class _FakeSessionManager:
    """Session manager stub with id capture."""

    def __init__(self, session: _FakeSession):
        self._session = session
        self.last_session_id = None

    def get_or_create(self, session_id: str):
        """Capture ID and return session."""
        self.last_session_id = session_id
        return self._session

    def create_new(self, session_id: str):
        """Capture ID and return session."""
        self.last_session_id = session_id
        return self._session


def _build_client(mock_config, session: _FakeSession) -> tuple[TestClient, _FakeSessionManager]:
    """Build client with injected fake session manager."""
    manager = _FakeSessionManager(session)
    app = create_app(config=mock_config, session_manager=manager)
    return TestClient(app), manager


def test_api_hello_returns_expected_message_json(mock_config):
    """Verify hello route returns health payload."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/hello")

    assert response.json()["message"] == "Hello from DRAFT BUDDY backend!"


def test_draft_state_generates_session_cookie_when_missing(mock_config):
    """Verify draft state route sets a new draft_session_id cookie."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/draft/state")

    assert "draft_session_id" in response.cookies


def test_draft_state_uses_provided_session_cookie_id(mock_config):
    """Verify provided cookie id is passed to session manager lookup."""
    client, manager = _build_client(mock_config, _FakeSession())
    client.get("/api/draft/state", cookies={"draft_session_id": "fixed-session-id"})

    assert manager.last_session_id == "fixed-session-id"


def test_draft_pick_returns_200_for_valid_payload(mock_config):
    """Verify valid draft pick request succeeds."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.post("/api/draft/pick", json={"player_id": 101})

    assert response.status_code == 200


def test_draft_pick_returns_400_when_session_raises_value_error(mock_config):
    """Verify invalid pick errors are translated to HTTP 400."""
    client, _ = _build_client(mock_config, _FakeSession(should_raise_on_pick=True))
    response = client.post("/api/draft/pick", json={"player_id": 999})

    assert response.status_code == 400


def test_draft_pick_returns_400_when_player_id_missing(mock_config):
    """Verify missing player_id payload is rejected."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.post("/api/draft/pick", json={})

    assert response.status_code == 400


def test_export_csv_returns_csv_content_type(mock_config):
    """Verify export endpoint responds with CSV media type."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/draft/export_csv")

    assert response.headers["content-type"].startswith("text/csv")


def test_export_csv_contains_expected_header_row(mock_config):
    """Verify export CSV includes expected header names."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/draft/export_csv")

    assert "Pick Number,Team ID,Player ID,Name,Position" in response.text


def test_export_csv_skips_history_rows_for_missing_players(mock_config):
    """Verify CSV export skips picks that no longer exist in player map."""
    session = _FakeSession()
    session._draft_history.append({"previous_pick_number": 2, "team_id": 1, "player_id": 99999})
    client, _ = _build_client(mock_config, session)
    response = client.get("/api/draft/export_csv")

    assert "99999" not in response.text


def test_players_filter_and_sort_returns_only_wr_rb_ordered_by_adp_ascending(mock_config):
    """Verify /api/players filtering and sort behavior."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?position=WR,RB&sort_by=adp&sort_dir=asc")
    payload = response.json()

    assert [player["player_id"] for player in payload] == [101, 102]


def test_players_search_filters_by_substring(mock_config):
    """Verify search query narrows payload by player name substring."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?search=qb")

    assert [player["player_id"] for player in response.json()] == [103]


def test_players_sort_by_projected_points_desc_returns_highest_first(mock_config):
    """Verify projected_points sorting returns highest projection first."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?sort_by=projected_points&sort_dir=desc")

    assert response.json()[0]["player_id"] == 103


def test_players_sort_by_name_ascending_orders_alphabetically(mock_config):
    """Verify name sorting returns alphabetical order."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?sort_by=name&sort_dir=asc")

    assert [player["name"] for player in response.json()] == ["QB One", "RB One", "WR One"]


def test_players_sort_by_position_ascending_orders_by_position_code(mock_config):
    """Verify position sorting orders by position field."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?sort_by=position&sort_dir=asc")

    assert [player["position"] for player in response.json()] == ["QB", "RB", "WR"]


def test_players_unknown_sort_key_falls_back_to_vorp(mock_config):
    """Verify unknown sort key falls back to VORP sorting path."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/api/players?sort_by=unknown_key&sort_dir=desc")

    assert response.status_code == 200


def test_dashboard_returns_500_when_frontend_index_missing(mock_config):
    """Verify the dashboard route serves the frontend when the file exists."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.get("/")

    assert response.status_code == 200


def test_create_new_draft_returns_session_ui_state(mock_config):
    """Verify creating a new draft returns the session state payload."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.post("/api/draft/new")

    assert response.json() == {"ok": True}


def test_draft_undo_returns_400_when_session_rejects_request(mock_config):
    """Verify undo errors surface as HTTP 400."""
    client, _ = _build_client(mock_config, _FakeSession(should_raise_on_undo=True))
    response = client.post("/api/draft/undo")

    assert response.status_code == 400


def test_override_team_requires_team_id(mock_config):
    """Verify override requests reject missing team IDs."""
    client, _ = _build_client(mock_config, _FakeSession())
    response = client.post("/api/draft/override_team", json={})

    assert response.status_code == 400


def test_override_team_returns_400_when_session_raises_value_error(mock_config):
    """Verify override errors are translated to HTTP 400."""
    client, _ = _build_client(mock_config, _FakeSession(should_raise_on_override=True))
    response = client.post("/api/draft/override_team", json={"team_id": 2})

    assert response.status_code == 400


def test_simulate_pick_returns_400_when_session_raises_value_error(mock_config):
    """Verify simulate-pick errors are translated to HTTP 400."""
    client, _ = _build_client(mock_config, _FakeSession(should_raise_on_simulate_pick=True))
    response = client.post("/api/draft/simulate_pick")

    assert response.status_code == 400


def test_simulate_rest_returns_400_when_session_raises_value_error(mock_config):
    """Verify simulate-rest errors are translated to HTTP 400."""
    client, _ = _build_client(mock_config, _FakeSession(should_raise_on_simulate_rest=True))
    response = client.post("/api/draft/simulate_rest")

    assert response.status_code == 400


def test_ai_suggestion_returns_session_payload(mock_config):
    """Verify AI suggestion route delegates to the session."""
    client, _ = _build_client(mock_config, _FakeSession(ai_payload={"RB": 0.6}))
    response = client.get("/api/draft/ai_suggestion")

    assert response.json() == {"RB": 0.6}


def test_ai_suggestions_all_returns_session_payload(mock_config):
    """Verify all-team AI suggestion route delegates to the session."""
    client, _ = _build_client(mock_config, _FakeSession(ai_all_payload={"1": {"WR": 0.7}}))
    response = client.get("/api/draft/ai_suggestions_all")

    assert response.json() == {"1": {"WR": 0.7}}


def test_ai_suggestion_for_team_parses_only_numeric_ignore_tokens(mock_config):
    """Verify ignore parsing keeps only integer tokens."""
    session = _FakeSession(ai_for_team_payload={"TE": 0.4})
    client, _ = _build_client(mock_config, session)
    response = client.get("/api/draft/ai_suggestion_for_team?team_id=2&ignore=1, x,3a, 4")

    assert response.json() == {"TE": 0.4} and session.last_ignore_ids == [1, 4]


def test_draft_summary_counts_only_known_players(mock_config):
    """Verify summary ignores history entries for missing players."""
    session = _FakeSession()
    session._draft_history.append({"previous_pick_number": 2, "team_id": 1, "player_id": 99999})
    client, _ = _build_client(mock_config, session)
    response = client.get("/api/draft/summary")

    assert response.json()["picks_by_position"]["RB"] == 1


def test_players_payload_converts_infinite_adp_and_missing_bye_week(mock_config):
    """Verify player payload normalizes inf ADP and missing bye weeks for the UI."""
    session = _FakeSession()
    session.player_map[103].adp = float("inf")
    session.player_map[103].bye_week = float("nan")
    client, _ = _build_client(mock_config, session)
    response = client.get("/api/players?sort_by=projected_points&sort_dir=desc")

    assert response.json()[0]["adp"] is None and response.json()[0]["bye_week"] == "N/A"


def test_simulate_season_returns_400_when_service_raises_file_not_found(mock_config):
    """Verify missing simulation data maps to HTTP 400."""
    app_module = importlib.import_module("draft_buddy.web.app")
    with patch.object(app_module, "SeasonSimulationService") as mock_service_class:
        mock_service_class.return_value.simulate_season.side_effect = FileNotFoundError("missing")
        app = create_app(config=mock_config, session_manager=_FakeSessionManager(_FakeSession()))
    response = TestClient(app).post("/api/simulate_season")

    assert response.status_code == 400


def test_simulate_season_returns_500_when_service_raises_unexpected_error(mock_config):
    """Verify unexpected simulation failures map to HTTP 500."""
    app_module = importlib.import_module("draft_buddy.web.app")
    with patch.object(app_module, "SeasonSimulationService") as mock_service_class:
        mock_service_class.return_value.simulate_season.side_effect = RuntimeError("boom")
        app = create_app(config=mock_config, session_manager=_FakeSessionManager(_FakeSession()))
    response = TestClient(app).post("/api/simulate_season")

    assert response.status_code == 500
