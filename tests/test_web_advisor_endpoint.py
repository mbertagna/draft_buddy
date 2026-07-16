"""Tests for the draft assistant web endpoint."""

from __future__ import annotations

from draft_buddy.llm.model_registry import LlmProvider, ModelOption
from draft_buddy.web.draft_advisor_schemas import PickRecommendation
from draft_buddy.web.draft_advisor_service import DraftAdvisorService
from draft_buddy.web.draft_advisor_gateway import DraftAdvisorGateway
from draft_buddy.web.app import create_app
from fastapi.testclient import TestClient


class FakeAdvisorGateway(DraftAdvisorGateway):
    """Return a deterministic recommendation for tests."""

    def fetch_payload(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        """Return a fixed payload."""
        _ = (system_prompt, user_prompt)
        payload = {
            "advising_team_id": 1,
            "is_agent_team": True,
            "recommended_player_id": 2,
            "recommended_name": "RB One",
            "confidence": "high",
            "rationale_bullets": ["Best RB available."],
            "alternates": [],
            "flags": ["none"],
            "unknown_factors": [],
        }
        return payload, '{"recommended_player_id": 2}'

    def recommend(self, system_prompt: str, user_prompt: str) -> PickRecommendation:
        """Return a fixed recommendation."""
        from draft_buddy.web.draft_advisor_schemas import parse_pick_recommendation

        payload, _raw = self.fetch_payload(system_prompt, user_prompt)
        return parse_pick_recommendation(payload)


class FakeAdvisorRegistry:
    """Minimal advisor registry for endpoint tests."""

    def __init__(self) -> None:
        self._gateway = FakeAdvisorGateway()
        self._models = [
            ModelOption(id="gemini-2.5-flash", label="Gemini 2.5 Flash", provider=LlmProvider.GEMINI),
            ModelOption(
                id="gemini-2.5-flash-lite",
                label="Gemini 2.5 Flash Lite",
                provider=LlmProvider.GEMINI,
            ),
        ]

    def available_models(self) -> list[ModelOption]:
        return list(self._models)

    def default_agent_model(self) -> str:
        return "gemini-2.5-flash"

    def default_other_teams_model(self) -> str:
        return "gemini-2.5-flash-lite"

    def validate_model(self, model_id: str) -> None:
        if model_id not in {option.id for option in self._models}:
            raise ValueError(f"Model '{model_id}' is unavailable.")

    def get(self, model_id: str) -> DraftAdvisorGateway:
        _ = model_id
        return self._gateway


class FakeSessionManager:
    """Session manager returning one fake session."""

    def __init__(self, session):
        self.session = session

    def get_or_create(self, session_id: str):
        _ = session_id
        return self.session


def _build_client(session) -> TestClient:
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    return TestClient(
        create_app(
            session_manager=FakeSessionManager(session),
            advisor_service=service,
            advisor_registry=registry,
        )
    )


def test_draft_advisor_returns_recommendation(config, player_catalog) -> None:
    """Verify advisor endpoint returns a structured recommendation."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    client = _build_client(session)

    response = client.post(
        "/api/draft/advisor",
        json={
            "team_id": 1,
            "trigger": "manual",
            "scope": "all_teams",
            "agent_model": "gemini-2.5-flash",
            "other_teams_model": "gemini-2.5-flash-lite",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_player_id"] == 2
    assert body["advising_team_id"] == 1
    assert body["degraded"] is False


def test_draft_advisor_accepts_ignore_player_ids(config, player_catalog) -> None:
    """Verify the advisor endpoint accepts blinded player ids."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    ignored_id = next(iter(session.available_player_ids - {2}))
    client = _build_client(session)

    response = client.post(
        "/api/draft/advisor",
        json={
            "team_id": 1,
            "trigger": "manual",
            "scope": "all_teams",
            "agent_model": "gemini-2.5-flash",
            "other_teams_model": "gemini-2.5-flash-lite",
            "ignore_player_ids": [ignored_id],
        },
    )

    assert response.status_code == 200
    assert response.json()["recommended_player_id"] == 2


def test_draft_advisor_auto_scope_returns_403_for_non_agent_team(config, player_catalog) -> None:
    """Verify auto requests respect agent-only scope on the server."""
    from draft_buddy.web.session import DraftSession

    config.draft.AGENT_START_POSITION = 1
    session = DraftSession(config)
    client = _build_client(session)

    response = client.post(
        "/api/draft/advisor",
        json={
            "team_id": 2,
            "trigger": "auto",
            "scope": "agent_only",
            "agent_model": "gemini-2.5-flash",
            "other_teams_model": "gemini-2.5-flash-lite",
        },
    )

    assert response.status_code == 403


def test_draft_advisor_manual_bypasses_scope_403(config, player_catalog) -> None:
    """Verify manual requests proceed even when team is not the agent team."""
    from draft_buddy.web.session import DraftSession

    config.draft.AGENT_START_POSITION = 1
    session = DraftSession(config)
    client = _build_client(session)

    response = client.post(
        "/api/draft/advisor",
        json={
            "team_id": 2,
            "trigger": "manual",
            "scope": "agent_only",
            "agent_model": "gemini-2.5-flash",
            "other_teams_model": "gemini-2.5-flash-lite",
        },
    )

    assert response.status_code == 200


def test_draft_advisor_returns_503_when_unconfigured(config, fake_session) -> None:
    """Verify missing advisor service returns 503."""
    client = TestClient(
        create_app(config=config, session_manager=FakeSessionManager(fake_session))
    )

    response = client.post("/api/draft/advisor", json={"trigger": "manual"})

    assert response.status_code == 503


def test_draft_advisor_models_endpoint(config, fake_session) -> None:
    """Verify models endpoint returns configured defaults."""
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(fake_session),
            advisor_service=service,
            advisor_registry=registry,
        )
    )

    response = client.get("/api/draft/advisor/models")

    assert response.status_code == 200
    body = response.json()
    assert len(body["models"]) == 2
    assert body["defaults"]["agent_model"] == "gemini-2.5-flash"


def test_draft_advisor_rejects_unavailable_model(config, player_catalog) -> None:
    """Verify invalid model ids return 400."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    client = _build_client(session)

    response = client.post(
        "/api/draft/advisor",
        json={
            "team_id": 1,
            "trigger": "manual",
            "agent_model": "deepseek/deepseek-v4-pro",
            "other_teams_model": "gemini-2.5-flash-lite",
        },
    )

    assert response.status_code == 400
