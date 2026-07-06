"""Tests for the draft assistant web endpoint."""

from __future__ import annotations

from draft_buddy.web.draft_advisor_schemas import PickRecommendation
from draft_buddy.web.draft_advisor_service import DraftAdvisorService
from draft_buddy.web.app import create_app
from fastapi.testclient import TestClient


class FakeAdvisorGateway:
    """Return a deterministic recommendation for tests."""

    def recommend(self, system_prompt: str, user_prompt: str) -> PickRecommendation:
        """Return a fixed recommendation."""
        _ = (system_prompt, user_prompt)
        return PickRecommendation(
            advising_team_id=1,
            is_agent_team=True,
            recommended_player_id=2,
            recommended_name="RB One",
            confidence="high",
            rationale_bullets=["Best RB available."],
            alternates=[],
            flags=["none"],
            unknown_factors=[],
        )


class FakeSessionManager:
    """Session manager returning one fake session."""

    def __init__(self, session):
        self.session = session

    def get_or_create(self, session_id: str):
        _ = session_id
        return self.session


def test_draft_advisor_returns_recommendation(config, player_catalog) -> None:
    """Verify advisor endpoint returns a structured recommendation."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    service = DraftAdvisorService(FakeAdvisorGateway())
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(session),
            advisor_service=service,
        )
    )

    response = client.post(
        "/api/draft/advisor",
        json={"team_id": 1, "trigger": "manual", "scope": "all_teams"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_player_id"] == 2
    assert body["advising_team_id"] == 1


def test_draft_advisor_auto_scope_returns_403_for_non_agent_team(config, player_catalog) -> None:
    """Verify auto requests respect agent-only scope on the server."""
    from draft_buddy.web.session import DraftSession

    config.draft.AGENT_START_POSITION = 1
    session = DraftSession(config)
    service = DraftAdvisorService(FakeAdvisorGateway())
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(session),
            advisor_service=service,
        )
    )

    response = client.post(
        "/api/draft/advisor",
        json={"team_id": 2, "trigger": "auto", "scope": "agent_only"},
    )

    assert response.status_code == 403


def test_draft_advisor_manual_bypasses_scope_403(config, player_catalog) -> None:
    """Verify manual requests proceed even when team is not the agent team."""
    from draft_buddy.web.session import DraftSession

    config.draft.AGENT_START_POSITION = 1
    session = DraftSession(config)
    service = DraftAdvisorService(FakeAdvisorGateway())
    client = TestClient(
        create_app(
            config=config,
            session_manager=FakeSessionManager(session),
            advisor_service=service,
        )
    )

    response = client.post(
        "/api/draft/advisor",
        json={"team_id": 2, "trigger": "manual", "scope": "agent_only"},
    )

    assert response.status_code == 200


def test_draft_advisor_returns_503_when_unconfigured(config, fake_session) -> None:
    """Verify missing advisor service returns 503."""
    client = TestClient(
        create_app(config=config, session_manager=FakeSessionManager(fake_session))
    )

    response = client.post("/api/draft/advisor", json={"trigger": "manual"})

    assert response.status_code == 503
