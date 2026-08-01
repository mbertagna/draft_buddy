"""Tests for draft advisor service model routing."""

from __future__ import annotations

import pytest

from draft_buddy.llm.model_registry import LlmProvider, ModelOption
from draft_buddy.web.draft_advisor_schemas import AdvisorRequest, PickRecommendation
from draft_buddy.web.draft_advisor_service import DraftAdvisorService, DraftAdvisorValidationError
from draft_buddy.web.draft_advisor_gateway import DraftAdvisorGateway


class RecordingAdvisorGateway(DraftAdvisorGateway):
    """Capture the model gateway used for a recommendation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.last_user_prompt: str | None = None

    def fetch_payload(self, system_prompt: str, user_prompt: str) -> tuple[dict, str]:
        """Return a fixed payload and record the call."""
        self.calls += 1
        self.last_user_prompt = user_prompt
        _ = system_prompt
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
        """Return a fixed recommendation and record the call."""
        from draft_buddy.web.draft_advisor_schemas import parse_pick_recommendation

        payload, _raw = self.fetch_payload(system_prompt, user_prompt)
        return parse_pick_recommendation(payload)


class FakeAdvisorRegistry:
    """Minimal registry mapping model ids to recording gateways."""

    def __init__(self) -> None:
        self.agent_gateway = RecordingAdvisorGateway("agent")
        self.other_gateway = RecordingAdvisorGateway("other")
        self._models = [
            ModelOption(id="agent-model", label="Agent", provider=LlmProvider.GEMINI),
            ModelOption(id="other-model", label="Other", provider=LlmProvider.GEMINI),
        ]

    def available_models(self) -> list[ModelOption]:
        return list(self._models)

    def default_agent_model(self) -> str:
        return "agent-model"

    def default_other_teams_model(self) -> str:
        return "other-model"

    def validate_model(self, model_id: str) -> None:
        if model_id not in {"agent-model", "other-model"}:
            raise ValueError(f"Model '{model_id}' is unavailable.")

    def get(self, model_id: str) -> DraftAdvisorGateway:
        if model_id == "agent-model":
            return self.agent_gateway
        return self.other_gateway


def test_recommend_uses_agent_model_for_agent_team(config, player_catalog) -> None:
    """Verify agent team requests route to the agent model gateway."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    request = AdvisorRequest(
        team_id=session.agent_team_id,
        agent_model="agent-model",
        other_teams_model="other-model",
    )

    service.recommend(session, request, insights={})

    assert registry.agent_gateway.calls == 1
    assert registry.other_gateway.calls == 0


def test_recommend_uses_other_model_for_non_agent_team(config, player_catalog) -> None:
    """Verify non-agent team requests route to the other teams model gateway."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    other_team = 2 if session.agent_team_id != 2 else 3
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    request = AdvisorRequest(
        team_id=other_team,
        agent_model="agent-model",
        other_teams_model="other-model",
    )

    service.recommend(session, request, insights={})

    assert registry.other_gateway.calls == 1
    assert registry.agent_gateway.calls == 0


def test_recommend_rejects_unavailable_model(config, player_catalog) -> None:
    """Verify unavailable model ids return a validation error."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    request = AdvisorRequest(
        team_id=session.agent_team_id,
        agent_model="missing-model",
        other_teams_model="other-model",
    )

    with pytest.raises(DraftAdvisorValidationError, match="unavailable"):
        service.recommend(session, request, insights={})


def test_recommend_excludes_ignored_player_from_context(config, player_catalog) -> None:
    """Verify blinded players are omitted from assistant candidate context."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    available_ids = session.available_player_ids
    ignored_id = next(iter(available_ids - {2})) if len(available_ids) > 1 else next(iter(available_ids))
    ignored_player = session.player_catalog.require(ignored_id)
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    request = AdvisorRequest(
        team_id=session.agent_team_id,
        agent_model="agent-model",
        other_teams_model="other-model",
        ignore_player_ids=[ignored_id],
    )

    # Recommendation may degrade if fixed id 2 was the ignored player; context is the assert.
    service.recommend(session, request, insights={})

    prompt = registry.agent_gateway.last_user_prompt
    assert prompt is not None
    assert f"| {ignored_id} | {ignored_player.name} |" not in prompt


def test_recommend_rejects_when_all_players_ignored(config, player_catalog) -> None:
    """Verify ignoring the full available pool raises a validation error."""
    from draft_buddy.web.session import DraftSession

    session = DraftSession(config)
    registry = FakeAdvisorRegistry()
    service = DraftAdvisorService(registry)
    request = AdvisorRequest(
        team_id=session.agent_team_id,
        agent_model="agent-model",
        other_teams_model="other-model",
        ignore_player_ids=list(session.available_player_ids),
    )

    with pytest.raises(DraftAdvisorValidationError, match="No players remain"):
        service.recommend(session, request, insights={})
