"""Tests for draft advisor gateway registry."""

from __future__ import annotations

import pytest

from draft_buddy.llm.model_registry import DEFAULT_GEMINI_MODEL, DEFAULT_OPENROUTER_AGENT_MODEL
from draft_buddy.web.advisor_factory import AdvisorGatewayRegistry, build_advisor_registry
from draft_buddy.web.draft_advisor_gateway import GeminiFlashAdvisorGateway


def test_build_advisor_registry_returns_none_without_keys(monkeypatch) -> None:
    """Verify registry is unavailable when no provider keys are configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert build_advisor_registry() is None


def test_advisor_registry_caches_gateways(monkeypatch) -> None:
    """Verify advisor gateways are cached per model id."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    class StubAdvisorGateway(GeminiFlashAdvisorGateway):
        def __init__(self, api_key: str, model: str) -> None:
            self._model = model

    monkeypatch.setattr(
        "draft_buddy.web.advisor_factory.GeminiFlashAdvisorGateway",
        StubAdvisorGateway,
    )
    registry = build_advisor_registry()
    assert registry is not None
    first = registry.get(DEFAULT_GEMINI_MODEL)
    second = registry.get(DEFAULT_GEMINI_MODEL)
    assert first is second
    assert isinstance(first, StubAdvisorGateway)


def test_advisor_registry_rejects_unavailable_model(monkeypatch) -> None:
    """Verify unavailable models raise a validation error."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    registry = build_advisor_registry()
    assert registry is not None
    with pytest.raises(ValueError, match="unavailable"):
        registry.validate_model(DEFAULT_OPENROUTER_AGENT_MODEL)
