"""Tests for synthesis LLM factory."""

from __future__ import annotations

import pytest

from draft_buddy.data.insights.gemini_flash_gateway import GeminiFlashGateway
from draft_buddy.data.insights.openrouter_synthesis_gateway import OpenRouterSynthesisGateway
from draft_buddy.data.insights.synthesis_factory import build_synthesis_gateway, resolve_synthesis_model
from draft_buddy.llm.model_registry import DEFAULT_GEMINI_MODEL, DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL


def test_resolve_synthesis_model_prefers_cli(monkeypatch) -> None:
    """Verify CLI model overrides environment defaults."""
    monkeypatch.setenv("INSIGHTS_LLM_MODEL", DEFAULT_GEMINI_MODEL)
    assert resolve_synthesis_model(DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL) == DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL


def test_build_synthesis_gateway_gemini(monkeypatch) -> None:
    """Verify Gemini synthesis gateway is built for Gemini models."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    class StubGeminiGateway(GeminiFlashGateway):
        def __init__(self, api_key: str, model: str) -> None:
            self._model = model

    monkeypatch.setattr(
        "draft_buddy.data.insights.synthesis_factory.GeminiFlashGateway",
        StubGeminiGateway,
    )
    gateway = build_synthesis_gateway(DEFAULT_GEMINI_MODEL)
    assert isinstance(gateway, StubGeminiGateway)


def test_build_synthesis_gateway_openrouter(monkeypatch) -> None:
    """Verify OpenRouter synthesis gateway is built for OpenRouter models."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    gateway = build_synthesis_gateway(DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL)
    assert isinstance(gateway, OpenRouterSynthesisGateway)


def test_build_synthesis_gateway_requires_matching_provider(monkeypatch) -> None:
    """Verify provider override must match the model provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    with pytest.raises(ValueError, match="belongs to provider"):
        build_synthesis_gateway(DEFAULT_GEMINI_MODEL, provider="openrouter")


def test_build_synthesis_gateway_requires_api_key(monkeypatch) -> None:
    """Verify missing provider API keys raise a clear error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_synthesis_gateway(DEFAULT_GEMINI_MODEL)
