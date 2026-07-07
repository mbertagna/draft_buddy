"""Tests for LLM model registry."""

from __future__ import annotations

import pytest

from draft_buddy.llm.model_registry import (
    DEFAULT_GEMINI_LITE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_AGENT_MODEL,
    DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL,
    LlmProvider,
    available_models,
    default_advisor_agent_model,
    default_advisor_other_teams_model,
    default_synthesis_model,
    infer_provider,
    lookup_model,
    resolve_synthesis_provider,
)


def test_lookup_model_returns_metadata() -> None:
    """Verify supported model metadata is returned."""
    option = lookup_model(DEFAULT_GEMINI_MODEL)
    assert option.id == DEFAULT_GEMINI_MODEL
    assert option.provider == LlmProvider.GEMINI


def test_lookup_model_rejects_unknown_model() -> None:
    """Verify unsupported model ids raise a clear error."""
    with pytest.raises(ValueError, match="Unsupported model"):
        lookup_model("unknown/model")


def test_infer_provider_for_openrouter_model() -> None:
    """Verify OpenRouter models map to the openrouter provider."""
    assert infer_provider(DEFAULT_OPENROUTER_AGENT_MODEL) == LlmProvider.OPENROUTER


def test_available_models_filters_by_api_keys(monkeypatch) -> None:
    """Verify only configured providers appear in available models."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    options = available_models()
    assert all(option.provider == LlmProvider.GEMINI for option in options)


def test_default_advisor_models_prefer_openrouter_when_configured(monkeypatch) -> None:
    """Verify OpenRouter defaults apply when its API key is present."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("ADVISOR_AGENT_MODEL", raising=False)
    monkeypatch.delenv("ADVISOR_OTHER_TEAMS_MODEL", raising=False)
    monkeypatch.delenv("ADVISOR_GEMINI_MODEL", raising=False)
    assert default_advisor_agent_model() == DEFAULT_OPENROUTER_AGENT_MODEL
    assert default_advisor_other_teams_model() == DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL


def test_default_advisor_models_fall_back_to_gemini(monkeypatch) -> None:
    """Verify Gemini defaults apply when OpenRouter is not configured."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("ADVISOR_AGENT_MODEL", raising=False)
    monkeypatch.delenv("ADVISOR_OTHER_TEAMS_MODEL", raising=False)
    monkeypatch.delenv("ADVISOR_GEMINI_MODEL", raising=False)
    assert default_advisor_agent_model() == DEFAULT_GEMINI_MODEL
    assert default_advisor_other_teams_model() == DEFAULT_GEMINI_LITE_MODEL


def test_default_synthesis_model_honors_env_precedence(monkeypatch) -> None:
    """Verify synthesis model env resolution order."""
    monkeypatch.setenv("INSIGHTS_LLM_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("INSIGHTS_GEMINI_MODEL", "gemini-2.5-flash-lite")
    assert default_synthesis_model() == "deepseek/deepseek-v4-flash"


def test_resolve_synthesis_provider_from_env(monkeypatch) -> None:
    """Verify synthesis provider can be resolved from environment."""
    monkeypatch.setenv("INSIGHTS_LLM_PROVIDER", "openrouter")
    assert resolve_synthesis_provider() == LlmProvider.OPENROUTER
