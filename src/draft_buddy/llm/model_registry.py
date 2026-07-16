"""Canonical LLM model registry and provider resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class LlmProvider(str, Enum):
    """Supported LLM backend providers."""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_LITE_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_AGENT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL = "deepseek/deepseek-v4-flash"

SUPPORTED_PROVIDERS = frozenset({LlmProvider.GEMINI, LlmProvider.OPENROUTER})

_MODEL_SPECS: Dict[str, tuple[LlmProvider, str]] = {
    DEFAULT_GEMINI_MODEL: (LlmProvider.GEMINI, "Gemini 2.5 Flash"),
    DEFAULT_GEMINI_LITE_MODEL: (LlmProvider.GEMINI, "Gemini 2.5 Flash Lite"),
    DEFAULT_OPENROUTER_AGENT_MODEL: (LlmProvider.OPENROUTER, "DeepSeek V4 Pro"),
    DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL: (LlmProvider.OPENROUTER, "DeepSeek V4 Flash"),
}

_PROVIDER_ENV_KEYS: Dict[LlmProvider, str] = {
    LlmProvider.GEMINI: "GEMINI_API_KEY",
    LlmProvider.OPENROUTER: "OPENROUTER_API_KEY",
}


@dataclass(frozen=True, slots=True)
class ModelOption:
    """One selectable LLM model exposed to clients."""

    id: str
    label: str
    provider: LlmProvider


def lookup_model(model_id: str) -> ModelOption:
    """Return metadata for a supported model id.

    Parameters
    ----------
    model_id : str
        Canonical model identifier.

    Returns
    -------
    ModelOption
        Model metadata.

    Raises
    ------
    ValueError
        When the model id is unsupported.
    """
    spec = _MODEL_SPECS.get(model_id)
    if spec is None:
        supported = ", ".join(sorted(_MODEL_SPECS))
        raise ValueError(f"Unsupported model '{model_id}'. Choose from: {supported}")
    provider, label = spec
    return ModelOption(id=model_id, label=label, provider=provider)


def infer_provider(model_id: str) -> LlmProvider:
    """Infer the provider for a supported model id."""
    return lookup_model(model_id).provider


def provider_api_key_env(provider: LlmProvider) -> str:
    """Return the environment variable name for a provider API key."""
    return _PROVIDER_ENV_KEYS[provider]


def resolve_provider_api_key(provider: LlmProvider) -> Optional[str]:
    """Return the configured API key for a provider, if any."""
    value = os.environ.get(provider_api_key_env(provider), "").strip()
    return value or None


def provider_is_configured(provider: LlmProvider) -> bool:
    """Return whether the provider has a non-empty API key configured."""
    return resolve_provider_api_key(provider) is not None


def available_models() -> List[ModelOption]:
    """Return models whose provider API keys are configured."""
    options: List[ModelOption] = []
    for model_id in sorted(_MODEL_SPECS):
        option = lookup_model(model_id)
        if provider_is_configured(option.provider):
            options.append(option)
    return options


def default_advisor_agent_model() -> str:
    """Return the default advisor model for the agent team."""
    explicit = os.environ.get("ADVISOR_AGENT_MODEL", "").strip()
    if explicit:
        return explicit
    legacy = os.environ.get("ADVISOR_GEMINI_MODEL", "").strip()
    if legacy:
        return legacy
    if provider_is_configured(LlmProvider.GEMINI):
        return DEFAULT_GEMINI_LITE_MODEL
    if provider_is_configured(LlmProvider.OPENROUTER):
        return DEFAULT_OPENROUTER_AGENT_MODEL
    return DEFAULT_GEMINI_LITE_MODEL


def default_advisor_other_teams_model() -> str:
    """Return the default advisor model for non-agent teams."""
    explicit = os.environ.get("ADVISOR_OTHER_TEAMS_MODEL", "").strip()
    if explicit:
        return explicit
    if provider_is_configured(LlmProvider.GEMINI):
        return DEFAULT_GEMINI_LITE_MODEL
    if provider_is_configured(LlmProvider.OPENROUTER):
        return DEFAULT_OPENROUTER_OTHER_TEAMS_MODEL
    return DEFAULT_GEMINI_LITE_MODEL


def default_synthesis_model() -> str:
    """Return the default synthesis model id."""
    for env_name in ("INSIGHTS_LLM_MODEL", "INSIGHTS_GEMINI_MODEL"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_GEMINI_MODEL


def resolve_synthesis_provider(cli_provider: Optional[str] = None) -> LlmProvider:
    """Resolve synthesis provider from CLI flag or environment.

    Parameters
    ----------
    cli_provider : str, optional
        Explicit provider from ``--provider``.

    Returns
    -------
    LlmProvider
        Resolved provider enum value.
    """
    provider_name = (cli_provider or os.environ.get("INSIGHTS_LLM_PROVIDER") or "").strip().lower()
    if not provider_name:
        return infer_provider(default_synthesis_model())
    if provider_name not in {provider.value for provider in SUPPORTED_PROVIDERS}:
        supported = ", ".join(sorted(provider.value for provider in SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported synthesis provider '{provider_name}'. Choose from: {supported}"
        )
    return LlmProvider(provider_name)
