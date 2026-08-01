"""Factory for insight synthesis LLM gateways."""

from __future__ import annotations

from typing import Optional

from draft_buddy.data.insights.gemini_flash_gateway import GeminiFlashGateway
from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.data.insights.openrouter_synthesis_gateway import OpenRouterSynthesisGateway
from draft_buddy.llm.model_registry import (
    LlmProvider,
    default_synthesis_model,
    infer_provider,
    lookup_model,
    provider_api_key_env,
    resolve_provider_api_key,
    resolve_synthesis_provider,
)


def resolve_synthesis_model(cli_model: Optional[str] = None) -> str:
    """Resolve the synthesis model from CLI flag or environment.

    Parameters
    ----------
    cli_model : str, optional
        Explicit model from ``--model``.

    Returns
    -------
    str
        Canonical model identifier.
    """
    model = cli_model or default_synthesis_model()
    lookup_model(model)
    return model


def build_synthesis_gateway(
    model: str,
    provider: Optional[str] = None,
) -> InsightSynthesisGateway:
    """Build a synthesis gateway for the requested model.

    Parameters
    ----------
    model : str
        Canonical model identifier.
    provider : str, optional
        Explicit provider override (``gemini`` or ``openrouter``).

    Returns
    -------
    InsightSynthesisGateway
        Configured synthesis gateway.

    Raises
    ------
    ValueError
        When the model or provider is unsupported or API keys are missing.
    """
    option = lookup_model(model)
    if provider:
        resolved_provider = resolve_synthesis_provider(provider)
    else:
        resolved_provider = infer_provider(model)

    if resolved_provider != option.provider:
        raise ValueError(
            f"Model '{model}' belongs to provider '{option.provider.value}', "
            f"but provider '{resolved_provider.value}' was requested."
        )

    api_key = resolve_provider_api_key(resolved_provider)
    if not api_key:
        env_name = provider_api_key_env(resolved_provider)
        raise ValueError(
            f"{env_name} environment variable is required for model '{model}'."
        )

    if resolved_provider == LlmProvider.GEMINI:
        return GeminiFlashGateway(api_key=api_key, model=model)
    return OpenRouterSynthesisGateway(api_key=api_key, model=model)
