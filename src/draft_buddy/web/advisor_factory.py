"""Factory for draft assistant LLM gateways."""

from __future__ import annotations

from typing import Dict, List, Optional

from draft_buddy.llm.model_registry import (
    LlmProvider,
    ModelOption,
    available_models,
    default_advisor_agent_model,
    default_advisor_other_teams_model,
    infer_provider,
    lookup_model,
    provider_api_key_env,
    resolve_provider_api_key,
)
from draft_buddy.web.draft_advisor_gateway import DraftAdvisorGateway, GeminiFlashAdvisorGateway
from draft_buddy.web.openrouter_advisor_gateway import OpenRouterAdvisorGateway


class AdvisorGatewayRegistry:
    """Cache and resolve draft advisor gateways by model id."""

    def __init__(self, models: Optional[List[ModelOption]] = None) -> None:
        """
        Parameters
        ----------
        models : list[ModelOption], optional
            Pre-filtered available models. Defaults to env-configured models.
        """
        self._models = models if models is not None else available_models()
        self._model_ids = {option.id for option in self._models}
        self._gateways: Dict[str, DraftAdvisorGateway] = {}

    def available_models(self) -> List[ModelOption]:
        """Return models available for advisor requests."""
        return list(self._models)

    def default_agent_model(self) -> str:
        """Return the default model for the agent team."""
        return self._coerce_available(default_advisor_agent_model())

    def default_other_teams_model(self) -> str:
        """Return the default model for non-agent teams."""
        return self._coerce_available(default_advisor_other_teams_model())

    def validate_model(self, model_id: str) -> None:
        """Ensure a model id is configured and available.

        Raises
        ------
        ValueError
            When the model is unsupported or unavailable.
        """
        lookup_model(model_id)
        if model_id not in self._model_ids:
            available = ", ".join(sorted(self._model_ids))
            raise ValueError(
                f"Model '{model_id}' is unavailable. Configure its provider API key. "
                f"Available models: {available}"
            )

    def get(self, model_id: str) -> DraftAdvisorGateway:
        """Return a cached gateway for the requested model.

        Parameters
        ----------
        model_id : str
            Canonical model identifier.

        Returns
        -------
        DraftAdvisorGateway
            Configured advisor gateway.
        """
        self.validate_model(model_id)
        if model_id not in self._gateways:
            self._gateways[model_id] = _build_advisor_gateway(model_id)
        return self._gateways[model_id]

    def _coerce_available(self, model_id: str) -> str:
        """Return model_id when available, otherwise the first available model."""
        lookup_model(model_id)
        if model_id in self._model_ids:
            return model_id
        if not self._models:
            return model_id
        return self._models[0].id


def build_advisor_registry() -> Optional[AdvisorGatewayRegistry]:
    """Build an advisor gateway registry when any LLM provider is configured.

    Returns
    -------
    AdvisorGatewayRegistry | None
        Registry when at least one model is available, else ``None``.
    """
    models = available_models()
    if not models:
        return None
    return AdvisorGatewayRegistry(models=models)


def _build_advisor_gateway(model_id: str) -> DraftAdvisorGateway:
    """Construct a single advisor gateway for a supported model."""
    provider = infer_provider(model_id)
    api_key = resolve_provider_api_key(provider)
    if not api_key:
        raise ValueError(
            f"{provider_api_key_env(provider)} environment variable is required "
            f"for model '{model_id}'."
        )
    if provider == LlmProvider.GEMINI:
        return GeminiFlashAdvisorGateway(api_key=api_key, model=model_id)
    return OpenRouterAdvisorGateway(api_key=api_key, model=model_id)
