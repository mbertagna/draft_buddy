"""OpenRouter implementation for structured player insight synthesis."""

from __future__ import annotations

from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.data.insights.schemas import PlayerInsight, sanitize_synthesis_payload
from draft_buddy.llm.openrouter_client import SYNTHESIS_MAX_TOKENS, OpenRouterClient


class OpenRouterSynthesisGateway(InsightSynthesisGateway):
    """OpenRouter implementation using strict JSON schema output."""

    def __init__(self, api_key: str, model: str) -> None:
        """
        Parameters
        ----------
        api_key : str
            OpenRouter API key.
        model : str
            OpenRouter model slug.
        """
        self._client = OpenRouterClient(api_key=api_key, model=model)

    @property
    def model(self) -> str:
        """Return the configured OpenRouter model slug."""
        return self._client.model

    def synthesize(self, system_prompt: str, user_prompt: str) -> PlayerInsight:
        """Call OpenRouter and parse a structured PlayerInsight response.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User content with player context and snippets.

        Returns
        -------
        PlayerInsight
            Parsed structured insight.
        """
        payload = self._client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=PlayerInsight,
            schema_name="player_insight",
            max_tokens=SYNTHESIS_MAX_TOKENS,
            reasoning_effort="none",
        )
        return PlayerInsight.model_validate(sanitize_synthesis_payload(payload))
