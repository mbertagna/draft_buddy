"""OpenRouter gateway for the live draft assistant."""

from __future__ import annotations

from typing import Any

from draft_buddy.llm.openrouter_client import OpenRouterClient
from draft_buddy.web.draft_advisor_gateway import DraftAdvisorGateway
from draft_buddy.web.draft_advisor_schemas import PickRecommendation, parse_pick_recommendation


class OpenRouterAdvisorGateway(DraftAdvisorGateway):
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

    def fetch_payload(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
        """Call OpenRouter and return structured JSON plus raw text.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            Markdown context payload.

        Returns
        -------
        tuple[dict[str, Any], str]
            Parsed JSON payload and raw response text.
        """
        return self._client.generate_structured_with_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=PickRecommendation,
            schema_name="pick_recommendation",
            reasoning_effort="low",
            use_response_healing=False,
        )

    def recommend(self, system_prompt: str, user_prompt: str) -> PickRecommendation:
        """Call OpenRouter and parse a structured recommendation.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            Markdown context payload.

        Returns
        -------
        PickRecommendation
            Parsed recommendation.
        """
        payload, _raw_content = self.fetch_payload(system_prompt, user_prompt)
        return parse_pick_recommendation(payload)
