"""OpenRouter implementation for structured LLM synthesis."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
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

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
    ) -> dict[str, Any]:
        """Call OpenRouter and return the raw structured JSON payload.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User content with entity context and snippets.
        response_model : Type[BaseModel]
            Pydantic model defining the expected response shape.
        schema_name : str
            Schema name for OpenRouter's ``json_schema`` response format.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload, not yet sanitized or validated.
        """
        return self._client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            schema_name=schema_name,
            max_tokens=SYNTHESIS_MAX_TOKENS,
            reasoning_effort="none",
        )
