"""Gemini Flash implementation for structured LLM synthesis."""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel

from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.llm.model_registry import DEFAULT_GEMINI_MODEL


class GeminiFlashGateway(InsightSynthesisGateway):
    """Gemini Flash implementation using structured JSON output."""

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        """
        Parameters
        ----------
        api_key : str
            Gemini API key.
        model : str, optional
            Gemini model name.
        """
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        """Return the configured Gemini model name."""
        return self._model

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
    ) -> dict[str, Any]:
        """Call Gemini Flash and return the raw structured JSON payload.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User content with entity context and snippets.
        response_model : Type[BaseModel]
            Pydantic model defining the expected response shape.
        schema_name : str
            Unused by Gemini; present for interface parity with OpenRouter.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload, not yet sanitized or validated.
        """
        _ = schema_name
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=0.2,
            ),
        )
        text = response.text or "{}"
        return json.loads(text)
