"""Gemini Flash implementation for structured player insight synthesis."""

from __future__ import annotations

import json
from typing import Any

from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.data.insights.schemas import PlayerInsight, sanitize_synthesis_payload
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

    def synthesize(self, system_prompt: str, user_prompt: str) -> PlayerInsight:
        """Call Gemini Flash and parse a structured PlayerInsight response.

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
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=PlayerInsight,
                temperature=0.2,
            ),
        )
        text = response.text or "{}"
        payload: dict[str, Any] = json.loads(text)
        return PlayerInsight.model_validate(sanitize_synthesis_payload(payload))
