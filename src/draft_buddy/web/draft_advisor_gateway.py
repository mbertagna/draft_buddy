"""Gemini gateway for the live draft assistant."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from draft_buddy.web.draft_advisor_schemas import PickRecommendation

DEFAULT_ADVISOR_MODEL = "gemini-2.5-flash"


class GeminiAdvisorGateway(ABC):
    """Abstract interface for structured draft assistant recommendations."""

    @abstractmethod
    def recommend(self, system_prompt: str, user_prompt: str) -> PickRecommendation:
        """Return a structured pick recommendation.

        Parameters
        ----------
        system_prompt : str
            System instructions for the assistant.
        user_prompt : str
            Markdown draft context payload.

        Returns
        -------
        PickRecommendation
            Parsed recommendation.
        """


class GeminiFlashAdvisorGateway(GeminiAdvisorGateway):
    """Gemini Flash implementation using structured JSON output."""

    def __init__(self, api_key: str, model: str = DEFAULT_ADVISOR_MODEL) -> None:
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

    def recommend(self, system_prompt: str, user_prompt: str) -> PickRecommendation:
        """Call Gemini Flash and parse a structured recommendation.

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
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=PickRecommendation,
                temperature=0.2,
            ),
        )
        text = response.text or "{}"
        payload = json.loads(text)
        return PickRecommendation.model_validate(payload)
