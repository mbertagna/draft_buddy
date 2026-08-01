"""Gateway interfaces for the live draft assistant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from draft_buddy.web.draft_advisor_schemas import PickRecommendation, parse_pick_recommendation

DEFAULT_ADVISOR_MODEL = "gemini-2.5-flash"


class DraftAdvisorGateway(ABC):
    """Abstract interface for structured draft assistant recommendations."""

    @abstractmethod
    def fetch_payload(self, system_prompt: str, user_prompt: str) -> Tuple[Dict[str, Any], str]:
        """Return parsed JSON and raw model text for one advisor request.

        Parameters
        ----------
        system_prompt : str
            System instructions for the assistant.
        user_prompt : str
            Markdown draft context payload.

        Returns
        -------
        Tuple[Dict[str, Any], str]
            Parsed JSON payload and raw response text.
        """

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
        payload, _raw_content = self.fetch_payload(system_prompt, user_prompt)
        return parse_pick_recommendation(payload)


# Backward-compatible alias
GeminiAdvisorGateway = DraftAdvisorGateway


class GeminiFlashAdvisorGateway(DraftAdvisorGateway):
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

    @property
    def model(self) -> str:
        """Return the configured Gemini model name."""
        return self._model

    def fetch_payload(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str]:
        """Call Gemini Flash and return structured JSON plus raw text.

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
        import json

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
        return json.loads(text), text

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
        payload, _raw_content = self.fetch_payload(system_prompt, user_prompt)
        return parse_pick_recommendation(payload)
