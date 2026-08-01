"""Abstract gateway interface for structured LLM synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class InsightSynthesisGateway(ABC):
    """Abstract interface for structured synthesis over an arbitrary schema."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
    ) -> dict[str, Any]:
        """Synthesize raw structured JSON matching a Pydantic schema.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User content with entity context and snippets.
        response_model : Type[BaseModel]
            Pydantic model defining the expected response shape.
        schema_name : str
            Schema name used by providers that require one.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload, not yet sanitized or validated.
        """


# Backward-compatible alias
GeminiGateway = InsightSynthesisGateway
