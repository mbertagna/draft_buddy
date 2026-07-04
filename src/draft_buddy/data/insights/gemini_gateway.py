"""Abstract Gemini gateway interface for structured player insight synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod

from draft_buddy.data.insights.schemas import PlayerInsight


class GeminiGateway(ABC):
    """Abstract interface for Gemini structured synthesis."""

    @abstractmethod
    def synthesize(self, system_prompt: str, user_prompt: str) -> PlayerInsight:
        """Synthesize a player insight from prompts.

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
