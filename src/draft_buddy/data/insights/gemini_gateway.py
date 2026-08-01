"""Abstract gateway interface for structured player insight synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod

from draft_buddy.data.insights.schemas import PlayerInsight


class InsightSynthesisGateway(ABC):
    """Abstract interface for structured player insight synthesis."""

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


# Backward-compatible alias
GeminiGateway = InsightSynthesisGateway
