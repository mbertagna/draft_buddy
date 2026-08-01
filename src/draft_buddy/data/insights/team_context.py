"""Team context record for insight enrichment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InsightTeamContext:
    """NFL team used as context for search and synthesis.

    Parameters
    ----------
    team_abbr : str
        NFL team abbreviation (e.g. ``SF``).
    draft_year : int
        Draft season year used in search query templates.
    """

    team_abbr: str
    draft_year: int
