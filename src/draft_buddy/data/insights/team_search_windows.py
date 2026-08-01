"""Date windows and ranking instructions for team insight search queries."""

from __future__ import annotations

from draft_buddy.data.insights.search_windows import outlook_window_start
from draft_buddy.data.insights.team_query_builder import TeamQueryKind

VALYU_TEAM_INSTRUCTIONS = (
    "Prefer season-long team analysis and outlook articles for the draft year. "
    "Deprioritize single-game recaps and stale prior-season summaries."
)


def team_start_date_for_query_kind(query_kind: TeamQueryKind, draft_year: int) -> str:
    """Return the Valyu ``start_date`` (YYYY-MM-DD) for a team query kind.

    All team query kinds share a single offseason window; there is no
    injury-style lookback for team-level content.

    Parameters
    ----------
    query_kind : TeamQueryKind
        Team search query kind.
    draft_year : int
        Draft season year.

    Returns
    -------
    str
        Inclusive start date as ``YYYY-MM-DD``.
    """
    _ = query_kind
    return outlook_window_start(draft_year)


def valyu_instructions_for_team_query_kind(query_kind: TeamQueryKind) -> str | None:
    """Return Valyu ranking instructions for a team query kind.

    Parameters
    ----------
    query_kind : TeamQueryKind
        Team search query kind.

    Returns
    -------
    str or None
        Instructions for Valyu, applied uniformly to all team query kinds.
    """
    _ = query_kind
    return VALYU_TEAM_INSTRUCTIONS
