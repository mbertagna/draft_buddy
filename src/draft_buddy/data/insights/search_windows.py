"""Date windows and ranking instructions for insight search queries."""

from __future__ import annotations

from datetime import date, timedelta

from draft_buddy.data.insights.query_builder import QueryKind

OUTLOOK_ROLE_MONTH = 3
OUTLOOK_ROLE_DAY = 1
INJURY_LOOKBACK_DAYS = 90

VALYU_OUTLOOK_INSTRUCTIONS = (
    "Prefer season fantasy outlooks for the draft year. "
    "Deprioritize mid-season weekly injury blurbs and unrelated listicles."
)


def start_date_for_query_kind(
    query_kind: QueryKind,
    draft_year: int,
    today: date | None = None,
) -> str:
    """Return the Valyu ``start_date`` (YYYY-MM-DD) for a query kind.

    Parameters
    ----------
    query_kind : QueryKind
        Search query kind.
    draft_year : int
        Draft season year.
    today : date, optional
        Reference date for injury lookback; defaults to UTC today.

    Returns
    -------
    str
        Inclusive start date as ``YYYY-MM-DD``.
    """
    if query_kind == QueryKind.INJURY_RECOVERY:
        reference = today or date.today()
        return (reference - timedelta(days=INJURY_LOOKBACK_DAYS)).isoformat()
    return date(draft_year, OUTLOOK_ROLE_MONTH, OUTLOOK_ROLE_DAY).isoformat()


def outlook_window_start(draft_year: int) -> str:
    """Return the offseason outlook window start for ``draft_year``.

    Parameters
    ----------
    draft_year : int
        Draft season year.

    Returns
    -------
    str
        ``YYYY-MM-DD`` for March 1 of the draft year.
    """
    return date(draft_year, OUTLOOK_ROLE_MONTH, OUTLOOK_ROLE_DAY).isoformat()


def injury_window_start(today: date | None = None) -> str:
    """Return the injury-recovery lookback start date.

    Parameters
    ----------
    today : date, optional
        Reference date; defaults to local today.

    Returns
    -------
    str
        ``YYYY-MM-DD`` for today minus the injury lookback days.
    """
    reference = today or date.today()
    return (reference - timedelta(days=INJURY_LOOKBACK_DAYS)).isoformat()


def valyu_instructions_for_query_kind(query_kind: QueryKind) -> str | None:
    """Return Valyu ranking instructions for outlook/role queries.

    Parameters
    ----------
    query_kind : QueryKind
        Search query kind.

    Returns
    -------
    str or None
        Instructions for Valyu, or ``None`` when not applicable.
    """
    if query_kind in (QueryKind.OUTLOOK, QueryKind.ROLE):
        return VALYU_OUTLOOK_INSTRUCTIONS
    return None
