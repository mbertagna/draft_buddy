"""Tests for insight search date windows."""

from __future__ import annotations

from datetime import date

from draft_buddy.data.insights.query_builder import QueryKind
from draft_buddy.data.insights.search_windows import (
    start_date_for_query_kind,
    tavily_topic_for_query_kind,
    valyu_instructions_for_query_kind,
)


def test_outlook_start_date_is_march_first_of_draft_year() -> None:
    """Verify outlook/role windows start on March 1 of the draft year."""
    assert start_date_for_query_kind(QueryKind.OUTLOOK, 2026) == "2026-03-01"
    assert start_date_for_query_kind(QueryKind.ROLE, 2026) == "2026-03-01"


def test_injury_start_date_looks_back_ninety_days() -> None:
    """Verify injury recovery window uses a 90-day lookback."""
    assert (
        start_date_for_query_kind(
            QueryKind.INJURY_RECOVERY,
            2026,
            today=date(2026, 7, 16),
        )
        == "2026-04-17"
    )


def test_valyu_instructions_only_for_outlook_and_role() -> None:
    """Verify ranking instructions apply to outlook/role only."""
    assert valyu_instructions_for_query_kind(QueryKind.OUTLOOK) is not None
    assert valyu_instructions_for_query_kind(QueryKind.ROLE) is not None
    assert valyu_instructions_for_query_kind(QueryKind.INJURY_RECOVERY) is None


def test_tavily_topic_is_general_for_outlook_and_role() -> None:
    """Verify outlook and role queries use Tavily's general topic."""
    assert tavily_topic_for_query_kind(QueryKind.OUTLOOK) == "general"
    assert tavily_topic_for_query_kind(QueryKind.ROLE) == "general"


def test_tavily_topic_is_news_for_injury_recovery() -> None:
    """Verify injury-recovery queries use Tavily's news topic."""
    assert tavily_topic_for_query_kind(QueryKind.INJURY_RECOVERY) == "news"
