"""Tests for team insight search query templates."""

from __future__ import annotations

from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_query_builder import (
    LeagueInsightQueryBuilder,
    TeamInsightQueryBuilder,
    TeamQueryKind,
)


def _team(**overrides) -> InsightTeamContext:
    """Build a default team context with optional overrides."""
    defaults = {"team_abbr": "SF", "draft_year": 2026}
    defaults.update(overrides)
    return InsightTeamContext(**defaults)


def test_build_queries_returns_all_four_kinds_in_order() -> None:
    """Verify all four query kinds are built, in outlook/offense/defense/schedule order."""
    builder = TeamInsightQueryBuilder()
    queries = builder.build_queries(_team())

    assert [query.kind for query in queries] == [
        TeamQueryKind.OUTLOOK,
        TeamQueryKind.OFFENSE,
        TeamQueryKind.DEFENSE,
        TeamQueryKind.SCHEDULE,
    ]


def test_queries_include_team_abbreviation_and_draft_year() -> None:
    """Verify every query text includes the team abbreviation and draft year."""
    builder = TeamInsightQueryBuilder()
    queries = builder.build_queries(_team(team_abbr="KC", draft_year=2027))

    for query in queries:
        assert "KC" in query.text
        assert "2027" in query.text


def test_offense_query_mentions_offensive_line() -> None:
    """Verify the offense query targets offensive-line and skill-position talent."""
    builder = TeamInsightQueryBuilder()
    queries = builder.build_queries(_team())

    offense_query = next(query for query in queries if query.kind == TeamQueryKind.OFFENSE)
    assert "offensive line" in offense_query.text
    assert "skill position" in offense_query.text


def test_defense_query_mentions_pass_rush_and_secondary() -> None:
    """Verify the defense query targets pass rush and secondary."""
    builder = TeamInsightQueryBuilder()
    queries = builder.build_queries(_team())

    defense_query = next(query for query in queries if query.kind == TeamQueryKind.DEFENSE)
    assert "pass rush" in defense_query.text
    assert "secondary" in defense_query.text


def test_schedule_query_mentions_strength_of_schedule() -> None:
    """Verify the schedule query targets strength-of-schedule difficulty."""
    builder = TeamInsightQueryBuilder()
    queries = builder.build_queries(_team())

    schedule_query = next(query for query in queries if query.kind == TeamQueryKind.SCHEDULE)
    assert "strength of schedule" in schedule_query.text


def test_league_builder_returns_all_four_kinds_in_order() -> None:
    """Verify league-wide queries cover the same four kinds, in order."""
    builder = LeagueInsightQueryBuilder()
    queries = builder.build_queries(2026)

    assert [query.kind for query in queries] == [
        TeamQueryKind.OUTLOOK,
        TeamQueryKind.OFFENSE,
        TeamQueryKind.DEFENSE,
        TeamQueryKind.SCHEDULE,
    ]


def test_league_queries_mention_all_32_teams_and_draft_year() -> None:
    """Verify league-wide queries are broad (no single team) and include the draft year."""
    builder = LeagueInsightQueryBuilder()
    queries = builder.build_queries(2027)

    for query in queries:
        assert "all 32 teams" in query.text
        assert "2027" in query.text
