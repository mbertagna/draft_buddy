"""Tests for insight search query templates."""

from __future__ import annotations

from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQueryBuilder, QueryKind


def _player(**overrides) -> InsightPlayerContext:
    """Build a default player context with optional overrides."""
    defaults = {
        "sleeper_id": "4034",
        "name": "Christian McCaffrey",
        "position": "RB",
        "team": "SF",
        "adp": 6.0,
        "projected_points": 11.65,
        "games_played_frac": 0.24,
        "draft_year": 2026,
        "sleeper_injury_status": None,
        "sleeper_depth_chart_position": "RB",
    }
    defaults.update(overrides)
    return InsightPlayerContext(**defaults)


def test_build_queries_includes_outlook_and_role_for_rb() -> None:
    """Verify RB players get outlook and role queries."""
    builder = InsightQueryBuilder()
    queries = builder.build_queries(_player())

    kinds = [query.kind for query in queries]
    assert kinds[:2] == [QueryKind.OUTLOOK, QueryKind.ROLE]
    assert "snap share" in queries[1].text


def test_build_queries_adds_injury_recovery_for_low_gp_veteran() -> None:
    """Verify low GP fraction triggers the injury-recovery query."""
    builder = InsightQueryBuilder()
    queries = builder.build_queries(_player(games_played_frac=0.24))

    assert any(query.kind == QueryKind.INJURY_RECOVERY for query in queries)
    assert "injury recovery" in queries[-1].text


def test_build_queries_skips_injury_recovery_for_rookie() -> None:
    """Verify rookies do not receive the injury-recovery query."""
    builder = InsightQueryBuilder()
    queries = builder.build_queries(
        _player(games_played_frac="R", sleeper_injury_status="Questionable")
    )

    assert all(query.kind != QueryKind.INJURY_RECOVERY for query in queries)


def test_build_queries_adds_injury_recovery_for_injury_status() -> None:
    """Verify Sleeper injury status triggers the injury-recovery query."""
    builder = InsightQueryBuilder()
    queries = builder.build_queries(
        _player(games_played_frac=0.95, sleeper_injury_status="Questionable")
    )

    assert any(query.kind == QueryKind.INJURY_RECOVERY for query in queries)


def test_role_query_templates_vary_by_position() -> None:
    """Verify position-specific role query templates."""
    builder = InsightQueryBuilder()

    wr_query = builder.build_queries(_player(position="WR", games_played_frac=1.0))[1].text
    qb_query = builder.build_queries(_player(position="QB", games_played_frac=1.0))[1].text

    assert "target share" in wr_query
    assert "pass attempts" in qb_query
