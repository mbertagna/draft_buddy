"""Tests that pipeline CLIs default the draft year from the season overlay."""

from __future__ import annotations

import argparse

import fetch_player_insight_search
import generate_position_guide
import synthesize_player_insights


def test_insights_search_defaults_year_from_season() -> None:
    """Verify insight search uses the overlay year when --year is omitted."""
    args = fetch_player_insight_search.parse_args([], default_year=2026)

    assert args.year == 2026


def test_insights_synthesize_defaults_year_from_season() -> None:
    """Verify insight synthesis uses the overlay year when --year is omitted."""
    args = synthesize_player_insights.parse_args([], default_year=2026)

    assert args.year == 2026


def test_position_guide_defaults_year_from_season() -> None:
    """Verify position-guide generation uses the overlay year when --year is omitted."""
    runtime_defaults = argparse.Namespace(num_teams=10, year=2026)
    args = generate_position_guide.parse_args(runtime_defaults, [])

    assert args.year == 2026
    assert args.num_teams == 10
