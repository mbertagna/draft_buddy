"""Tests for distinct NFL team selection from the generated player CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from draft_buddy.data.insights.team_selector import InsightTeamSelector


def _write_fixture_csv(path: Path) -> None:
    """Write a small generated-player CSV fixture with duplicate teams."""
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "name": "Player One", "recent_team": "SF"},
            {"player_id": 2, "name": "Player Two", "recent_team": "DET"},
            {"player_id": 3, "name": "Player Three", "recent_team": "SF"},
            {"player_id": 4, "name": "Player Four", "recent_team": "ATL"},
            {"player_id": 5, "name": "Player Five", "recent_team": None},
        ]
    )
    dataframe.to_csv(path, index=False)


def test_select_returns_distinct_teams_sorted_alphabetically(tmp_path: Path) -> None:
    """Verify selector returns distinct, sorted team abbreviations."""
    csv_path = tmp_path / "generated_player_data.csv"
    _write_fixture_csv(csv_path)

    selector = InsightTeamSelector(str(csv_path), draft_year=2026)
    teams = selector.select()

    assert [team.team_abbr for team in teams] == ["ATL", "DET", "SF"]
    assert all(team.draft_year == 2026 for team in teams)


def test_select_raises_when_csv_missing(tmp_path: Path) -> None:
    """Verify missing CSV raises FileNotFoundError with guidance."""
    selector = InsightTeamSelector(str(tmp_path / "missing.csv"), draft_year=2026)

    with pytest.raises(FileNotFoundError, match="docker compose run --rm data"):
        selector.select()


def test_select_honors_start_index_and_max_teams(tmp_path: Path) -> None:
    """Verify start_index and max_teams slice the sorted team list."""
    csv_path = tmp_path / "generated_player_data.csv"
    _write_fixture_csv(csv_path)
    selector = InsightTeamSelector(str(csv_path), draft_year=2026)

    teams = selector.select(start_index=1, max_teams=1)

    assert [team.team_abbr for team in teams] == ["DET"]


def test_select_falls_back_to_team_column(tmp_path: Path) -> None:
    """Verify selector falls back to a 'team' column when 'recent_team' is absent."""
    csv_path = tmp_path / "generated_player_data.csv"
    dataframe = pd.DataFrame(
        [
            {"player_id": 1, "name": "Player One", "team": "BUF"},
            {"player_id": 2, "name": "Player Two", "team": "KC"},
        ]
    )
    dataframe.to_csv(csv_path, index=False)
    selector = InsightTeamSelector(str(csv_path), draft_year=2026)

    teams = selector.select()

    assert [team.team_abbr for team in teams] == ["BUF", "KC"]
