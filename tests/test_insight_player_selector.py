"""Tests for top-N ADP player selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from draft_buddy.data.insights.player_selector import InsightPlayerSelector


def _write_fixture_csv(path: Path) -> None:
    """Write a small generated-player CSV fixture."""
    dataframe = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Player One",
                "position": "RB",
                "projected_points": 18.0,
                "adp": 2.0,
                "games_played_frac": 0.9,
                "recent_team": "DET",
            },
            {
                "player_id": 2,
                "name": "Player Two",
                "position": "WR",
                "projected_points": 17.0,
                "adp": 1.0,
                "games_played_frac": "R",
                "recent_team": "ATL",
            },
            {
                "player_id": 3,
                "name": "Player Three",
                "position": "QB",
                "projected_points": 22.0,
                "adp": 3.0,
                "games_played_frac": 1.0,
                "recent_team": "BUF",
            },
        ]
    )
    dataframe.to_csv(path, index=False)


def test_select_returns_top_n_sorted_by_adp(tmp_path: Path) -> None:
    """Verify selector returns players ordered by ascending ADP."""
    csv_path = tmp_path / "generated_player_data.csv"
    _write_fixture_csv(csv_path)

    selector = InsightPlayerSelector(str(csv_path), draft_year=2026)
    players = selector.select(top_n=3)

    assert [player.name for player in players] == ["Player Two", "Player One", "Player Three"]
    assert players[0].is_rookie() is True
    assert players[0].sleeper_id == "2"


def test_select_raises_when_csv_missing(tmp_path: Path) -> None:
    """Verify missing CSV raises FileNotFoundError with guidance."""
    selector = InsightPlayerSelector(str(tmp_path / "missing.csv"), draft_year=2026)

    with pytest.raises(FileNotFoundError, match="docker compose run --rm data"):
        selector.select(top_n=1)


def test_select_raises_when_not_enough_rows(tmp_path: Path) -> None:
    """Verify selector errors when top_n exceeds available rows."""
    csv_path = tmp_path / "generated_player_data.csv"
    _write_fixture_csv(csv_path)
    selector = InsightPlayerSelector(str(csv_path), draft_year=2026)

    with pytest.raises(ValueError, match="requested top_n=5"):
        selector.select(top_n=5)


def test_select_honors_start_index_and_max_players(tmp_path: Path) -> None:
    """Verify start_index and max_players slice the ADP list."""
    csv_path = tmp_path / "generated_player_data.csv"
    _write_fixture_csv(csv_path)
    selector = InsightPlayerSelector(str(csv_path), draft_year=2026)

    players = selector.select(top_n=3, start_index=1, max_players=1)

    assert len(players) == 1
    assert players[0].name == "Player One"
