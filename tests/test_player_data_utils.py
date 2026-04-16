"""Tests for simulation data utilities."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.player_data_utils import _convert_to_simulation_format, get_simulation_dfs


def test_convert_to_simulation_format_uses_existing_pts_list() -> None:
    """Verify already-converted weekly projections keep their pts list."""
    converted = _convert_to_simulation_format({1: {"pos": "QB", "pts": [1.0, 2.0]}})

    assert converted == {1: {"pos": "QB", "pts": [1.0, 2.0]}}


def test_convert_to_simulation_format_builds_pts_from_week_keys() -> None:
    """Verify week-number keys are converted into an ordered pts list."""
    converted = _convert_to_simulation_format({1: {"position": "WR", 2: 4.0, "1": 3.0}})

    assert converted == {1: {"pos": "WR", "pts": [3.0, 4.0]}}


def test_get_simulation_dfs_converts_processor_output(monkeypatch) -> None:
    """Verify get_simulation_dfs converts process_draft_data output into simulator format."""

    class FakeProcessor:
        def __init__(self, **kwargs):
            _ = kwargs

        def process_draft_data(self, **kwargs):
            _ = kwargs
            return pd.DataFrame([{"player_id": 1}]), {1: {"position": "QB", 1: 5.0, 2: 0.0}}

    monkeypatch.setattr("draft_buddy.data.player_data_utils.FantasyDataProcessor", FakeProcessor)
    draft_players_df, weekly_projections = get_simulation_dfs(season=2025, ps_start_year=2020)

    assert int(draft_players_df.iloc[0]["player_id"]) == 1 and weekly_projections == {1: {"pos": "QB", "pts": [5.0, 0.0]}}
