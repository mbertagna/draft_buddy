"""Tests for the canonical player-loading API."""

from __future__ import annotations

import math

import pandas as pd

from draft_buddy.data import load_player_catalog


def test_load_player_catalog_returns_players_sorted_by_adp(config, player_dataframe) -> None:
    """Verify player loading returns a PlayerCatalog sorted by ADP."""
    player_dataframe.to_csv(config.paths.PLAYER_DATA_CSV, index=False)
    catalog = load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)

    assert [player.player_id for player in catalog][:4] == [1, 2, 3, 4]


def test_load_player_catalog_generates_mock_adp_when_missing(config) -> None:
    """Verify missing ADP values are synthesized through the canonical loader."""
    pd.DataFrame(
        [
            {"player_id": 1, "name": "A", "position": "QB", "projected_points": 100.0},
            {"player_id": 2, "name": "B", "position": "RB", "projected_points": 200.0},
        ]
    ).to_csv(config.paths.PLAYER_DATA_CSV, index=False)
    catalog = load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)

    assert all(math.isfinite(player.adp) for player in catalog)
