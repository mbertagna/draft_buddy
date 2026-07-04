"""Tests for the canonical player-loading API."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

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


def test_load_player_catalog_creates_dummy_csv_when_file_is_missing(config, tmp_path: Path) -> None:
    """Verify loading a missing file creates and loads the fallback CSV."""
    missing_path = tmp_path / "nested" / "players.csv"

    catalog = load_player_catalog(str(missing_path), config.draft.MOCK_ADP_CONFIG)

    assert missing_path.exists() and len(catalog) == 8


def test_load_player_catalog_rejects_missing_required_columns(config) -> None:
    """Verify missing required player columns raise a descriptive error."""
    pd.DataFrame([{"player_id": 1, "name": "A", "position": "QB"}]).to_csv(
        config.paths.PLAYER_DATA_CSV, index=False
    )

    with pytest.raises(ValueError, match="Missing required column 'projected_points'"):
        load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)


def test_load_player_catalog_preserves_rookie_fraction_and_prefers_recent_team(config) -> None:
    """Verify loader preserves rookie markers and uses recent_team over team."""
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "qb",
                "projected_points": 100.0,
                "games_played_frac": "R",
                "adp": 1.0,
                "recent_team": "BUF",
                "team": "KC",
            }
        ]
    ).to_csv(config.paths.PLAYER_DATA_CSV, index=False)

    catalog = load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)

    assert catalog.require(1).games_played_frac == "R" and catalog.require(1).team == "BUF"


def test_load_player_catalog_reads_sleeper_columns_when_present(config) -> None:
    """Verify Sleeper-derived columns populate the corresponding Player fields."""
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "QB",
                "projected_points": 100.0,
                "adp": 1.0,
                "sleeper_id": "4984",
                "sleeper_status": "Active",
                "sleeper_injury_status": "Questionable",
                "sleeper_depth_chart_position": "QB",
            }
        ]
    ).to_csv(config.paths.PLAYER_DATA_CSV, index=False)

    catalog = load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)

    player = catalog.require(1)
    assert player.sleeper_id == "4984" and player.sleeper_injury_status == "Questionable"


def test_load_player_catalog_defaults_sleeper_fields_when_columns_absent(config, player_dataframe) -> None:
    """Verify CSVs without Sleeper columns still load with null sleeper fields."""
    player_dataframe.to_csv(config.paths.PLAYER_DATA_CSV, index=False)

    catalog = load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)

    assert catalog.require(1).sleeper_id is None


def test_load_player_catalog_raises_when_mock_adp_generation_is_disabled(config) -> None:
    """Verify missing ADP data fails when synthetic ADP generation is disabled."""
    pd.DataFrame(
        [{"player_id": 1, "name": "A", "position": "QB", "projected_points": 100.0}]
    ).to_csv(config.paths.PLAYER_DATA_CSV, index=False)
    adp_config = dict(config.draft.MOCK_ADP_CONFIG)
    adp_config["enabled"] = False

    with pytest.raises(ValueError, match="Mock ADP generation is disabled"):
        load_player_catalog(config.paths.PLAYER_DATA_CSV, adp_config)
