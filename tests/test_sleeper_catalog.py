"""Tests for building the Sleeper-anchored draft catalog."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.sleeper_catalog import SleeperCatalogBuilder


def _sleeper_players_df() -> pd.DataFrame:
    """Return a small, unfiltered Sleeper player directory fixture."""
    return pd.DataFrame(
        [
            {
                "sleeper_id": "1",
                "full_name": "Josh Allen",
                "position": "QB",
                "team": "BUF",
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": "QB",
                "years_exp": 7,
            },
            {
                "sleeper_id": "2",
                "full_name": "Free Agent Guy",
                "position": "WR",
                "team": None,
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": None,
                "years_exp": 3,
            },
            {
                "sleeper_id": "3",
                "full_name": "Some Kicker",
                "position": "K",
                "team": "KC",
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": "K",
                "years_exp": 2,
            },
        ]
    )


def test_build_base_catalog_keeps_rostered_and_active_free_agent_skill_players() -> None:
    """Verify active free agents at skill positions are retained in the catalog."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    assert list(catalog_df["sleeper_id"]) == ["1", "2"]


def test_build_base_catalog_derives_player_id_from_sleeper_id() -> None:
    """Verify player_id is an int cast of sleeper_id."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    assert int(catalog_df.iloc[0]["player_id"]) == 1


def test_build_base_catalog_renames_sleeper_columns_for_catalog_use() -> None:
    """Verify Sleeper fields are renamed to catalog and sleeper_* column names."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    row = catalog_df.iloc[0]
    assert row["player_display_name"] == "Josh Allen"
    assert row["recent_team"] == "BUF"
    assert row["sleeper_status"] == "Active"
    assert row["sleeper_depth_chart_position"] == "QB"


def test_find_rostered_players_excluded_by_filter_reports_gaps() -> None:
    """Verify a league-rostered player excluded by the base filter is reported."""
    builder = SleeperCatalogBuilder()
    all_players_df = _sleeper_players_df()
    catalog_df = builder.build_base_catalog(all_players_df, positions=["QB", "RB", "WR", "TE"])
    rostered_df = pd.DataFrame([{"roster_id": 1, "sleeper_id": "1"}, {"roster_id": 1, "sleeper_id": "3"}])

    excluded_df = builder.find_rostered_players_excluded_by_filter(all_players_df, rostered_df, catalog_df)

    assert list(excluded_df["sleeper_id"]) == ["3"]


def test_find_rostered_players_excluded_by_filter_empty_when_fully_covered() -> None:
    """Verify no gaps are reported when every rostered player is in the catalog."""
    builder = SleeperCatalogBuilder()
    all_players_df = _sleeper_players_df()
    catalog_df = builder.build_base_catalog(all_players_df, positions=["QB", "RB", "WR", "TE"])
    rostered_df = pd.DataFrame([{"roster_id": 1, "sleeper_id": "1"}])

    excluded_df = builder.find_rostered_players_excluded_by_filter(all_players_df, rostered_df, catalog_df)

    assert excluded_df.empty
