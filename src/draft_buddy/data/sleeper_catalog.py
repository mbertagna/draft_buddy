"""
Builds the Sleeper-anchored draft catalog.

Sleeper is treated as the source of truth for player identity, current
team/position, and roster status. This module filters Sleeper's full
player directory down to fantasy-relevant, rostered players and shapes
it into the base catalog that nflverse stats and FantasyPros ADP are
later attached onto.
"""

from typing import Iterable

import pandas as pd

CATALOG_COLUMN_RENAMES = {
    "full_name": "player_display_name",
    "team": "recent_team",
    "status": "sleeper_status",
    "injury_status": "sleeper_injury_status",
    "depth_chart_position": "sleeper_depth_chart_position",
}


class SleeperCatalogBuilder:
    """Builds and validates the Sleeper-sourced draft catalog."""

    def build_base_catalog(
        self, sleeper_players_df: pd.DataFrame, positions: Iterable[str]
    ) -> pd.DataFrame:
        """Filter and shape Sleeper's player directory into a base catalog.

        Parameters
        ----------
        sleeper_players_df : pd.DataFrame
            Full Sleeper player directory, as returned by
            ``SleeperGateway.fetch_all_players``.
        positions : Iterable[str]
            Fantasy-relevant positions to keep (e.g. ``["QB", "RB", "WR", "TE"]``).

        Returns
        -------
        pd.DataFrame
            One row per fantasy-relevant player with ``player_id``
            (int, derived from ``sleeper_id``), ``player_display_name``,
            ``position``, ``recent_team``, ``gsis_id``, ``sleeper_id``,
            ``sleeper_status``, ``sleeper_injury_status``,
            ``sleeper_depth_chart_position``, and ``years_exp``.
        """
        position_list = list(positions)
        is_fantasy_relevant = sleeper_players_df["position"].isin(position_list)
        is_on_team = sleeper_players_df["team"].notna()
        is_active_without_team = (
            sleeper_players_df["status"].eq("Active") & sleeper_players_df["team"].isna()
        )
        is_rosterable = is_fantasy_relevant & (is_on_team | is_active_without_team)
        catalog_df = sleeper_players_df[is_rosterable].rename(columns=CATALOG_COLUMN_RENAMES).copy()
        catalog_df["player_id"] = catalog_df["sleeper_id"].astype(int)
        return catalog_df.reset_index(drop=True)

    def find_rostered_players_excluded_by_filter(
        self,
        all_sleeper_players_df: pd.DataFrame,
        rostered_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return league-rostered Sleeper players excluded by the base filter.

        A safety net confirming ``build_base_catalog``'s position/team filter
        didn't wrongly drop a player who is actually rostered in a real league.

        Parameters
        ----------
        all_sleeper_players_df : pd.DataFrame
            Full, unfiltered Sleeper player directory.
        rostered_df : pd.DataFrame
            Rostered players for one league, as returned by
            ``SleeperGateway.fetch_league_rosters``.
        catalog_df : pd.DataFrame
            The catalog produced by ``build_base_catalog``.

        Returns
        -------
        pd.DataFrame
            Sleeper player rows that are rostered in the league but absent
            from the catalog.
        """
        catalog_ids = set(catalog_df["sleeper_id"])
        rostered_ids = {value for value in rostered_df["sleeper_id"] if pd.notna(value)}
        excluded_ids = rostered_ids - catalog_ids
        return (
            all_sleeper_players_df[all_sleeper_players_df["sleeper_id"].isin(excluded_ids)]
            .reset_index(drop=True)
        )
