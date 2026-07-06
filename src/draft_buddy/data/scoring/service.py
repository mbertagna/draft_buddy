"""
Scoring service for applying fantasy scoring rules to player statistics.

Receives raw player statistics and applies configured scoring rules
to generate projected points. Also handles legacy stats aggregation,
games-played fractions, and merging roster with legacy data.
"""

from typing import Dict, Optional

import pandas as pd

from draft_buddy.data.name_matching import standardize_name
from draft_buddy.data.scoring.presets import ESPN_FULL_PPR_TRACKABLE

from .engine import ScoringEngine


def _weekly_projections_from_draft_players(df: pd.DataFrame) -> Dict:
    """
    Build week-to-week point projections from a draft players DataFrame.

    Produces a dict mapping player_id to per-week points (weeks 1-18),
    with bye weeks zeroed out.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: player_id, position, total_pts, bye_week.

    Returns
    -------
    dict
        Mapping player_id -> {'position': str, 1: float, 2: float, ... 18: float}.
    """
    weekly_projections = {}
    for _, row in df.iterrows():
        player_id = row["player_id"]
        avg_pts = row["total_pts"] if pd.notna(row["total_pts"]) else 0
        bye = row["bye_week"]
        weekly_projections[player_id] = {"position": row["position"]}
        for week in range(1, 19):
            weekly_projections[player_id][week] = 0 if week == bye else avg_pts
    return weekly_projections


class ScoringService:
    """
    Applies fantasy scoring rules to player statistics.
    """

    def __init__(self, scoring_rules: Optional[Dict[str, float]] = None):
        """
        Parameters
        ----------
        scoring_rules : dict, optional
            Fantasy scoring rules (stat name -> points per unit).
        """
        self._scoring_rules = scoring_rules or self._default_rules()

    def _default_rules(self) -> Dict[str, float]:
        """Return fallback scoring rules when no league profile is provided."""
        return dict(ESPN_FULL_PPR_TRACKABLE)

    def apply_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fantasy scoring to a stats DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw player statistics (nflverse format).

        Returns
        -------
        pd.DataFrame
            DataFrame with 'total_pts' column added/updated.
        """
        if df is None or df.empty:
            out = df.copy() if df is not None else pd.DataFrame()
            if "total_pts" not in out.columns:
                out["total_pts"] = pd.Series(dtype=float)
            return out

        prepared = ScoringEngine.prepare_offense_kicking_features(df)
        scored = ScoringEngine.apply_scoring(prepared, self._scoring_rules)
        if "total_pts" not in scored.columns:
            scored["total_pts"] = 0.0
        else:
            scored["total_pts"] = pd.to_numeric(scored["total_pts"], errors="coerce").fillna(0.0)
        return scored

    def calculate_games_played_frac(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the fraction of possible games a player has played in their career.

        Parameters
        ----------
        historical_df : pd.DataFrame
            DataFrame with historical week-to-week stats (player_id, season, recent_team, week).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns player_id, games_played_frac.
        """
        if historical_df.empty:
            return pd.DataFrame(columns=["player_id", "games_played_frac"])

        team_games_per_season = (
            historical_df.groupby(["season", "recent_team"])["week"]
            .nunique()
            .reset_index()
        )
        team_games_per_season.rename(columns={"week": "num_team_games"}, inplace=True)

        player_seasons = historical_df[["player_id", "season", "recent_team"]].drop_duplicates()
        player_team_games = player_seasons.merge(
            team_games_per_season, on=["season", "recent_team"], how="left"
        )
        total_team_games = player_team_games.groupby("player_id")["num_team_games"].sum()
        total_player_games = historical_df.groupby("player_id").size()
        games_played_frac = (total_player_games / total_team_games).reset_index(
            name="games_played_frac"
        )
        return games_played_frac

    def aggregate_legacy_stats(
        self,
        scored_historical_df: pd.DataFrame,
        measure_of_center: str = "median",
    ) -> pd.DataFrame:
        """
        Aggregate scored historical stats into legacy stats per player.

        Computes total_pts (median or mean), games_played_frac, and latest metadata.

        Parameters
        ----------
        scored_historical_df : pd.DataFrame
            Scored historical stats with total_pts, player_id, season, etc.
        measure_of_center : str, optional
            'median' or 'mean' for total_pts aggregation.

        Returns
        -------
        pd.DataFrame
            Legacy stats with player_id, total_pts, games_played_frac,
            player_display_name, position, recent_team.
        """
        agg_func = "median" if measure_of_center == "median" else "mean"
        legacy_stats_df = scored_historical_df.groupby("player_id").agg(
            total_pts=("total_pts", agg_func)
        ).reset_index()

        fraction_df = self.calculate_games_played_frac(scored_historical_df)
        legacy_stats_df = legacy_stats_df.merge(
            fraction_df, on="player_id", how="left"
        )

        try:
            legacy_meta = (
                scored_historical_df.sort_values("season")
                .groupby("player_id")
                .agg(
                    player_display_name=("player_display_name", "last"),
                    position=("position", "last"),
                    recent_team=("recent_team", "last"),
                )
                .reset_index()
            )
            legacy_stats_df = legacy_stats_df.merge(
                legacy_meta, on="player_id", how="left"
            )
        except Exception:
            pass

        return legacy_stats_df

    def merge_roster_with_legacy(
        self,
        draft_pool_df: pd.DataFrame,
        legacy_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge roster (draft pool) with legacy stats by name+position.

        Assigns rookie IDs, tracks is_rookie_original, and returns draft_players_df.

        Parameters
        ----------
        draft_pool_df : pd.DataFrame
            Roster with player_display_name, position, player_id.
        legacy_stats_df : pd.DataFrame
            Legacy stats with player_id, total_pts, games_played_frac, metadata.

        Returns
        -------
        pd.DataFrame
            Merged draft players with is_rookie_original column.
        """
        merge_keys_left = ["player_display_name", "position"]
        merge_keys_right = (
            ["player_display_name", "position"]
            if all(
                k in legacy_stats_df.columns
                for k in ["player_display_name", "position"]
            )
            else ["player_id"]
        )

        if merge_keys_right == ["player_id"] and "player_id" not in draft_pool_df.columns:
            draft_players_df = draft_pool_df.copy()
            draft_players_df["player_id"] = pd.NA
            draft_players_df["total_pts"] = pd.NA
            draft_players_df["games_played_frac"] = pd.NA
        else:
            legacy_cols = ["player_id", "total_pts", "games_played_frac"]
            if set(["player_display_name", "position"]).issubset(legacy_stats_df.columns):
                legacy_cols.extend(["player_display_name", "position"])
            left_keys = (
                merge_keys_left if merge_keys_right == ["player_display_name", "position"] else ["player_id"]
            )

            draft_players_df = draft_pool_df.merge(
                legacy_stats_df[legacy_cols],
                left_on=left_keys,
                right_on=merge_keys_right,
                how="left",
                suffixes=("_roster", "_legacy"),
            )
            if "player_id_legacy" in draft_players_df.columns:
                draft_players_df["player_id"] = pd.to_numeric(
                    draft_players_df["player_id_legacy"], errors="coerce"
                ).astype("Int64")
            elif "player_id" in draft_players_df.columns:
                draft_players_df["player_id"] = pd.to_numeric(
                    draft_players_df["player_id"], errors="coerce"
                ).astype("Int64")
            for col in ["player_id_roster", "player_id_legacy"]:
                if col in draft_players_df.columns:
                    draft_players_df.drop(columns=[col], inplace=True)

        draft_players_df["is_rookie_original"] = draft_players_df["total_pts"].isna()

        if "player_id" not in draft_players_df.columns:
            draft_players_df["player_id"] = pd.Series(
                pd.NA, index=draft_players_df.index, dtype="Int64"
            )

        rookie_mask = draft_players_df["is_rookie_original"]
        if rookie_mask.any():
            try:
                max_legacy_id = int(
                    pd.to_numeric(legacy_stats_df["player_id"], errors="coerce").max()
                )
            except Exception:
                max_legacy_id = None
            start_id = (
                (max_legacy_id + 1)
                if max_legacy_id is not None and not pd.isna(max_legacy_id)
                else 1
            )
            num_new = int(rookie_mask.sum())
            if num_new > 0:
                new_ids = pd.Series(
                    range(start_id, start_id + num_new),
                    index=draft_players_df[rookie_mask].index,
                    dtype="Int64",
                )
                draft_players_df.loc[rookie_mask, "player_id"] = new_ids

        if "player_id" in draft_players_df.columns:
            draft_players_df = draft_players_df.drop_duplicates(
                subset=["player_id"], keep="first"
            )

        return draft_players_df

    def attach_legacy_stats(
        self,
        draft_pool_df: pd.DataFrame,
        legacy_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach legacy total_pts/games_played_frac by standardized name+position.

        The join uses ``standardize_name`` (not a raw string match) because
        nflverse's own roster and stats exports are occasionally
        inconsistent about suffixes (e.g. a roster listing "Chris Godwin
        Jr." against stats recorded under "Chris Godwin"). Unlike
        ``merge_roster_with_legacy``, this never reassigns ``player_id``:
        every row's id is assumed to already be final and correct (e.g. a
        Sleeper-derived id). Rows without a legacy stats match simply keep
        their existing id, with ``total_pts`` left as NaN and
        ``is_rookie_original`` set to True for the caller's rookie-projection
        step.

        Parameters
        ----------
        draft_pool_df : pd.DataFrame
            Catalog with a stable ``player_id`` plus ``player_display_name``
            and ``position`` columns used purely as the join key.
        legacy_stats_df : pd.DataFrame
            Legacy stats with ``player_display_name``, ``position``,
            ``total_pts``, and ``games_played_frac`` (see
            ``aggregate_legacy_stats``).

        Returns
        -------
        pd.DataFrame
            ``draft_pool_df`` with ``total_pts``, ``games_played_frac``, and
            ``is_rookie_original`` columns added; ``player_id`` untouched.
        """
        join_keys = ["std_name", "position"]
        legacy_cols = [
            column for column in ["player_display_name", "position", "total_pts", "games_played_frac"]
            if column in legacy_stats_df.columns
        ]

        left_df = draft_pool_df.copy()
        left_df["std_name"] = left_df["player_display_name"].apply(standardize_name)

        right_df = legacy_stats_df[legacy_cols].copy()
        right_df["std_name"] = right_df["player_display_name"].apply(standardize_name)
        right_df = right_df.drop(columns=["player_display_name"]).drop_duplicates(subset=join_keys, keep="first")

        merged_df = left_df.merge(right_df, on=join_keys, how="left").drop(columns=["std_name"])
        for optional_column in ("total_pts", "games_played_frac"):
            if optional_column not in merged_df.columns:
                merged_df[optional_column] = pd.NA

        merged_df["is_rookie_original"] = merged_df["total_pts"].isna()
        return merged_df.drop_duplicates(subset=["player_id"], keep="first")

    def attach_legacy_stats_by_player_id(
        self,
        catalog_df: pd.DataFrame,
        legacy_stats_df: pd.DataFrame,
        *,
        nflverse_player_id_col: str = "nflverse_player_id",
    ) -> pd.DataFrame:
        """
        Attach legacy stats to a Sleeper catalog by nflverse player id.

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Sleeper catalog with stable ``player_id`` and
            ``nflverse_player_id`` join column.
        legacy_stats_df : pd.DataFrame
            Aggregated legacy stats keyed by nflverse ``player_id``.
        nflverse_player_id_col : str, optional
            Column on ``catalog_df`` holding the nflverse stats id.

        Returns
        -------
        pd.DataFrame
            Catalog with ``total_pts``, ``games_played_frac``, and
            ``is_rookie_original`` added. Sleeper ``player_id`` is preserved.
        """
        legacy_cols = [
            column
            for column in ["player_id", "total_pts", "games_played_frac"]
            if column in legacy_stats_df.columns
        ]
        if legacy_cols:
            stats_df = legacy_stats_df[legacy_cols].rename(columns={"player_id": nflverse_player_id_col})
            merged_df = catalog_df.merge(stats_df, on=nflverse_player_id_col, how="left")
        else:
            merged_df = catalog_df.copy()
        for optional_column in ("total_pts", "games_played_frac"):
            if optional_column not in merged_df.columns:
                merged_df[optional_column] = pd.NA
        merged_df["is_rookie_original"] = merged_df["total_pts"].isna()
        return merged_df.drop_duplicates(subset=["player_id"], keep="first")

    def merge_draft_year_with_legacy(
        self,
        draft_year_scored_df: pd.DataFrame,
        legacy_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge draft-year player pool with legacy stats by player_id.

        Parameters
        ----------
        draft_year_scored_df : pd.DataFrame
            Scored draft-year stats.
        legacy_stats_df : pd.DataFrame
            Legacy stats per player.

        Returns
        -------
        pd.DataFrame
            Draft players merged with legacy.
        """
        draft_pool_ids = draft_year_scored_df[
            ["player_id", "player_display_name", "position", "recent_team"]
        ].drop_duplicates(subset=["player_id"])
        return draft_pool_ids.merge(legacy_stats_df, on="player_id", how="left")

    def apply_rookie_metadata(self, draft_players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark rookie games_played_frac as 'R' and drop is_rookie_original.

        Parameters
        ----------
        draft_players_df : pd.DataFrame
            Draft players with is_rookie_original and games_played_frac.

        Returns
        -------
        pd.DataFrame
            DataFrame with rookie metadata applied.
        """
        df = draft_players_df.copy()
        if "games_played_frac" in df.columns and "is_rookie_original" in df.columns:
            mask_rookie = df["is_rookie_original"] == True
            df["games_played_frac"] = df["games_played_frac"].astype(object)
            df.loc[mask_rookie, "games_played_frac"] = "R"
        if "is_rookie_original" in df.columns:
            df.drop(columns=["is_rookie_original"], inplace=True)
        return df

    def finalize_draft_players(self, draft_players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by total_pts and select final output columns.

        Parameters
        ----------
        draft_players_df : pd.DataFrame
            Draft players with all columns.

        Returns
        -------
        pd.DataFrame
            Sorted and trimmed to final columns.
        """
        final_cols = [
            "player_id",
            "player_display_name",
            "position",
            "recent_team",
            "total_pts",
            "games_played_frac",
            "bye_week",
            "sleeper_id",
            "sleeper_status",
            "sleeper_injury_status",
            "sleeper_depth_chart_position",
        ]
        cols_exist = [c for c in final_cols if c in draft_players_df.columns]
        return (
            draft_players_df[cols_exist]
            .sort_values(by="total_pts", ascending=False)
            .reset_index(drop=True)
        )

    def generate_weekly_projections(self, df: pd.DataFrame) -> Dict:
        """
        Build week-to-week point projections from a draft players DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: player_id, position, total_pts, bye_week.

        Returns
        -------
        dict
            Mapping player_id -> {'position': str, 1: float, 2: float, ... 18: float}.
        """
        return _weekly_projections_from_draft_players(df)
