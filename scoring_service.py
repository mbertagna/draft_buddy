"""
Scoring service for applying fantasy scoring rules to player statistics.

Receives raw player statistics and applies configured scoring rules
to generate projected points. Also handles legacy stats aggregation,
games-played fractions, and merging roster with legacy data.
"""

from typing import Dict, Optional

import pandas as pd

from utils.scoring_utils import ScoringEngine


def generate_weekly_projections(df: pd.DataFrame) -> Dict:
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
        """Returns default PPR scoring rules."""
        return {
            "passing_yards": 0.04,
            "passing_tds": 4,
            "interceptions": -2,
            "passing_2pt_conversions": 2,
            "passing_yards_300_399_game": 2,
            "passing_yards_400_plus_game": 6,
            "receiving_yards": 0.1,
            "receptions": 1,
            "receiving_tds": 6,
            "receiving_2pt_conversions": 2,
            "receiving_yards_100_199_game": 3,
            "receiving_yards_200_plus_game": 6,
            "rushing_yards": 0.1,
            "rushing_tds": 6,
            "rushing_2pt_conversions": 2,
            "rushing_yards_100_199_game": 3,
            "rushing_yards_200_plus_game": 6,
            "pat_made": 1,
            "fg_missed": -1,
            "fg_made_0_39": 3,
            "fg_made_40_49": 4,
            "fg_made_50_59": 5,
            "fg_made_60_": 6,
            "total_fumbles_lost": -2,
        }

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

    def generate_weekly_projections(self, df: pd.DataFrame) -> Dict:
        """
        Build week-to-week point projections from a draft players DataFrame.

        Delegates to the module-level generate_weekly_projections helper.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: player_id, position, total_pts, bye_week.

        Returns
        -------
        dict
            Mapping player_id -> {'position': str, 1: float, 2: float, ... 18: float}.
        """
        return generate_weekly_projections(df)

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

            draft_players_df = draft_pool_df.merge(
                legacy_stats_df[legacy_cols],
                left_on=merge_keys_left,
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
        ]
        cols_exist = [c for c in final_cols if c in draft_players_df.columns]
        return (
            draft_players_df[cols_exist]
            .sort_values(by="total_pts", ascending=False)
            .reset_index(drop=True)
        )
