"""
Rookie projection logic for fantasy football.

Encapsulates the statistical logic for estimating rookie performance
based on draft slot or ADP interpolation.
"""

from typing import Optional

import numpy as np
import pandas as pd


class RookieProjector:
    """
    Estimates fantasy points for rookies based on draft number or ADP.
    """

    def __init__(
        self,
        scale_min: int = 5,
        scale_max: int = 80,
        udfa_percentile: float = 75,
        adp_matcher=None,
    ):
        """
        Parameters
        ----------
        scale_min : int
            Minimum percentile for drafted players.
        scale_max : int
            Maximum percentile for drafted players.
        udfa_percentile : float
            Percentile for undrafted free agents.
        adp_matcher : AdpMatcher, optional
            Injected ADP matcher for adp/hybrid projection methods.
        """
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._udfa_percentile = udfa_percentile
        self._adp_matcher = adp_matcher

    def estimate_by_draft_slot(
        self, rookies_df: pd.DataFrame, veterans_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimates rookie points using draft number scaling.

        Parameters
        ----------
        rookies_df : pd.DataFrame
            Rookies with draft_number column.
        veterans_df : pd.DataFrame
            Veterans with total_pts for percentile lookup.

        Returns
        -------
        pd.DataFrame
            Rookies with estimated total_pts.
        """
        projected_list = []
        for position in rookies_df["position"].unique():
            pos_rookies = rookies_df[rookies_df["position"] == position].copy()
            pos_veterans = veterans_df[veterans_df["position"] == position]

            if pos_veterans.empty:
                pos_rookies["total_pts"] = 0
                projected_list.append(pos_rookies)
                continue

            vet_pts = pos_veterans["total_pts"].values
            draft_numbers = pos_rookies["draft_number"].dropna()
            min_draft = draft_numbers.min() if not draft_numbers.empty else 0
            max_draft = draft_numbers.max() if not draft_numbers.empty else 0

            def scale_to_percentile(x):
                if pd.isna(x):
                    return self._udfa_percentile
                if max_draft == min_draft:
                    return 100 - self._scale_min
                scaled = (
                    (self._scale_max - self._scale_min)
                    * (x - min_draft)
                    / (max_draft - min_draft)
                ) + self._scale_min
                return 100 - scaled

            pos_rookies["draft_percentile"] = pos_rookies["draft_number"].apply(scale_to_percentile)
            pos_rookies["total_pts"] = pos_rookies["draft_percentile"].apply(
                lambda p: np.percentile(vet_pts, p)
            )
            projected_list.append(pos_rookies)

        return pd.concat(projected_list, ignore_index=True) if projected_list else pd.DataFrame()

    def _project_rookies_with_adp(
        self,
        draft_players_df: pd.DataFrame,
        adp_filepath: str,
        match_threshold: int = 85,
        adp_col_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Project rookie fantasy points using ADP-based interpolation.

        Parameters
        ----------
        draft_players_df : pd.DataFrame
            Draft players with rookies (total_pts NaN).
        adp_filepath : str
            Path to ADP CSV file.
        match_threshold : int, optional
            Fuzzy match threshold.
        adp_col_map : dict, optional
            ADP column mapping.

        Returns
        -------
        pd.DataFrame
            Draft players with rookie points filled via ADP interpolation.
        """
        if not adp_filepath or not self._adp_matcher:
            return draft_players_df

        draft_players_df = draft_players_df.copy()
        draft_players_df["is_rookie"] = draft_players_df["total_pts"].isna()

        merged_df, _, _ = self._adp_matcher.merge_adp_data(
            computed_df=draft_players_df,
            adp_filepath=adp_filepath,
            match_threshold=match_threshold,
            adp_col_map=adp_col_map,
        )

        if merged_df.empty:
            return draft_players_df

        adp_val_col = (
            "AVG" if "AVG" in merged_df.columns else ("Rank" if "Rank" in merged_df.columns else None)
        )
        if adp_val_col is None:
            return draft_players_df

        pos_col = (
            "Pos"
            if "Pos" in merged_df.columns
            else ("position" if "position" in merged_df.columns else None)
        )
        if pos_col is None:
            return draft_players_df

        veteran_mask = (merged_df["is_rookie"] == False) & merged_df["total_pts"].notna()
        pos_to_vet_median = (
            merged_df[veteran_mask].groupby(pos_col)["total_pts"].median().to_dict()
        )

        predicted_points_by_player_id = {}

        for position_value, group in merged_df.groupby(pos_col):
            group_sorted = group.sort_values(by=adp_val_col, ascending=True).reset_index(
                drop=True
            )
            vet_indices = group_sorted[
                (group_sorted["is_rookie"] == False) & group_sorted["total_pts"].notna()
            ].index.tolist()
            if not vet_indices:
                continue

            vet_idx_set = set(vet_indices)

            for idx, row in group_sorted.iterrows():
                if not bool(row.get("is_rookie", False)):
                    continue

                adp_val = row.get(adp_val_col)
                if pd.isna(adp_val):
                    continue

                lower_idx = None
                upper_idx = None

                for i in range(idx - 1, -1, -1):
                    if i in vet_idx_set:
                        lower_idx = i
                        break

                for j in range(idx + 1, len(group_sorted)):
                    if j in vet_idx_set:
                        upper_idx = j
                        break

                predicted = None
                if (lower_idx is not None) and (upper_idx is not None):
                    low_row = group_sorted.loc[lower_idx]
                    high_row = group_sorted.loc[upper_idx]
                    x0, x1 = low_row[adp_val_col], high_row[adp_val_col]
                    y0, y1 = low_row["total_pts"], high_row["total_pts"]
                    if (
                        pd.notna(x0)
                        and pd.notna(x1)
                        and (x1 > x0)
                        and pd.notna(y0)
                        and pd.notna(y1)
                    ):
                        t = (adp_val - x0) / (x1 - x0)
                        predicted = y0 + t * (y1 - y0)
                elif lower_idx is not None:
                    predicted = group_sorted.loc[lower_idx]["total_pts"]
                elif upper_idx is not None:
                    predicted = group_sorted.loc[upper_idx]["total_pts"]

                if predicted is None or pd.isna(predicted):
                    predicted = pos_to_vet_median.get(position_value, np.nan)

                pid = row.get("player_id")
                if pd.notna(predicted) and pd.notna(pid):
                    predicted_points_by_player_id[pid] = float(predicted)

        if not predicted_points_by_player_id:
            return draft_players_df

        mask = (
            draft_players_df["player_id"].isin(predicted_points_by_player_id.keys())
            & draft_players_df["is_rookie"]
        )
        draft_players_df.loc[mask, "total_pts"] = draft_players_df.loc[
            mask, "player_id"
        ].map(predicted_points_by_player_id)

        draft_players_df.drop(columns=["is_rookie"], inplace=True)
        return draft_players_df

    def project_rookies(
        self,
        draft_players_df: pd.DataFrame,
        method: str,
        adp_filepath: Optional[str] = None,
        match_threshold: int = 85,
        adp_col_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Project rookie points using the specified method (draft, adp, or hybrid).

        Parameters
        ----------
        draft_players_df : pd.DataFrame
            Draft players with veterans and rookies (is_rookie_original).
        method : str
            'draft', 'adp', or 'hybrid'.
        adp_filepath : str, optional
            Path to ADP CSV (required for adp/hybrid).
        match_threshold : int, optional
            Fuzzy match threshold for ADP.
        adp_col_map : dict, optional
            ADP column mapping.

        Returns
        -------
        pd.DataFrame
            Draft players with rookie points projected.
        """
        veterans_df = draft_players_df[
            draft_players_df["is_rookie_original"] == False
        ].copy()
        rookies_df = draft_players_df[
            draft_players_df["is_rookie_original"] == True
        ].copy()

        if rookies_df.empty:
            return draft_players_df

        if method == "adp":
            draft_players_df = self._project_rookies_with_adp(
                draft_players_df=draft_players_df,
                adp_filepath=adp_filepath or "",
                match_threshold=match_threshold,
                adp_col_map=adp_col_map,
            )
            remaining = draft_players_df[draft_players_df["total_pts"].isna()].copy()
            if not remaining.empty:
                fallback = self.estimate_by_draft_slot(remaining, veterans_df)
                draft_players_df = pd.concat(
                    [
                        draft_players_df[draft_players_df["total_pts"].notna()],
                        fallback,
                    ],
                    ignore_index=True,
                )
        elif method == "hybrid":
            adp_version = self._project_rookies_with_adp(
                draft_players_df=draft_players_df.copy(),
                adp_filepath=adp_filepath or "",
                match_threshold=match_threshold,
                adp_col_map=adp_col_map,
            )
            draft_based = pd.concat(
                [
                    veterans_df,
                    self.estimate_by_draft_slot(rookies_df, veterans_df),
                ],
                ignore_index=True,
            )
            merged_versions = draft_based[["player_id", "total_pts"]].merge(
                adp_version[["player_id", "total_pts"]],
                on="player_id",
                how="outer",
                suffixes=("_draft", "_adp"),
            )
            merged_versions["total_pts_final"] = merged_versions[
                ["total_pts_draft", "total_pts_adp"]
            ].mean(axis=1, skipna=True)
            draft_players_df = draft_players_df.merge(
                merged_versions[["player_id", "total_pts_final"]],
                on="player_id",
                how="left",
            )
            draft_players_df["total_pts"] = draft_players_df["total_pts"].combine_first(
                draft_players_df["total_pts_final"]
            )
            draft_players_df.drop(columns=["total_pts_final"], inplace=True)
            remaining = draft_players_df[draft_players_df["total_pts"].isna()].copy()
            if not remaining.empty:
                fallback = self.estimate_by_draft_slot(remaining, veterans_df)
                draft_players_df = pd.concat(
                    [
                        draft_players_df[draft_players_df["total_pts"].notna()],
                        fallback,
                    ],
                    ignore_index=True,
                )
        else:
            projected = self.estimate_by_draft_slot(rookies_df, veterans_df)
            draft_players_df = pd.concat(
                [veterans_df, projected], ignore_index=True
            )

        return draft_players_df
