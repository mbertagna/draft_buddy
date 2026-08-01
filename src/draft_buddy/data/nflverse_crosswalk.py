"""Build sleeper_id to nflverse GSIS crosswalks from cached roster files."""

from __future__ import annotations

import glob
import os
from typing import Optional

import pandas as pd

from draft_buddy.data.cache_paths import nflverse_cache_dir
from draft_buddy.data.nflverse_ids import normalize_sleeper_id
from draft_buddy.data.nflverse_client import NflverseCsvDownloader


class NflverseCrosswalkBuilder:
    """Builds sleeper_id crosswalk rows for GSIS lookup and draft slot metadata."""

    def __init__(self, downloader: Optional[NflverseCsvDownloader] = None) -> None:
        """Initialize the builder with an optional nflverse downloader.

        Parameters
        ----------
        downloader : NflverseCsvDownloader, optional
            Downloader used to refresh roster cache files when needed.
        """
        self._downloader = downloader

    def build(self, data_root: str, draft_year: int) -> pd.DataFrame:
        """Return sleeper_id crosswalk rows from cached nflverse rosters.

        Ensures at least one roster file is present for ``draft_year`` by
        delegating to the nflverse downloader when the cache is empty or stale.

        Parameters
        ----------
        data_root : str
            Root data directory (e.g. ``./data``).
        draft_year : int
            Draft season used to refresh roster cache when needed.

        Returns
        -------
        pd.DataFrame
            Columns: ``sleeper_id``, ``gsis_id``, ``draft_number``.
        """
        cache_dir = nflverse_cache_dir(data_root)
        os.makedirs(cache_dir, exist_ok=True)
        if self._downloader is not None:
            self._downloader.ensure_roster_cached(draft_year)

        roster_paths = sorted(glob.glob(os.path.join(cache_dir, "roster_*.csv")))
        if not roster_paths:
            return pd.DataFrame(columns=["sleeper_id", "gsis_id", "draft_number"])

        frames = [pd.read_csv(path) for path in roster_paths]
        combined_df = pd.concat(frames, ignore_index=True)
        if "sleeper_id" not in combined_df.columns:
            return pd.DataFrame(columns=["sleeper_id", "gsis_id", "draft_number"])

        combined_df["sleeper_id"] = combined_df["sleeper_id"].apply(normalize_sleeper_id)
        combined_df = combined_df.dropna(subset=["sleeper_id"])
        if combined_df.empty:
            return pd.DataFrame(columns=["sleeper_id", "gsis_id", "draft_number"])

        if "season" in combined_df.columns:
            combined_df = combined_df.sort_values("season", ascending=True)

        keep_columns = ["sleeper_id", "gsis_id", "draft_number"]
        available_columns = [column for column in keep_columns if column in combined_df.columns]
        crosswalk_df = (
            combined_df[available_columns]
            .drop_duplicates(subset=["sleeper_id"], keep="last")
            .reset_index(drop=True)
        )
        if "draft_number" not in crosswalk_df.columns:
            crosswalk_df["draft_number"] = pd.NA
        return crosswalk_df
