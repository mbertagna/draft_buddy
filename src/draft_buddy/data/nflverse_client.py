"""
Nflverse-backed client for fetching fantasy football player data.

Defines the abstract ``DataDownloader`` contract and a concrete implementation
that downloads nflverse CSV releases into a local cache.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import requests

from draft_buddy.data.nflverse_ids import normalize_gsis_id

DEFAULT_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


class DataDownloader(ABC):
    """
    Abstract interface for fetching nflverse stats and optional roster cache files.
    """

    @abstractmethod
    def fetch_legacy_stats(
        self, draft_year: int, positions: list, start_year: int, end_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical and draft-year weekly stats for scoring.

        Parameters
        ----------
        draft_year : int
            The year of the draft.
        positions : list
            List of positions to include (e.g., ['QB', 'RB', 'WR', 'TE']).
        start_year : int
            First year of historical data.
        end_year : int
            Last year of historical data (typically draft_year).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (legacy_stats_df, draft_year_stats_df).
        """
        pass

    @abstractmethod
    def fetch_draft_year_roster(self, draft_year: int, positions: list) -> pd.DataFrame:
        """
        Fetch the nflverse roster file for non-rookie processing paths.

        Parameters
        ----------
        draft_year : int
            The draft year roster to download.
        positions : list
            Positions to retain.

        Returns
        -------
        pd.DataFrame
            Roster rows with ``player_display_name`` and ``recent_team``.
        """
        pass


class NflverseCsvDownloader(DataDownloader):
    """
    Downloads and loads player data from nflverse GitHub releases.
    """

    def __init__(self, cache_dir: str = "./data", cache_max_age_seconds: int = DEFAULT_CACHE_MAX_AGE_SECONDS):
        """
        Parameters
        ----------
        cache_dir : str
            Directory for caching downloaded files.
        cache_max_age_seconds : int, optional
            Maximum cache age, in seconds, before a cached file is
            re-downloaded. nflverse continuously updates these releases in
            place (new stats, roster moves), so a cached copy left in place
            indefinitely silently goes stale.
        """
        self._cache_dir = cache_dir
        self._cache_max_age_seconds = cache_max_age_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def download_file(self, file_name: str, url: str) -> pd.DataFrame:
        """
        Downloads file if not cached or the cache is stale, returns DataFrame.

        Parameters
        ----------
        file_name : str
            Local filename in cache.
        url : str
            URL to download from.

        Returns
        -------
        pd.DataFrame
            Loaded CSV data.
        """
        file_path = os.path.join(self._cache_dir, file_name)
        if not self._is_cache_fresh(file_path):
            self._download_from_url(url, file_path)
        return pd.read_csv(file_path)

    def _is_cache_fresh(self, file_path: str) -> bool:
        """Return True when a cached file exists, is non-empty, and isn't stale."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        cache_age_seconds = time.time() - os.path.getmtime(file_path)
        return cache_age_seconds <= self._cache_max_age_seconds

    def _download_from_url(self, url: str, file_path: str, chunk_size: int = 8192) -> None:
        """Downloads a file from URL to local path."""
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                total_size = int(r.headers.get("content-length", 0))
                print(f"Downloading {os.path.basename(file_path)} ({total_size/1e6:.2f} MB)...")
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        print(f"Successfully downloaded to {file_path}")

    def _normalize_player_id_column(self, df: pd.DataFrame, id_column: str = "player_id") -> None:
        """
        Normalize player_id to integer form (strip non-digits, convert to Int64).

        Modifies the DataFrame in place.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the id column.
        id_column : str, optional
            Column name for player ID. Default 'player_id'.
        """
        if id_column not in df.columns:
            return
        df[id_column] = df[id_column].apply(normalize_gsis_id).astype("Int64")

    def ensure_roster_cached(self, draft_year: int) -> None:
        """Download the draft-year roster CSV when cache is missing or stale.

        Parameters
        ----------
        draft_year : int
            Season whose roster file should be present in the cache.
        """
        roster_url = (
            f"https://github.com/nflverse/nflverse-data/releases/download/"
            f"rosters/roster_{draft_year}.csv"
        )
        self.download_file(f"roster_{draft_year}.csv", roster_url)

    def fetch_legacy_stats(
        self, draft_year: int, positions: list, start_year: int, end_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch nflverse weekly stats split into legacy and draft-year frames."""
        stats_url = (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            "player_stats/player_stats.csv"
        )
        kicking_url = (
            "https://github.com/nflverse/nflverse-data/releases/download/"
            "player_stats/player_stats_kicking.csv"
        )

        ps_df = self.download_file("player_stats.csv", stats_url)
        psk_df = self.download_file("player_stats_kicking.csv", kicking_url)

        merge_cols = list(set(psk_df.columns).intersection(set(ps_df.columns)))
        merged_df = ps_df.merge(psk_df, how="outer", on=merge_cols)
        merged_df = merged_df[merged_df["season"].between(start_year, end_year)]
        merged_df = merged_df[merged_df["position"].isin(positions)]

        legacy_stats_df = merged_df[merged_df["season"] < draft_year].copy()
        draft_year_stats_df = merged_df[merged_df["season"] == draft_year].copy()

        self._normalize_player_id_column(legacy_stats_df)
        self._normalize_player_id_column(draft_year_stats_df)

        return legacy_stats_df, draft_year_stats_df

    def fetch_draft_year_roster(self, draft_year: int, positions: list) -> pd.DataFrame:
        """Fetch nflverse roster rows for the legacy non-rookie processing path."""
        self.ensure_roster_cached(draft_year)
        roster_df = pd.read_csv(os.path.join(self._cache_dir, f"roster_{draft_year}.csv"))
        roster_df = roster_df.rename(columns={"team": "recent_team"})
        roster_df = roster_df[roster_df["position"].isin(positions)].copy()
        if "player_id" not in roster_df.columns:
            roster_df["player_id"] = roster_df.index.astype(int)
        roster_df["player_display_name"] = roster_df["full_name"]
        self._normalize_player_id_column(roster_df)
        return roster_df
