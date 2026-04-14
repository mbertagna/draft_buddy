"""
Nflverse-backed client for fetching fantasy football player data.

Defines the abstract ``DataDownloader`` contract and a concrete implementation
that downloads nflverse CSV releases into a local cache.
"""

import os
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import requests


class DataDownloader(ABC):
    """
    Abstract interface for fetching player pool data.
    """

    @abstractmethod
    def fetch_player_pool(
        self, draft_year: int, positions: list, start_year: int, end_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetches the player pool and historical stats for the draft year.

        Parameters
        ----------
        draft_year : int
            The year of the draft.
        positions : list
            List of positions to include (e.g., ['QB', 'RB', 'WR', 'TE']).
        start_year : int
            First year of historical data.
        end_year : int
            Last year of historical data (typically draft_year to include draft-year stats).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (draft_pool_df, legacy_stats_df, draft_year_stats_df) - roster, historical stats,
            and draft-year stats (empty if end_year < draft_year).
        """
        pass


class NflverseCsvDownloader(DataDownloader):
    """
    Downloads and loads player data from nflverse GitHub releases.
    """

    def __init__(self, cache_dir: str = "./data"):
        """
        Parameters
        ----------
        cache_dir : str
            Directory for caching downloaded files.
        """
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def download_file(self, file_name: str, url: str) -> pd.DataFrame:
        """
        Downloads file if not cached, returns DataFrame.

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
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            self._download_from_url(url, file_path)
        return pd.read_csv(file_path)

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

    def _normalize_player_id(self, df: pd.DataFrame, id_column: str = "player_id") -> None:
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
        df[id_column] = (
            df[id_column]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        df[id_column] = df[id_column].replace("", pd.NA).astype("Int64")

    def fetch_player_pool(
        self, draft_year: int, positions: list, start_year: int, end_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetches nflverse player stats and roster data."""
        stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
        kicking_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_kicking.csv"

        ps_df = self.download_file("player_stats.csv", stats_url)
        psk_df = self.download_file("player_stats_kicking.csv", kicking_url)

        merge_cols = list(set(psk_df.columns).intersection(set(ps_df.columns)))
        merged_df = ps_df.merge(psk_df, how="outer", on=merge_cols)
        merged_df = merged_df[merged_df["season"].between(start_year, end_year)]
        merged_df = merged_df[merged_df["position"].isin(positions)]

        legacy_stats_df = merged_df[merged_df["season"] < draft_year].copy()
        draft_year_stats_df = merged_df[merged_df["season"] == draft_year].copy()

        roster_url = f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{draft_year}.csv"
        draft_pool_df = self.download_file(f"roster_{draft_year}.csv", roster_url)
        draft_pool_df = draft_pool_df.rename(columns={"team": "recent_team"})
        draft_pool_df = draft_pool_df[draft_pool_df["position"].isin(positions)]

        self._normalize_player_id(legacy_stats_df)
        if "player_id" not in draft_pool_df.columns:
            draft_pool_df["player_id"] = draft_pool_df.index.astype(int)
        draft_pool_df["player_display_name"] = draft_pool_df["full_name"]
        self._normalize_player_id(draft_pool_df)

        return draft_pool_df, legacy_stats_df, draft_year_stats_df
