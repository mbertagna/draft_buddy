"""
Sleeper-backed client for fetching NFL player and league roster data.

Defines the abstract ``SleeperGateway`` contract and a concrete HTTP
implementation that caches the (large) player directory locally, per
Sleeper's guidance to call that endpoint at most once per day.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import requests

PLAYER_DIRECTORY_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

PLAYER_DATAFRAME_COLUMNS = [
    "sleeper_id",
    "full_name",
    "position",
    "team",
    "status",
    "injury_status",
    "depth_chart_position",
    "depth_chart_order",
    "gsis_id",
    "years_exp",
]


class SleeperGateway(ABC):
    """Abstract interface for fetching Sleeper player and roster data."""

    @abstractmethod
    def fetch_all_players(self) -> pd.DataFrame:
        """Fetch the full Sleeper NFL player directory.

        Returns
        -------
        pd.DataFrame
            One row per Sleeper player with columns matching
            ``PLAYER_DATAFRAME_COLUMNS``.
        """

    @abstractmethod
    def fetch_league_rosters(self, league_id: str) -> pd.DataFrame:
        """Fetch rostered Sleeper player ids for every team in a league.

        Parameters
        ----------
        league_id : str
            Sleeper league identifier.

        Returns
        -------
        pd.DataFrame
            One row per rostered player with columns ``roster_id`` and
            ``sleeper_id``.
        """


class SleeperHttpGateway(SleeperGateway):
    """Fetches Sleeper data over HTTP with a daily-refreshed local cache."""

    _BASE_URL = "https://api.sleeper.app/v1"
    _PLAYERS_CACHE_FILENAME = "sleeper_players.json"

    def __init__(
        self,
        cache_dir: str = "./data",
        cache_max_age_seconds: int = PLAYER_DIRECTORY_CACHE_MAX_AGE_SECONDS,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir : str, optional
            Directory used to cache the player directory download.
        cache_max_age_seconds : int, optional
            Maximum cache age, in seconds, before re-downloading the
            player directory.
        """
        self._cache_dir = cache_dir
        self._cache_max_age_seconds = cache_max_age_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_all_players(self) -> pd.DataFrame:
        """Fetch the full Sleeper NFL player directory, using a daily cache."""
        raw_players = self._load_cached_players()
        if raw_players is None:
            raw_players = self._download_and_cache_players()
        return self._to_player_dataframe(raw_players)

    def fetch_league_rosters(self, league_id: str) -> pd.DataFrame:
        """Fetch rostered Sleeper player ids for every team in a league."""
        rosters = self._get_json(f"{self._BASE_URL}/league/{league_id}/rosters")
        rows = [
            {"roster_id": roster.get("roster_id"), "sleeper_id": player_id}
            for roster in rosters
            for player_id in (roster.get("players") or [])
        ]
        return pd.DataFrame(rows, columns=["roster_id", "sleeper_id"])

    def _get_json(self, url: str):
        """Issue a GET request and return the parsed JSON body."""
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _cache_file_path(self) -> str:
        """Return the local cache file path for the player directory."""
        return os.path.join(self._cache_dir, self._PLAYERS_CACHE_FILENAME)

    def _load_cached_players(self) -> Optional[dict]:
        """Return cached player directory contents when fresh, else None."""
        cache_path = self._cache_file_path()
        if not os.path.exists(cache_path):
            return None
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds > self._cache_max_age_seconds:
            return None
        with open(cache_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _download_and_cache_players(self) -> dict:
        """Download the player directory and persist it to the local cache."""
        players = self._get_json(f"{self._BASE_URL}/players/nfl")
        with open(self._cache_file_path(), "w", encoding="utf-8") as file_obj:
            json.dump(players, file_obj)
        return players

    @staticmethod
    def _to_player_dataframe(raw_players: dict) -> pd.DataFrame:
        """Flatten the raw Sleeper player directory into a DataFrame."""
        rows = [
            {
                "sleeper_id": sleeper_id,
                "full_name": player.get("full_name"),
                "position": player.get("position"),
                "team": player.get("team"),
                "status": player.get("status"),
                "injury_status": player.get("injury_status"),
                "depth_chart_position": player.get("depth_chart_position"),
                "depth_chart_order": player.get("depth_chart_order"),
                "gsis_id": player.get("gsis_id"),
                "years_exp": player.get("years_exp"),
            }
            for sleeper_id, player in raw_players.items()
        ]
        return pd.DataFrame(rows, columns=PLAYER_DATAFRAME_COLUMNS)
