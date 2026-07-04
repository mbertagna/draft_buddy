"""Canonical data package for loading and generating draft inputs."""

from draft_buddy.data.cache_paths import (
    adp_cache_dir,
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    nflverse_cache_dir,
    player_insights_output_path,
    sleeper_cache_dir,
)
from draft_buddy.data.data_processor import FantasyDataProcessor
from draft_buddy.data.insights import load_player_insights
from draft_buddy.data.nflverse_client import NflverseCsvDownloader
from draft_buddy.data.player_data_utils import get_simulation_dfs
from draft_buddy.data.player_loader import load_player_catalog
from draft_buddy.data.rookie_projector import RookieProjector
from draft_buddy.data.sleeper_catalog import SleeperCatalogBuilder
from draft_buddy.data.sleeper_client import SleeperGateway, SleeperHttpGateway

__all__ = [
    "FantasyDataProcessor",
    "NflverseCsvDownloader",
    "RookieProjector",
    "SleeperCatalogBuilder",
    "SleeperGateway",
    "SleeperHttpGateway",
    "adp_cache_dir",
    "get_simulation_dfs",
    "insights_search_cache_dir",
    "insights_synthesis_cache_dir",
    "load_player_catalog",
    "load_player_insights",
    "nflverse_cache_dir",
    "player_insights_output_path",
    "sleeper_cache_dir",
]
