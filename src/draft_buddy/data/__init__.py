"""Canonical data package for loading and generating draft inputs."""

from draft_buddy.data.cache_paths import (
    adp_cache_dir,
    fantasypros_cache_dir,
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    model_adp_output_path,
    nflverse_cache_dir,
    player_insights_exports_dir,
    player_insights_output_path,
    resolve_latest_player_insights_path,
    sleeper_cache_dir,
)
from draft_buddy.data.data_processor import FantasyDataProcessor
from draft_buddy.data.insights import LoadedPlayerInsights, load_latest_player_insights, load_player_insights
from draft_buddy.data.market_data import (
    FantasyProsGateway,
    MarketDataUnavailable,
    MarketRankingRequest,
    consensus_adp_from_sources,
    rankings_payload_to_dataframe,
)
from draft_buddy.data.nflverse_client import NflverseCsvDownloader
from draft_buddy.data.player_data_utils import get_simulation_dfs
from draft_buddy.data.player_filter import exclude_inactive_players
from draft_buddy.data.player_loader import load_player_catalog
from draft_buddy.data.player_valuation import add_static_player_value_metrics
from draft_buddy.data.rookie_projector import RookieProjector
from draft_buddy.data.sleeper_catalog import SleeperCatalogBuilder
from draft_buddy.data.sleeper_client import SleeperGateway, SleeperHttpGateway
from draft_buddy.data.sleeper_draft_sync import SleeperDraftSyncService
from draft_buddy.data.sleeper_player_resolver import PlayerResolver, encode_sleeper_player_id
from draft_buddy.data.valuation_calibration import (
    apply_blend_weights,
    build_rolling_origin_splits,
    calculate_realized_player_values,
    evaluate_forecast_columns,
    fit_blend_weights,
    write_validation_report,
)

__all__ = [
    "FantasyDataProcessor",
    "NflverseCsvDownloader",
    "RookieProjector",
    "SleeperCatalogBuilder",
    "SleeperDraftSyncService",
    "SleeperGateway",
    "SleeperHttpGateway",
    "PlayerResolver",
    "encode_sleeper_player_id",
    "adp_cache_dir",
    "add_static_player_value_metrics",
    "apply_blend_weights",
    "build_rolling_origin_splits",
    "calculate_realized_player_values",
    "consensus_adp_from_sources",
    "evaluate_forecast_columns",
    "exclude_inactive_players",
    "fantasypros_cache_dir",
    "FantasyProsGateway",
    "fit_blend_weights",
    "get_simulation_dfs",
    "LoadedPlayerInsights",
    "insights_search_cache_dir",
    "insights_synthesis_cache_dir",
    "load_latest_player_insights",
    "load_player_catalog",
    "load_player_insights",
    "MarketDataUnavailable",
    "MarketRankingRequest",
    "model_adp_output_path",
    "nflverse_cache_dir",
    "player_insights_exports_dir",
    "player_insights_output_path",
    "resolve_latest_player_insights_path",
    "rankings_payload_to_dataframe",
    "sleeper_cache_dir",
    "write_validation_report",
]
