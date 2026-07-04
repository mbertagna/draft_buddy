"""Tests for raw data cache directory conventions."""

from __future__ import annotations

from draft_buddy.data.cache_paths import (
    adp_cache_dir,
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    nflverse_cache_dir,
    player_insights_output_path,
    sleeper_cache_dir,
)


def test_nflverse_cache_dir_is_nested_under_data_root() -> None:
    """Verify the nflverse cache path is scoped under a 'cache/nflverse' subdirectory."""
    assert nflverse_cache_dir("./data") == "./data/cache/nflverse"


def test_sleeper_cache_dir_is_nested_under_data_root() -> None:
    """Verify the Sleeper cache path is scoped under a 'cache/sleeper' subdirectory."""
    assert sleeper_cache_dir("./data") == "./data/cache/sleeper"


def test_adp_cache_dir_is_nested_under_data_root() -> None:
    """Verify the ADP cache path is scoped under a 'cache/adp' subdirectory."""
    assert adp_cache_dir("./data") == "./data/cache/adp"


def test_insights_search_cache_dir_is_nested_under_data_root() -> None:
    """Verify the insights search cache path is under cache/insights/search."""
    assert insights_search_cache_dir("./data") == "./data/cache/insights/search"


def test_insights_synthesis_cache_dir_is_nested_under_data_root() -> None:
    """Verify the insights synthesis cache path is under cache/insights/synthesis."""
    assert insights_synthesis_cache_dir("./data") == "./data/cache/insights/synthesis"


def test_player_insights_output_path_includes_year() -> None:
    """Verify merged insights output path includes the draft year."""
    assert player_insights_output_path("./data", 2026) == "./data/player_insights_2026.json"
