"""Tests for raw data cache directory conventions."""

from __future__ import annotations

from draft_buddy.data.cache_paths import adp_cache_dir, nflverse_cache_dir, sleeper_cache_dir


def test_nflverse_cache_dir_is_nested_under_data_root() -> None:
    """Verify the nflverse cache path is scoped under a 'cache/nflverse' subdirectory."""
    assert nflverse_cache_dir("./data") == "./data/cache/nflverse"


def test_sleeper_cache_dir_is_nested_under_data_root() -> None:
    """Verify the Sleeper cache path is scoped under a 'cache/sleeper' subdirectory."""
    assert sleeper_cache_dir("./data") == "./data/cache/sleeper"


def test_adp_cache_dir_is_nested_under_data_root() -> None:
    """Verify the ADP cache path is scoped under a 'cache/adp' subdirectory."""
    assert adp_cache_dir("./data") == "./data/cache/adp"
