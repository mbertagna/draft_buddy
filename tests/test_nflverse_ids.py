"""Tests for nflverse and Sleeper identifier normalization."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.nflverse_ids import normalize_gsis_id, normalize_sleeper_id


def test_normalize_gsis_id_strips_non_digits() -> None:
    """Verify GSIS ids normalize to integer nflverse player ids."""
    assert normalize_gsis_id("00-0039040") == 39040


def test_normalize_gsis_id_returns_none_for_missing_values() -> None:
    """Verify missing GSIS values normalize to None."""
    assert normalize_gsis_id(pd.NA) is None


def test_normalize_sleeper_id_strips_float_suffix() -> None:
    """Verify Sleeper ids read from CSV floats normalize to plain strings."""
    assert normalize_sleeper_id(9226.0) == "9226"


def test_normalize_sleeper_id_returns_none_for_missing_values() -> None:
    """Verify missing Sleeper ids normalize to None."""
    assert normalize_sleeper_id(None) is None
