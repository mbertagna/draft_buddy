"""Tests for draft assistant GP filter."""

from __future__ import annotations

from draft_buddy.web.draft_advisor_filter import passes_gp_filter


def test_rookie_always_passes_gp_filter() -> None:
    """Verify rookies bypass the GP minimum."""
    assert passes_gp_filter("R", 0.75) is True


def test_empty_gp_min_passes_all_players() -> None:
    """Verify no threshold allows every finite GP value."""
    assert passes_gp_filter(0.2, None) is True


def test_gp_below_threshold_is_filtered() -> None:
    """Verify players below the threshold are excluded."""
    assert passes_gp_filter(0.5, 0.7) is False


def test_gp_at_threshold_passes() -> None:
    """Verify players at the threshold are included."""
    assert passes_gp_filter(0.7, 0.7) is True


def test_non_finite_gp_frac_is_filtered() -> None:
    """Verify invalid GP values are excluded when a threshold is set."""
    assert passes_gp_filter(None, 0.7) is False
    assert passes_gp_filter("bad", 0.7) is False
