"""Tests for canonical RL package exports."""

from __future__ import annotations

import pytest

import draft_buddy.rl as rl_package


def test_rl_package_exposes_draft_gym_env_lazily() -> None:
    """Verify the canonical environment export is available from the package root."""
    from draft_buddy.rl.draft_gym_env import DraftGymEnv

    assert rl_package.DraftGymEnv is DraftGymEnv


def test_rl_package_raises_attribute_error_for_unknown_symbol() -> None:
    """Verify unknown lazy exports raise the standard attribute error."""
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(rl_package, "MissingSymbol")
