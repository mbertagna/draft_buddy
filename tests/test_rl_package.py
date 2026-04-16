"""Tests for lazy RL package exports."""

import pytest

import draft_buddy.rl as rl_package


def test_rl_package_exposes_reinforce_agent_lazily():
    """Verify lazy attribute lookup returns the ReinforceAgent class."""
    from draft_buddy.rl.reinforce_agent import ReinforceAgent

    assert rl_package.ReinforceAgent is ReinforceAgent


def test_rl_package_exposes_agent_model_bot_lazily():
    """Verify lazy attribute lookup returns the AgentModelBotGM class."""
    from draft_buddy.rl.agent_bot import AgentModelBotGM

    assert rl_package.AgentModelBotGM is AgentModelBotGM


def test_rl_package_raises_attribute_error_for_unknown_symbols():
    """Verify unknown lazy exports raise the standard attribute error."""
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(rl_package, "MissingSymbol")
