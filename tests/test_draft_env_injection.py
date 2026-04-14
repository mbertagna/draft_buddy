"""Tests for draft env layout and dependency injection."""


def test_state_normalizer_class_lives_in_draft_env_state_normalizer_module():
    """Observation normalization belongs next to the Gym environment."""
    from draft_buddy.draft_env.state_normalizer import StateNormalizer

    assert StateNormalizer.__module__ == "draft_buddy.draft_env.state_normalizer"


def test_fantasy_draft_env_uses_injected_state_normalizer(mock_config):
    """Environment must use the provided StateNormalizer instance (dependency injection)."""
    from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
    from draft_buddy.draft_env.state_normalizer import StateNormalizer

    injected = StateNormalizer(mock_config)
    env = FantasyFootballDraftEnv(mock_config, state_normalizer=injected)

    assert env._state_normalizer is injected
