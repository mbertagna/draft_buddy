"""Tests for the canonical RL environment boundary."""

from __future__ import annotations

import numpy as np

from draft_buddy.rl.draft_gym_env import DraftGymEnv


def test_draft_gym_env_reset_returns_observation_and_action_mask(config, player_catalog) -> None:
    """Verify environment reset returns a normalized observation plus mask info."""
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    observation, info = env.reset()

    assert isinstance(observation, np.ndarray) and info["action_mask"].shape == (4,)


def test_draft_gym_env_step_updates_agent_roster(config, player_catalog) -> None:
    """Verify one valid action applies a pick through the shared controller."""
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    _observation, info = env.reset()
    action = int(np.flatnonzero(info["action_mask"])[0])
    _next_observation, _reward, _terminated, _truncated, _step_info = env.step(action)

    assert env.team_rosters[env.agent_team_id].size == 1
