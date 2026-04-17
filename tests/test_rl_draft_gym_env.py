"""Tests for the canonical RL environment boundary."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_draft_gym_env_invalid_action_returns_penalty_and_ends_episode(config, player_catalog) -> None:
    """Verify invalid picks apply the configured penalty and terminate the episode."""
    config.reward.ENABLE_INVALID_ACTION_PENALTIES = True
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    _observation, _info = env.reset()
    env._controller.try_select_player_for_team = lambda *_args, **_kwargs: (False, None)
    env._get_best_available_player_by_pos = lambda _position: None

    _next_observation, reward, terminated, _truncated, info = env.step(0)

    assert terminated is True and info["invalid_action"] is True and reward == config.reward.INVALID_ACTION_PENALTIES["no_players_available"]


def test_draft_gym_env_set_current_team_picking_validates_team_range(config, player_catalog) -> None:
    """Verify invalid override team ids raise a descriptive error."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    with pytest.raises(ValueError, match="Invalid team ID"):
        env.set_current_team_picking(99)


def test_draft_gym_env_ai_suggestion_returns_error_when_model_missing(config, player_catalog) -> None:
    """Verify AI suggestions fail cleanly when no agent model is loaded."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env.agent_model = None

    assert env.get_ai_suggestion_for_team(1) == {"error": "AI model not loaded."}


def test_draft_gym_env_ai_suggestion_restores_available_players_after_ignore_list(
    config, player_catalog
) -> None:
    """Verify ignored players are removed temporarily and restored afterward."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    class FakeModel:
        def get_action_probabilities(self, *_args, **_kwargs):
            import torch

            return torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    original_available = set(env.available_player_ids)
    env.agent_model = FakeModel()
    suggestion = env.get_ai_suggestion_for_team(1, ignore_player_ids=[1, 9999])

    assert suggestion["TE"] == pytest.approx(0.4) and env.available_player_ids == original_available


def test_draft_gym_env_get_ai_suggestions_all_returns_error_without_model(
    config, player_catalog
) -> None:
    """Verify all-team suggestions surface the model-not-loaded error."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env.agent_model = None

    assert env.get_ai_suggestions_all() == {"error": "AI model not loaded."}


def test_draft_gym_env_load_matchups_returns_empty_when_files_missing(config, player_catalog) -> None:
    """Verify matchup loading returns an empty DataFrame when no schedule exists."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    assert env._load_matchups().empty


def test_draft_gym_env_stack_target_flag_detects_matching_receiver(config, player_catalog) -> None:
    """Verify stack-target detection finds WR/TE teammates for rostered QBs."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env.reset()
    env._controller.apply_pick(team_id=1, player_id=1, is_manual_pick=False)
    env._invalidate_sorted_available_cache()

    assert env._get_stack_target_available_flag_for_team(1) == 1
