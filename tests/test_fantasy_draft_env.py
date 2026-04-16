"""Contract tests for FantasyFootballDraftEnv Gym behavior."""

from unittest.mock import patch

import numpy as np

from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv


def test_reset_returns_observation_numpy_array_and_info_with_action_mask(mock_config):
    """Verify reset API returns observation and action_mask metadata."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    observation, info = env.reset()

    assert isinstance(observation, np.ndarray) and "action_mask" in info


def test_step_returns_five_tuple_contract_for_valid_action(mock_config):
    """Verify step returns Gym five-tuple for valid action."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    output = env.step(0)

    assert len(output) == 5


def test_step_valid_action_advances_current_pick_number(mock_config):
    """Verify a valid step advances pick number by one."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    before = env.current_pick_number
    env.step(0)

    assert env.current_pick_number > before


def test_get_action_mask_marks_qb_action_invalid_when_qb_slots_full(mock_config):
    """Verify action mask disables QB when team QB slots are exhausted."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    qb_limit = mock_config.draft.ROSTER_STRUCTURE["QB"] + mock_config.draft.BENCH_MAXES["QB"]
    env.teams_rosters[env.agent_team_id]["QB"] = qb_limit
    action_mask = env.get_action_mask()

    assert bool(action_mask[env.position_to_action["QB"]]) is False


def test_step_invalid_action_applies_configured_penalty_and_ends_episode(mock_config):
    """Verify invalid picks apply penalty and terminate safely."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    mock_config.reward.ENABLE_INVALID_ACTION_PENALTIES = True
    mock_config.reward.INVALID_ACTION_PENALTIES["roster_full_QB"] = -50.0
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    with patch.object(env, "_try_select_player_for_team", return_value=(False, None)):
        with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_final_reward", return_value=(0.0, {})):
            _, reward, done, _, _ = env.step(env.position_to_action["QB"])

    assert done is True and reward == -50.0
