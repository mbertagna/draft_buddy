"""
Regression tests for manual/API draft flows: simulated picks, overrides, and per-team state.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv


def test_simulate_competing_pick_fallback_does_not_call_missing_method(mock_config):
    """
    When the opponent strategy returns no player, the env falls back to a random
    eligible pick using simulated roster rules (not a non-existent helper).
    """
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    team_on_clock = env.draft_order[env.current_pick_idx]
    mock_strategy = MagicMock()
    mock_strategy.execute_pick.return_value = None
    env._opponent_strategies[team_on_clock] = mock_strategy
    global_features = env._compute_global_state_features()
    # Must not raise AttributeError from a bad method name on the fallback path.
    env._simulate_competing_pick(team_on_clock, global_features)


def test_draft_player_clears_override_when_player_not_available(mock_config):
    """A failed manual draft must not leave an override active (UI would stay stuck)."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.set_current_team_picking(env.draft_order[env.current_pick_idx])
    assert env._overridden_team_id is not None
    with pytest.raises(ValueError, match="not available"):
        env.draft_player(999_999)
    assert env._overridden_team_id is None


def test_draft_player_clears_override_when_roster_rules_reject(mock_config):
    """When the roster rejects a pick, the override is cleared before raising."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.set_current_team_picking(env.draft_order[env.current_pick_idx])
    with patch.object(env, "_can_team_draft_position_manual", return_value=False):
        with pytest.raises(ValueError, match="cannot draft"):
            env.draft_player(1)
    assert env._overridden_team_id is None


def test_get_state_for_team_matches_get_state_for_team_on_clock(mock_config):
    """Default observation should match explicit state for the scheduled team on the clock."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    clock_team = env.draft_order[env.current_pick_idx]
    np.testing.assert_array_equal(
        env._get_state_for_team(clock_team),
        env._get_state(),
    )


def test_build_state_map_agent_start_position_matches_perspective_team(mock_config):
    """Per-team state maps must encode the requested team's slot, not the RL agent id."""
    mock_config.draft.AGENT_START_POSITION = 7
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    gf = env._compute_global_state_features()
    m1 = env._build_state_map_for_team(1, gf)
    m2 = env._build_state_map_for_team(2, gf)
    if "agent_start_position" in mock_config.training.ENABLED_STATE_FEATURES:
        assert m1["agent_start_position"] == 1.0
        assert m2["agent_start_position"] == 2.0


def test_get_state_for_team_differs_from_clock_when_rosters_diverge(mock_config):
    """State for team A must not equal state for team B when their rosters differ."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    mock_config.draft.NUM_TEAMS = 2
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    # Draft one player onto team 1 only (simulated single pick for team 2 later would advance clock;
    # instead mutate rosters directly for a deterministic check.)
    qb = env.player_map[1]
    env._state.add_player_to_roster(1, qb)
    env._invalidate_sorted_available_cache()
    s1 = env._get_state_for_team(1)
    s2 = env._get_state_for_team(2)
    assert not np.allclose(s1, s2)
