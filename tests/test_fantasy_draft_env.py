"""Contract tests for FantasyFootballDraftEnv Gym behavior."""

from unittest.mock import patch

import numpy as np
import pytest

from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
from draft_buddy.domain.entities import Player


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


def test_reset_uses_randomized_agent_position_when_training_enabled(mock_config):
    """Verify training reset uses a randomized agent slot when configured."""
    mock_config.draft.NUM_TEAMS = 4
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.RANDOMIZE_AGENT_START_POSITION = True
    env = FantasyFootballDraftEnv(mock_config, training=True)

    with patch("random.randint", return_value=3):
        env.reset()

    assert env.agent_team_id == 3


def test_reset_marks_episode_ended_before_first_pick_when_schedule_exhausted(mock_config):
    """Verify reset flags drafts that end before the agent can pick."""
    mock_config.draft.AGENT_START_POSITION = 1
    env = FantasyFootballDraftEnv(mock_config, training=True)
    env.all_players_data = []
    env.player_map = {}

    with patch.object(env, "_simulate_competing_pick", return_value=None):
        _, info = env.reset()

    assert info["episode_ended_before_agent_first_pick"] is True


def test_generate_snake_draft_order_returns_empty_when_no_players_available(mock_config):
    """Verify snake order is empty when the player pool is empty."""
    env = FantasyFootballDraftEnv(mock_config)
    env.all_players_data = []

    assert env._generate_snake_draft_order(num_teams=4, total_picks_per_team=5) == []


def test_generate_snake_draft_order_uses_single_round_when_pool_smaller_than_team_count(mock_config):
    """Verify short player pools still produce at least one round when possible."""
    env = FantasyFootballDraftEnv(mock_config)
    env.all_players_data = [object(), object(), object()]

    assert env._generate_snake_draft_order(num_teams=12, total_picks_per_team=5) == list(range(1, 13))


def test_step_invalid_action_without_penalties_keeps_reward_at_zero(mock_config):
    """Verify invalid picks can terminate without deducting points."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    mock_config.reward.ENABLE_INVALID_ACTION_PENALTIES = False

    with patch.object(env, "_try_select_player_for_team", return_value=(False, None)):
        with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_final_reward", return_value=(0.0, {})):
            _, reward, done, _, info = env.step(env.position_to_action["QB"])

    assert done is True and reward == 0.0 and info["invalid_action"] is True


def test_step_marks_draft_complete_when_agent_roster_hits_capacity(mock_config):
    """Verify step reports draft completion when the agent fills the roster."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.teams_rosters[1]["PLAYERS"] = [env.player_map[2]] * (env.total_roster_size_per_team - 1)

    with patch.object(env, "_try_select_player_for_team", return_value=(True, env.player_map[1])):
        with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_step_reward", return_value=(0.0, {})):
            with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_final_reward", return_value=(0.0, {})):
                _, _, done, _, info = env.step(env.position_to_action["QB"])

    assert done is True and info["draft_complete"] is True


def test_step_marks_premature_end_when_schedule_exhausted_after_pick(mock_config):
    """Verify step reports a premature end when the draft order is exhausted."""
    mock_config.draft.NUM_TEAMS = 1
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.current_pick_idx = len(env.draft_order) - 1
    env.current_pick_number = len(env.draft_order)

    with patch.object(env, "_try_select_player_for_team", return_value=(True, env.player_map[1])):
        with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_step_reward", return_value=(0.0, {})):
            with patch("draft_buddy.rl.reward_calculator.RewardCalculator.calculate_final_reward", return_value=(0.0, {})):
                _, _, done, _, info = env.step(env.position_to_action["QB"])

    assert done is True and info["draft_ended_prematurely"] is True


def test_get_info_uses_override_team_when_present(mock_config):
    """Verify info payload reports the override team on the clock."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.set_current_team_picking(4)

    assert env._get_info()["current_team_picking"] == 4


def test_get_info_reports_none_when_draft_order_exhausted(mock_config):
    """Verify info payload reports no team on clock after the schedule ends."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.current_pick_idx = len(env.draft_order)

    assert env._get_info()["current_team_picking"] is None


def test_get_next_opponent_team_id_returns_agent_when_no_opponent_pick_remains(mock_config):
    """Verify next-opponent lookup falls back to the agent when no future opponent exists."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.current_pick_idx = len(env.draft_order) - 1

    assert env._get_next_opponent_team_id() == env.agent_team_id


def test_get_opponent_roster_count_returns_zero_for_unknown_team(mock_config):
    """Verify unknown teams report zero roster count."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()

    assert env._get_opponent_roster_count(999, "RB") == 0


def test_get_bye_week_conflict_count_returns_zero_when_best_player_has_no_bye(mock_config):
    """Verify missing bye weeks do not count as conflicts."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    player = env._get_best_available_player_by_pos("QB")
    player.bye_week = None

    assert env._get_bye_week_conflict_count("QB") == 0


def test_get_bye_week_vector_for_team_counts_only_weeks_four_through_fourteen(mock_config):
    """Verify bye-week vector excludes out-of-range and missing values."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.teams_rosters[1]["PLAYERS"] = [
        Player(201, "Week Four", "RB", 100.0, bye_week=4),
        Player(202, "Week Fourteen", "WR", 100.0, bye_week=14),
        Player(203, "Out Of Range", "TE", 100.0, bye_week=15),
        Player(204, "Missing", "QB", 100.0, bye_week=None),
    ]

    vector = env._get_bye_week_vector_for_team(1)

    assert vector[0] == 1 and vector[10] == 1 and float(vector.sum()) == 2.0


def test_get_stack_target_available_flag_for_team_returns_zero_without_qb_team(mock_config):
    """Verify stack target lookup requires a rostered quarterback with a team."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.teams_rosters[1]["PLAYERS"] = [Player(205, "Wideout", "WR", 120.0, team="BUF")]

    assert env._get_stack_target_available_flag_for_team(1) == 0


def test_get_stack_target_available_flag_for_team_returns_one_when_matching_receiver_available(mock_config):
    """Verify stack target lookup detects a matching WR or TE in the pool."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.teams_rosters[1]["PLAYERS"] = [Player(206, "Quarterback", "QB", 200.0, team="BUF")]
    target = next(player for player in env.player_map.values() if player.position in {"WR", "TE"})
    target.team = "BUF"

    assert env._get_stack_target_available_flag_for_team(1) == 1


def test_simulate_single_pick_force_picks_lowest_finite_adp_when_bot_returns_none(mock_config):
    """Verify simulated picks fall back to the best available ADP when no bot pick is returned."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    target_team = env.draft_order[env.current_pick_idx]

    with patch.object(env, "_simulate_competing_pick", return_value=None):
        env.simulate_single_pick()

    assert env._draft_history[-1]["team_id"] == target_team


def test_simulate_single_pick_skips_team_when_roster_is_full_and_no_pick_available(mock_config):
    """Verify simulated picks can skip full teams that have no valid selection."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    target_team = env.draft_order[env.current_pick_idx]
    env.teams_rosters[target_team]["PLAYERS"] = [env.player_map[1]] * env.total_roster_size_per_team
    before_pick = env.current_pick_number

    with patch.object(env, "_simulate_competing_pick", return_value=None):
        env.simulate_single_pick()

    assert env.current_pick_number == before_pick + 1


def test_undo_last_pick_restores_previous_pick_state(mock_config):
    """Verify undo rewinds pick counters after a manual draft."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    original_pick_number = env.current_pick_number
    player_id = next(iter(env.available_players_ids))
    env.draft_player(player_id)

    env.undo_last_pick()

    assert env.current_pick_number == original_pick_number


def test_undo_last_pick_raises_when_history_is_empty(mock_config):
    """Verify undo rejects requests when nothing has been drafted."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()

    with pytest.raises(ValueError, match="No picks to undo"):
        env.undo_last_pick()


def test_set_current_team_picking_rejects_invalid_team_ids(mock_config):
    """Verify manual overrides validate team boundaries."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()

    with pytest.raises(ValueError, match="Invalid team ID"):
        env.set_current_team_picking(0)


def test_get_ai_suggestions_all_returns_error_without_loaded_model(mock_config):
    """Verify bulk AI suggestions fail gracefully when no model is loaded."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.agent_model = None

    assert env.get_ai_suggestions_all() == {"error": "AI model not loaded."}


def test_get_ai_suggestion_for_team_restores_available_ids_after_ignoring_players(mock_config):
    """Verify ignore lists do not permanently mutate available players."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    before_ids = set(env.available_players_ids)
    env.agent_model = type(
        "StubModel",
        (),
        {"get_action_probabilities": lambda self, state, action_mask=None: np.array([[0.1, 0.2, 0.3, 0.4]])},
    )()

    result = env.get_ai_suggestion_for_team(team_id=1, ignore_player_ids=[1, 2])

    assert set(env.available_players_ids) == before_ids and result["TE"] == 0.4
