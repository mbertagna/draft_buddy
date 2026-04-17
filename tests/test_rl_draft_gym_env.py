"""Tests for the canonical RL environment boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

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


def test_draft_gym_env_reset_marks_episode_ended_before_agent_pick(config, player_catalog) -> None:
    """Verify reset reports when the draft is already exhausted before the agent picks."""
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    env._controller.reset(draft_order=[], agent_team_id=1)
    env._controller.reset = lambda draft_order, agent_team_id: None
    env._state.current_pick_index = 0
    env._state.draft_order = []

    _observation, info = env.reset()

    assert info["episode_ended_before_agent_first_pick"] is True


def test_draft_gym_env_invalid_action_uses_roster_full_penalty_when_players_exist(
    config, player_catalog
) -> None:
    """Verify invalid picks use the roster-full penalty when players remain at the position."""
    config.reward.ENABLE_INVALID_ACTION_PENALTIES = True
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    _observation, _info = env.reset()
    env._controller.try_select_player_for_team = lambda *_args, **_kwargs: (False, None)
    env._get_best_available_player_by_pos = lambda _position: object()

    _next_observation, reward, terminated, _truncated, info = env.step(0)

    assert terminated is True and reward == config.reward.INVALID_ACTION_PENALTIES["roster_full_QB"]


def test_draft_gym_env_step_marks_premature_draft_end(config, player_catalog, monkeypatch) -> None:
    """Verify step reports premature draft end when the order is exhausted early."""
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)
    _observation, info = env.reset()
    action = int(np.flatnonzero(info["action_mask"])[0])
    original_apply_pick = env._controller.apply_pick

    def wrapped_apply_pick(*args, **kwargs):
        original_apply_pick(*args, **kwargs)
        env._state.current_pick_index = len(env.draft_order)

    monkeypatch.setattr(env._controller, "apply_pick", wrapped_apply_pick)

    _next_observation, _reward, terminated, _truncated, step_info = env.step(action)

    assert terminated is True and step_info["draft_ended_prematurely"] is True


def test_draft_gym_env_draft_and_undo_helpers_mutate_roster(config, player_catalog) -> None:
    """Verify manual helper methods draft and undo through the shared controller."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    env.draft_player(1)
    env.undo_last_pick()

    assert env.team_rosters[1].size == 0 and 1 in env.available_player_ids


def test_draft_gym_env_get_ai_suggestion_returns_draft_over_error(config, player_catalog) -> None:
    """Verify AI suggestions report when the draft is finished."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env._state.current_pick_index = len(env.draft_order)

    assert env.get_ai_suggestion() == {"error": "Draft is over."}


def test_draft_gym_env_get_ai_suggestion_rejects_invalid_team(config, player_catalog) -> None:
    """Verify invalid team ids return an explicit error payload."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env.agent_model = object()

    assert env.get_ai_suggestion_for_team(99) == {"error": "Invalid team id 99"}


def test_draft_gym_env_load_matchups_prefers_generated_schedule(config, player_catalog, monkeypatch) -> None:
    """Verify random-matchup mode delegates to schedule generation."""
    config.reward.USE_RANDOM_MATCHUPS = True
    monkeypatch.setattr(
        "draft_buddy.rl.draft_gym_env.generate_round_robin_schedule",
        lambda names, weeks: pd.DataFrame([{"Week": 1, "Home Manager(s)": names[0], "Away Manager(s)": names[1]}]),
    )
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    matchups = env._load_matchups()

    assert len(matchups) == 1


def test_draft_gym_env_load_matchups_prefers_size_specific_file(
    config, player_catalog, tmp_path: Path
) -> None:
    """Verify matchup loading prefers the team-count-specific CSV when present."""
    config.paths.DATA_DIR = str(tmp_path)
    size_specific_path = tmp_path / f"red_league_matchups_2025_{config.draft.NUM_TEAMS}_team.csv"
    default_path = tmp_path / "red_league_matchups_2025.csv"
    pd.DataFrame([{"Week": 1, "Matchup": 1}]).to_csv(size_specific_path, index=False)
    pd.DataFrame([{"Week": 2, "Matchup": 2}]).to_csv(default_path, index=False)
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    matchups = env._load_matchups()

    assert int(matchups.iloc[0]["Matchup"]) == 1


def test_draft_gym_env_vorp_and_kth_best_helpers_use_available_players(config, player_catalog) -> None:
    """Verify positional helper methods return expected best-player and VORP values."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)

    second_rb = env._get_kth_best_available_player_by_pos("RB", 2)
    vorp = env._calculate_vorp("RB")

    assert second_rb.player_id == 6 and vorp > 0.0


def test_draft_gym_env_bye_week_conflict_counts_matching_roster_players(config, player_catalog) -> None:
    """Verify bye-week conflict counting matches the best available player's bye week."""
    env = DraftGymEnv(config, training=False, player_catalog=player_catalog)
    env._controller.apply_pick(team_id=1, player_id=2, is_manual_pick=False)
    env._invalidate_sorted_available_cache()

    assert env._get_bye_week_conflict_count_for_team(1, "QB") == 1
