"""Tests for RL reward calculation behavior."""

from __future__ import annotations

import pandas as pd

from draft_buddy.rl.reward_calculator import RewardCalculator


def test_step_reward_adds_static_intermediate_reward(config, fake_env, player_factory) -> None:
    """Verify static intermediate rewards are added when enabled."""
    config.reward.ENABLE_INTERMEDIATE_REWARD = True
    config.reward.INTERMEDIATE_REWARD_MODE = "STATIC"
    config.reward.INTERMEDIATE_REWARD_VALUE = 3.0
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = False
    reward, info = RewardCalculator.calculate_step_reward(config, fake_env, player_factory(9, "QB", 9.0), 0.0)

    assert reward == 3.0 and info == {}


def test_step_reward_adds_vorp_shaping_when_enabled(config, fake_env, player_factory) -> None:
    """Verify VORP shaping uses env._calculate_vorp for the drafted position."""
    config.reward.ENABLE_INTERMEDIATE_REWARD = False
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = True
    config.reward.VORP_PICK_SHAPING_WEIGHT = 2.0
    config.reward.ENABLE_STACKING_REWARD = False
    reward, info = RewardCalculator.calculate_step_reward(config, fake_env, player_factory(9, "QB", 9.0), 0.0)

    assert reward == 20.0 and info["vorp_shaping"] == 10.0


def test_final_reward_adds_full_roster_bonus(config, fake_env) -> None:
    """Verify full rosters receive the configured bonus."""
    config.reward.BONUS_FOR_FULL_ROSTER = 4.0
    reward, _info = RewardCalculator.calculate_final_reward(config, fake_env, pd.DataFrame())

    assert reward >= 4.0


def test_compute_regular_season_reward_returns_non_playoff_zero(config) -> None:
    """Verify non-playoff teams receive no seed reward."""
    regular_records = [("Other", {"W": 1, "L": 0, "T": 0, "pts": 1.0})]

    assert RewardCalculator.compute_regular_season_reward(config, regular_records, "Team 1") == (0.0, False, None)


def test_compute_playoff_placement_reward_returns_champion_reward(config) -> None:
    """Verify champions receive the CHAMPION reward branch."""
    regular_records = [("Team 1", {}), ("Team 2", {})]
    playoff_results_df = pd.DataFrame([{"Week": 15, "Home Manager(s)": "Team 1", "Away Manager(s)": "Team 2", "Home Score": 10.0, "Away Score": 5.0}])
    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config,
        regular_records,
        playoff_results_df,
        "Team 1",
        "Team 1",
    )

    assert reward == config.reward.PLAYOFF_PLACEMENT_REWARDS["CHAMPION"] and label == "CHAMPION"
