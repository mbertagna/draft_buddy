"""Tests for RL reward calculation behavior."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_step_reward_adds_proportional_intermediate_reward(config, fake_env, player_factory) -> None:
    """Verify proportional intermediate rewards scale with projected points."""
    config.reward.ENABLE_INTERMEDIATE_REWARD = True
    config.reward.INTERMEDIATE_REWARD_MODE = "PROPORTIONAL"
    config.reward.PROPORTIONAL_REWARD_SCALING_FACTOR = 0.5
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = False

    reward, _info = RewardCalculator.calculate_step_reward(
        config, fake_env, player_factory(9, "QB", 12.0), 0.0
    )

    assert reward == 6.0


def test_step_reward_adds_pick_shaping_delta_when_starters_improve(
    config, fake_env, player_factory
) -> None:
    """Verify starter delta shaping is added when the starter score increases."""
    config.reward.ENABLE_INTERMEDIATE_REWARD = False
    config.reward.ENABLE_PICK_SHAPING_REWARD = True
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = True
    config.reward.PICK_SHAPING_STARTER_DELTA_WEIGHT = 2.0
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = False

    reward, info = RewardCalculator.calculate_step_reward(
        config, fake_env, player_factory(9, "WR", 50.0, team="BUF"), 300.0
    )

    assert reward > 0.0 and info["pick_shaping_delta"] > 0.0


def test_step_reward_adds_stacking_reward_for_new_stack(config, fake_env, player_factory) -> None:
    """Verify stacking reward is added when the new pick creates a stack."""
    config.reward.ENABLE_INTERMEDIATE_REWARD = False
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = True
    config.reward.STACKING_REWARD_WEIGHT = 3.0
    fake_env._resolved[1] = [
        player_factory(1, "QB", 200.0, "BUF"),
        player_factory(2, "WR", 150.0, "BUF"),
        player_factory(9, "TE", 100.0, "BUF"),
    ]

    reward, info = RewardCalculator.calculate_step_reward(
        config, fake_env, player_factory(9, "TE", 100.0, "BUF"), 0.0
    )

    assert reward == 3.0 and info["stacking_reward"] == 3.0


def test_final_reward_adds_full_roster_bonus(config, fake_env) -> None:
    """Verify full rosters receive the configured bonus."""
    config.reward.BONUS_FOR_FULL_ROSTER = 4.0
    reward, _info = RewardCalculator.calculate_final_reward(config, fake_env, pd.DataFrame())

    assert reward >= 4.0


def test_final_reward_adds_competitive_component_against_max_opponent(config, fake_env) -> None:
    """Verify competitive rewards compare the agent against the max opponent score."""
    config.reward.ENABLE_COMPETITIVE_REWARD = True
    config.reward.COMPETITIVE_REWARD_MODE = "MAX_OPPONENT_DIFFERENCE"
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = False

    reward, info = RewardCalculator.calculate_final_reward(config, fake_env, pd.DataFrame())

    assert info["competitive_mode"] == "Max Opponent Difference" and "competitive_reward_component" in info and reward > 0.0


def test_final_reward_adds_average_opponent_mode_and_std_penalty(config, fake_env) -> None:
    """Verify average-opponent mode and std-dev penalty both contribute."""
    config.reward.ENABLE_COMPETITIVE_REWARD = True
    config.reward.COMPETITIVE_REWARD_MODE = "AVG_OPPONENT_DIFFERENCE"
    config.reward.ENABLE_OPPONENT_STD_DEV_PENALTY = True
    config.reward.OPPONENT_STD_DEV_PENALTY_WEIGHT = 1.0
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = False
    fake_env.team_rosters = {
        1: SimpleNamespace(player_ids=[1, 2]),
        2: SimpleNamespace(player_ids=[3, 4]),
        3: SimpleNamespace(player_ids=[5, 6]),
    }
    fake_env._resolved[3] = [
        SimpleNamespace(projected_points=100.0, position="QB", team="DAL", bye_week=7),
        SimpleNamespace(projected_points=80.0, position="WR", team="DAL", bye_week=7),
    ]

    reward, info = RewardCalculator.calculate_final_reward(config, fake_env, pd.DataFrame())

    assert info["competitive_mode"] == "Average Opponent Difference" and "opponent_std_dev_penalty" in info and reward > 0.0


def test_final_reward_adds_season_simulation_rewards(config, fake_env, monkeypatch) -> None:
    """Verify season-simulation rewards are merged into the terminal reward."""
    config.reward.ENABLE_SEASON_SIM_REWARD = True
    config.reward.REGULAR_SEASON_REWARD = {
        "NUM_PLAYOFF_TEAMS": 2,
        "MAKE_PLAYOFFS_BONUS": 1.0,
        "SEED_REWARD_MODE": "MAPPING",
        "SEED_REWARD_MAPPING": {1: 3.0},
    }
    config.reward.PLAYOFF_PLACEMENT_REWARDS = {"CHAMPION": 5.0, "RUNNER_UP": 2.0, "NON_PLAYOFF": 0.0}
    monkeypatch.setattr(
        "draft_buddy.rl.reward_calculator.simulate_season_fast",
        lambda *_args, **_kwargs: (
            None,
            [("Team 1", {}), ("Team 2", {})],
            pd.DataFrame(
                [{"Week": 15, "Home Manager(s)": "Team 1", "Away Manager(s)": "Team 2", "Home Score": 10.0, "Away Score": 5.0}]
            ),
            "",
            "Team 1",
        ),
    )

    reward, info = RewardCalculator.calculate_final_reward(
        config, fake_env, pd.DataFrame([{"Week": 1}])
    )

    assert info["playoff_placement"] == "CHAMPION" and info["season_sim_reward"] == 9.0 and reward >= 9.0


def test_final_reward_ignores_season_simulation_exceptions(config, fake_env, monkeypatch, capsys) -> None:
    """Verify season simulation errors are swallowed after printing a message."""
    config.reward.ENABLE_SEASON_SIM_REWARD = True
    monkeypatch.setattr(
        "draft_buddy.rl.reward_calculator.simulate_season_fast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    reward, info = RewardCalculator.calculate_final_reward(
        config, fake_env, pd.DataFrame([{"Week": 1}])
    )

    assert "season_sim_reward" not in info and "boom" in capsys.readouterr().out and reward >= 0.0


def test_compute_regular_season_reward_returns_non_playoff_zero(config) -> None:
    """Verify non-playoff teams receive no seed reward."""
    regular_records = [("Other", {"W": 1, "L": 0, "T": 0, "pts": 1.0})]

    assert RewardCalculator.compute_regular_season_reward(config, regular_records, "Team 1") == (0.0, False, None)


def test_compute_regular_season_reward_uses_linear_seed_interpolation(config) -> None:
    """Verify linear seed rewards interpolate between max and min."""
    config.reward.REGULAR_SEASON_REWARD = {
        "NUM_PLAYOFF_TEAMS": 3,
        "MAKE_PLAYOFFS_BONUS": 2.0,
        "SEED_REWARD_MODE": "LINEAR",
        "SEED_REWARD_MAX": 6.0,
        "SEED_REWARD_MIN": 2.0,
    }
    regular_records = [("Team 2", {}), ("Team 1", {}), ("Team 3", {})]

    reward, made_playoffs, seed = RewardCalculator.compute_regular_season_reward(
        config, regular_records, "Team 1"
    )

    assert reward == 6.0 and made_playoffs is True and seed == 2


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


def test_compute_playoff_placement_reward_returns_runner_up_when_agent_loses_final(config) -> None:
    """Verify the losing finalist receives the runner-up reward."""
    regular_records = [("Team 1", {}), ("Team 2", {})]
    playoff_results_df = pd.DataFrame(
        [{"Week": 15, "Home Manager(s)": "Team 1", "Away Manager(s)": "Team 2", "Home Score": 5.0, "Away Score": 10.0}]
    )

    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config, regular_records, playoff_results_df, "Team 2", "Team 1"
    )

    assert reward == config.reward.PLAYOFF_PLACEMENT_REWARDS["RUNNER_UP"] and label == "RUNNER_UP"


def test_compute_playoff_placement_reward_returns_quarterfinalist_fallback(config) -> None:
    """Verify placement falls back to quarterfinalist when bracket parsing fails."""
    regular_records = [("Team 1", {}), ("Team 2", {}), ("Team 3", {}), ("Team 4", {})]

    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config, regular_records, pd.DataFrame(), "Team 2", "Team 1"
    )

    assert reward == config.reward.PLAYOFF_PLACEMENT_REWARDS["QUARTERFINALIST"] and label == "QUARTERFINALIST"


def test_compute_playoff_placement_reward_returns_non_playoff_label(config) -> None:
    """Verify non-playoff teams receive the non-playoff placement branch."""
    regular_records = [("Other", {}), ("Another", {})]

    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config, regular_records, pd.DataFrame(), "Other", "Team 1"
    )

    assert reward == config.reward.PLAYOFF_PLACEMENT_REWARDS["NON_PLAYOFF"] and label == "NON_PLAYOFF"
