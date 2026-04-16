"""Tests for RL reward calculation logic."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from draft_buddy.config import Config
from draft_buddy.domain.entities import Player
from draft_buddy.rl.reward_calculator import RewardCalculator


def test_calculate_step_reward_proportional_mode_returns_expected_float():
    """Verify proportional intermediate reward equals projected points times scaling."""
    config = Config()
    config.reward.ENABLE_INTERMEDIATE_REWARD = True
    config.reward.INTERMEDIATE_REWARD_MODE = "PROPORTIONAL"
    config.reward.PROPORTIONAL_REWARD_SCALING_FACTOR = 1.0
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = False
    env = SimpleNamespace(teams_rosters={1: {"PLAYERS": []}}, agent_team_id=1)
    drafted_player = Player(1, "Reward Player", "WR", 150.0)
    reward, _ = RewardCalculator.calculate_step_reward(config, env, drafted_player, 0.0)

    assert reward == 150.0


def test_calculate_step_reward_adds_stacking_weight_when_stack_increases():
    """Verify stacking reward uses configured stacking weight."""
    config = Config()
    config.reward.ENABLE_INTERMEDIATE_REWARD = False
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = True
    config.reward.STACKING_REWARD_WEIGHT = 5.0
    roster = [
        Player(1, "QB BUF", "QB", 300.0, team="BUF"),
        Player(2, "WR BUF", "WR", 200.0, team="BUF"),
    ]
    env = SimpleNamespace(teams_rosters={1: {"PLAYERS": roster}}, agent_team_id=1)
    drafted_player = roster[-1]
    reward, _ = RewardCalculator.calculate_step_reward(config, env, drafted_player, 0.0)

    assert reward == 5.0


def test_calculate_final_reward_golden_master_runner_up_path_total():
    """Golden-master lock for terminal reward with runner-up season simulation output."""
    config = Config()
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = True
    config.reward.STARTER_POINTS_WEIGHT = 1.0
    config.reward.BENCH_POINTS_WEIGHT = 0.5
    config.reward.ENABLE_FINAL_BASE_REWARD = True
    config.reward.BONUS_FOR_FULL_ROSTER = 0.0
    config.reward.ENABLE_COMPETITIVE_REWARD = True
    config.reward.COMPETITIVE_REWARD_MODE = "MAX_OPPONENT_DIFFERENCE"
    config.reward.ENABLE_OPPONENT_STD_DEV_PENALTY = True
    config.reward.OPPONENT_STD_DEV_PENALTY_WEIGHT = 0.1
    config.reward.ENABLE_SEASON_SIM_REWARD = True
    config.reward.REGULAR_SEASON_REWARD = {
        "SEED_REWARD_MODE": "MAPPING",
        "NUM_PLAYOFF_TEAMS": 2,
        "MAKE_PLAYOFFS_BONUS": 0.0,
        "SEED_REWARD_MAPPING": {1: 0.0, 2: 0.0},
    }
    config.reward.PLAYOFF_PLACEMENT_REWARDS = {
        "CHAMPION": 100.0,
        "RUNNER_UP": 40.0,
        "SEMIFINALIST": 15.0,
        "QUARTERFINALIST": 5.0,
        "NON_PLAYOFF": 0.0,
    }
    config.draft.ROSTER_STRUCTURE = {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0}
    config.draft.BENCH_MAXES = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    config.draft.TEAM_MANAGER_MAPPING = {1: "Agent", 2: "OppA", 3: "OppB"}

    env = SimpleNamespace(
        agent_team_id=1,
        teams_rosters={
            1: {"PLAYERS": [Player(1, "A QB", "QB", 100.0), Player(2, "A RB", "RB", 80.0), Player(3, "A WR", "WR", 70.0)]},
            2: {"PLAYERS": [Player(4, "B QB", "QB", 90.0), Player(5, "B RB", "RB", 60.0), Player(6, "B WR", "WR", 50.0)]},
            3: {"PLAYERS": [Player(7, "C QB", "QB", 95.0), Player(8, "C RB", "RB", 85.0), Player(9, "C WR", "WR", 65.0)]},
        },
        total_roster_size_per_team=3,
        weekly_projections={},
    )
    matchups_df = pd.DataFrame([{"Week": 1, "Away Manager(s)": "Agent", "Home Manager(s)": "OppA"}])
    regular_records = [
        ("OppB", {"W": 3, "L": 0, "T": 0, "pts": 245.0}),
        ("Agent", {"W": 2, "L": 1, "T": 0, "pts": 250.0}),
        ("OppA", {"W": 1, "L": 2, "T": 0, "pts": 200.0}),
    ]
    playoff_results = pd.DataFrame(
        [{"Week": 15, "Matchup": 1, "Away Manager(s)": "Agent", "Away Score": 120.0, "Home Score": 130.0, "Home Manager(s)": "OppB"}]
    )
    with patch(
        "draft_buddy.rl.reward_calculator.simulate_season_fast",
        return_value=(pd.DataFrame(), regular_records, playoff_results, "tree", "OppB"),
    ):
        reward, _ = RewardCalculator.calculate_final_reward(config, env, matchups_df)

    assert reward == 292.75


def test_compute_regular_season_reward_returns_non_playoff_when_manager_missing():
    """Verify regular season reward returns non-playoff tuple when manager misses playoffs."""
    config = Config()
    config.reward.REGULAR_SEASON_REWARD = {"NUM_PLAYOFF_TEAMS": 2, "MAKE_PLAYOFFS_BONUS": 10.0}
    records = [("A", {"W": 5}), ("B", {"W": 4}), ("C", {"W": 3})]
    reward, made_playoffs, seed = RewardCalculator.compute_regular_season_reward(config, records, "C")

    assert (reward, made_playoffs, seed) == (0.0, False, None)


def test_compute_regular_season_reward_uses_mapping_mode_when_configured():
    """Verify mapping mode uses configured seed mapping value."""
    config = Config()
    config.reward.REGULAR_SEASON_REWARD = {
        "SEED_REWARD_MODE": "MAPPING",
        "NUM_PLAYOFF_TEAMS": 2,
        "MAKE_PLAYOFFS_BONUS": 1.0,
        "SEED_REWARD_MAPPING": {1: 5.0, 2: 3.0},
    }
    records = [("Agent", {"W": 5}), ("Other", {"W": 4})]
    reward, _, _ = RewardCalculator.compute_regular_season_reward(config, records, "Agent")

    assert reward == 6.0


def test_compute_playoff_placement_reward_returns_non_playoff_when_agent_not_qualified():
    """Verify non-playoff placement reward when manager misses playoff list."""
    config = Config()
    config.reward.REGULAR_SEASON_REWARD = {"NUM_PLAYOFF_TEAMS": 2}
    config.reward.PLAYOFF_PLACEMENT_REWARDS = {"NON_PLAYOFF": 2.0}
    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config,
        regular_records=[("A", {}), ("B", {}), ("Agent", {})],
        playoff_results_df=pd.DataFrame(),
        winner="A",
        agent_manager_name="Agent",
    )

    assert (reward, label) == (2.0, "NON_PLAYOFF")


def test_compute_playoff_placement_reward_returns_champion_for_winner():
    """Verify champion reward when agent is winner."""
    config = Config()
    config.reward.REGULAR_SEASON_REWARD = {"NUM_PLAYOFF_TEAMS": 2}
    config.reward.PLAYOFF_PLACEMENT_REWARDS = {"CHAMPION": 9.0}
    reward, label = RewardCalculator.compute_playoff_placement_reward(
        config,
        regular_records=[("Agent", {}), ("B", {})],
        playoff_results_df=pd.DataFrame([{"Week": 15, "Home Manager(s)": "Agent", "Away Manager(s)": "B"}]),
        winner="Agent",
        agent_manager_name="Agent",
    )

    assert (reward, label) == (9.0, "CHAMPION")
