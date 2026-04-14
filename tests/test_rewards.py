import pytest
from draft_buddy.utils.reward_calculator import RewardCalculator

def test_regular_season_reward(mock_config):
    # Mock records: (ManagerName, {'W': x, 'L': y, ...})
    records = [
        ('Ryan Freilich', {'W': 10}),
        ('Shane Spence', {'W': 9}),
        ('Paul Flores', {'W': 8}),
    ]
    
    # Ryan is seed 1
    reward, made, seed = RewardCalculator.compute_regular_season_reward(mock_config, records, 'Ryan Freilich')
    assert made is True
    assert seed == 1
    assert reward > 0

def test_stacking_reward_calculation(mock_config):
    from unittest.mock import MagicMock
    from draft_buddy.domain.entities import Player
    
    env = MagicMock()
    # Team 1 has Josh Allen (BUF)
    p1 = Player(1, "Josh Allen", "QB", 300.0, team="BUF")
    # Adding Stefon Diggs (BUF) should create a stack
    p2 = Player(2, "Stefon Diggs", "WR", 200.0, team="BUF")
    
    env.agent_team_id = 1
    env.teams_rosters = {1: {'PLAYERS': [p1, p2]}}
    
    # calculate_step_reward(config, env, drafted_player, prev_starter_points)
    reward, info = RewardCalculator.calculate_step_reward(mock_config, env, p2, 0.0)
    
    if mock_config.reward.ENABLE_STACKING_REWARD:
        assert info.get('stacking_reward', 0) > 0
