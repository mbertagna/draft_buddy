import numpy as np
import pytest
from draft_buddy.config import Config
from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv

def test_env_initialization(mock_config):
    """Verify that the environment initializes with correct dimensions."""
    env = FantasyFootballDraftEnv(mock_config)
    
    assert env.observation_space.shape == (len(mock_config.training.ENABLED_STATE_FEATURES),)
    assert env.action_space.n == mock_config.draft.ACTION_SPACE_SIZE

def test_env_reset(mock_config):
    """Verify that resetting the environment returns valid observations."""
    env = FantasyFootballDraftEnv(mock_config)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (len(mock_config.training.ENABLED_STATE_FEATURES),)
    assert "action_mask" in info

def test_env_step(mock_config):
    """Verify taking a step in the environment."""
    mock_config.draft.AGENT_START_POSITION = 1 # Ensure agent starts first for test
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    
    # Take QB pick (action 0)
    obs, reward, done, truncated, info = env.step(0)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "drafted_player" in info
