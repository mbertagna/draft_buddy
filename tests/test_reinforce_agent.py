"""Behavioral tests for the REINFORCE agent."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from draft_buddy.rl.reinforce_agent import ReinforceAgent


class _StubEnv:
    """Minimal environment stub for agent tests."""

    def __init__(self):
        self.action_space = SimpleNamespace(n=4)
        self.agent_team_id = 1
        self.teams_rosters = {1: {"PLAYERS": [SimpleNamespace(projected_points=10.0)]}}

    def reset(self):
        """Return the initial state and action mask."""
        return np.zeros(2, dtype=np.float32), {"action_mask": np.array([True, False, True, True])}

    def step(self, action):
        """Terminate after one step and return a new action mask."""
        del action
        return np.ones(2, dtype=np.float32), 2.5, True, False, {"action_mask": np.array([True, True, True, True])}


def _build_config(mock_config):
    """Shrink the training config for deterministic tests."""
    mock_config.training.ENABLED_STATE_FEATURES = ["f1", "f2"]
    mock_config.training.HIDDEN_DIM = 4
    mock_config.training.LEARNING_RATE = 0.01
    mock_config.training.DISCOUNT_FACTOR = 0.5
    mock_config.training.TOTAL_EPISODES = 2
    mock_config.training.BATCH_EPISODES = 1
    mock_config.training.LOG_SAVE_INTERVAL_EPISODES = 1
    return mock_config


def test_calculate_returns_applies_discount_factor(mock_config):
    """Verify discounted returns are accumulated from the end of the episode."""
    agent = ReinforceAgent(_StubEnv(), _build_config(mock_config))

    returns = agent._calculate_returns([1.0, 2.0, 3.0])

    assert returns == [2.75, 3.5, 3.0]


def test_rollout_episode_uses_action_mask_when_enabled(mock_config):
    """Verify rollout passes the current action mask into policy sampling."""
    config = _build_config(mock_config)
    config.training.ENABLE_ACTION_MASKING = True
    agent = ReinforceAgent(_StubEnv(), config)
    seen = {}

    def _sample_action(state_tensor, action_mask=None):
        seen["mask"] = action_mask
        return 0, torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

    agent.policy_network.sample_action = _sample_action
    episode = agent._rollout_episode()

    assert seen["mask"].tolist() == [True, False, True, True] and episode["total_reward"] == 2.5


def test_rollout_episode_skips_action_mask_when_disabled(mock_config):
    """Verify rollout omits action masks when masking is disabled."""
    config = _build_config(mock_config)
    config.training.ENABLE_ACTION_MASKING = False
    agent = ReinforceAgent(_StubEnv(), config)
    seen = {}

    def _sample_action(state_tensor, action_mask=None):
        seen["mask"] = action_mask
        return 0, torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)

    agent.policy_network.sample_action = _sample_action
    agent._rollout_episode()

    assert seen["mask"] is None


def test_update_networks_returns_zero_explained_variance_for_constant_returns(mock_config):
    """Verify explained variance falls back to zero when returns have no variance."""
    agent = ReinforceAgent(_StubEnv(), _build_config(mock_config))
    batch_data = {
        "returns": [1.0, 1.0],
        "states": [torch.zeros(2), torch.ones(2)],
        "log_probs": [torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)],
        "entropies": [torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)],
    }

    loss, explained_variance = agent._update_networks(batch_data)

    assert isinstance(loss, float) and explained_variance == 0.0


def test_save_checkpoint_writes_only_metrics_when_run_directory_is_missing(mock_config):
    """Verify checkpoint save still persists metrics when no run directory is supplied."""
    metrics_logger = MagicMock()
    checkpoint_manager = MagicMock()
    agent = ReinforceAgent(
        _StubEnv(),
        _build_config(mock_config),
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
    )

    agent.save_checkpoint(None, "/tmp/logs", 3, [1.0], [0.5])

    assert checkpoint_manager.save_checkpoint.called is False and metrics_logger.write_rewards.called


def test_save_checkpoint_writes_only_checkpoint_when_logs_directory_is_missing(mock_config):
    """Verify checkpoint save can skip metrics persistence."""
    metrics_logger = MagicMock()
    checkpoint_manager = MagicMock()
    agent = ReinforceAgent(
        _StubEnv(),
        _build_config(mock_config),
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
    )

    agent.save_checkpoint("/tmp/run", None, 3, [1.0], [0.5])

    assert checkpoint_manager.save_checkpoint.called and metrics_logger.write_rewards.called is False


def test_train_saves_checkpoints_on_interval_and_in_finally(mock_config):
    """Verify training performs interval saves and a final save on exit."""
    metrics_logger = MagicMock()
    checkpoint_manager = MagicMock()
    agent = ReinforceAgent(
        _StubEnv(),
        _build_config(mock_config),
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
    )

    with patch.object(agent, "_rollout_episode", return_value={"states": [], "log_probs": [], "entropies": [], "rewards": [1.0], "total_reward": 1.0, "actual_points": 10.0}):
        with patch.object(agent, "_update_networks", return_value=(2.0, 0.5)):
            with patch.object(agent, "save_checkpoint") as mock_save_checkpoint:
                with patch("draft_buddy.rl.reinforce_agent.signal.signal"):
                    rewards, losses = agent.train(start_episode=1, run_version_dir="/tmp/run", logs_dir="/tmp/logs")

    assert rewards == [1.0, 1.0] and losses[-1] == 2.0 and mock_save_checkpoint.call_count >= 3


def test_train_stops_after_first_episode_when_signal_handler_sets_stop_flag(mock_config):
    """Verify the stop flag breaks out of the training loop early."""
    metrics_logger = MagicMock()
    checkpoint_manager = MagicMock()
    agent = ReinforceAgent(
        _StubEnv(),
        _build_config(mock_config),
        metrics_logger=metrics_logger,
        checkpoint_manager=checkpoint_manager,
    )
    handlers = []

    def _record_signal(sig, handler):
        handlers.append(handler)

    def _rollout_once():
        handlers[0](None, None)
        return {"states": [], "log_probs": [], "entropies": [], "rewards": [1.0], "total_reward": 1.0, "actual_points": 10.0}

    with patch.object(agent, "_rollout_episode", side_effect=_rollout_once):
        with patch.object(agent, "_update_networks", return_value=(2.0, 0.5)):
            with patch("draft_buddy.rl.reinforce_agent.signal.signal", side_effect=_record_signal):
                rewards, _ = agent.train(start_episode=1, run_version_dir=None, logs_dir=None)

    assert rewards == [1.0]
