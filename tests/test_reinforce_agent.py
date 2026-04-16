"""Tests for REINFORCE agent behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from draft_buddy.rl.reinforce_agent import ReinforceAgent


class FakeTrainingEnv:
    """Small deterministic env for training-loop tests."""

    def __init__(self):
        self.action_space = SimpleNamespace(n=2)
        self.agent_team_id = 1
        self._reset_calls = 0

    def reset(self):
        self._reset_calls += 1
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), {"action_mask": np.array([True, True])}

    def step(self, action: int):
        _ = action
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), 2.0, True, False, {"action_mask": np.array([True, True])}

    def resolve_roster_players(self, team_id: int):
        _ = team_id
        return [SimpleNamespace(projected_points=12.0)]


class FakeCheckpointManager:
    """Checkpoint manager that records save and load calls."""

    def __init__(self):
        self.saved = []

    def save_checkpoint(self, run_version_dir, episode, config):
        self.saved.append((run_version_dir, episode, config))
        return "saved"

    def load_checkpoint(self, filepath, config, is_training):
        _ = (filepath, config, is_training)
        return 7


def test_calculate_returns_applies_discount_factor(tiny_training_config) -> None:
    """Verify rewards are discounted from the end of the episode."""
    agent = ReinforceAgent(FakeTrainingEnv(), tiny_training_config, checkpoint_manager=FakeCheckpointManager())

    assert agent._calculate_returns([1.0, 1.0]) == [1.9, 1.0]


def test_rollout_episode_collects_reward_and_points(tiny_training_config, monkeypatch) -> None:
    """Verify rollout stores reward totals and final roster points."""
    agent = ReinforceAgent(FakeTrainingEnv(), tiny_training_config, checkpoint_manager=FakeCheckpointManager())
    monkeypatch.setattr(
        agent.policy_network,
        "sample_action",
        lambda state_tensor, action_mask=None: (
            0,
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(0.0, requires_grad=True),
        ),
    )
    episode = agent._rollout_episode()

    assert episode["total_reward"] == 2.0 and episode["actual_points"] == 12.0


def test_load_checkpoint_delegates_to_checkpoint_manager(tiny_training_config) -> None:
    """Verify load_checkpoint forwards to the checkpoint manager."""
    checkpoint_manager = FakeCheckpointManager()
    agent = ReinforceAgent(FakeTrainingEnv(), tiny_training_config, checkpoint_manager=checkpoint_manager)

    assert agent.load_checkpoint("checkpoint.pth", is_training=True) == 7
