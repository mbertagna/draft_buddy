"""
Checkpoint manager for reinforcement learning models.

Responsible for saving and loading model weights, value network, and optimizer state.
Isolates disk operations from the agent for testability.
"""

import os
from typing import Optional

import torch


class CheckpointManager:
    """
    Handles saving and loading of model checkpoints.
    """

    def __init__(
        self,
        policy_network: torch.nn.Module,
        value_network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize with the networks and optimizer to manage.

        Parameters
        ----------
        policy_network : torch.nn.Module
            The policy network to save/load.
        value_network : torch.nn.Module
            The value network to save/load.
        optimizer : torch.optim.Optimizer
            The optimizer to save/load.
        """
        self._policy_network = policy_network
        self._value_network = value_network
        self._optimizer = optimizer

    def save_checkpoint(
        self,
        run_version_dir: Optional[str],
        episode: int,
    ) -> None:
        """
        Saves policy, value network, and optimizer to the run directory.

        Parameters
        ----------
        run_version_dir : str, optional
            Directory for checkpoints. If None, nothing is saved.
        episode : int
            Episode number for the checkpoint filename.
        """
        if not run_version_dir:
            return
        os.makedirs(run_version_dir, exist_ok=True)
        policy_path = os.path.join(run_version_dir, f"checkpoint_episode_{episode}.pth")
        value_path = os.path.join(run_version_dir, f"value_episode_{episode}.pth")
        optimizer_path = os.path.join(run_version_dir, f"optimizer_episode_{episode}.pt")
        torch.save(self._policy_network.state_dict(), policy_path)
        torch.save(self._value_network.state_dict(), value_path)
        torch.save(self._optimizer.state_dict(), optimizer_path)

    def load_policy(self, filepath: str) -> None:
        """
        Loads policy network state from file.

        Parameters
        ----------
        filepath : str
            Path to the .pth file.
        """
        self._policy_network.load_state_dict(
            torch.load(filepath, map_location=torch.device("cpu"))
        )
        self._policy_network.eval()

    def load_value_network(self, filepath: str) -> None:
        """
        Loads value network state from file.

        Parameters
        ----------
        filepath : str
            Path to the .pth file.
        """
        self._value_network.load_state_dict(
            torch.load(filepath, map_location=torch.device("cpu"))
        )
        self._value_network.eval()

    def load_optimizer(self, filepath: str) -> None:
        """
        Loads optimizer state from file.

        Parameters
        ----------
        filepath : str
            Path to the .pt file.
        """
        state = torch.load(filepath, map_location=torch.device("cpu"))
        self._optimizer.load_state_dict(state)
