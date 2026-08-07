import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Neural network that represents the agent's policy.
    Takes the environment state as input and outputs action probabilities.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initializes the policy network.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input state.
        output_dim : int
            Dimensionality of the action space.
        hidden_dim : int, optional
            Number of neurons in hidden layers.
        """
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        state : torch.Tensor
            Current state observation.

        Returns
        -------
        torch.Tensor
            Action logits.
        """
        return self.network(state)

    def get_action_probabilities(
        self,
        state: torch.Tensor,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute masked action probabilities.

        Parameters
        ----------
        state : torch.Tensor
            Current state observation.
        action_mask : Optional[np.ndarray], optional
            Boolean validity mask per action.
        temperature : float, optional
            Softmax temperature. Values above ``1.0`` flatten the
            distribution toward uniform while preserving relative ranking;
            ``1.0`` (default) reproduces the model's raw output unchanged.

        Returns
        -------
        torch.Tensor
            Action probabilities.

        Raises
        ------
        ValueError
            If ``temperature`` is not strictly positive.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        logits = self.forward(state) / temperature
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e9).to(logits.device))
        return F.softmax(logits, dim=-1)

    def sample_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.

        Parameters
        ----------
        state : torch.Tensor
            Current state observation.
        action_mask : Optional[np.ndarray], optional
            Boolean validity mask per action.
        temperature : float, optional
            Softmax temperature applied before sampling. See
            ``get_action_probabilities`` for semantics.

        Returns
        -------
        tuple[int, torch.Tensor, torch.Tensor]
            Action, log probability, and entropy.
        """
        state = state.float().unsqueeze(0)
        action_probs = self.get_action_probabilities(state, action_mask, temperature=temperature)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action), distribution.entropy()
