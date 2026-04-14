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
        self, state: torch.Tensor, action_mask: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Compute masked action probabilities.

        Parameters
        ----------
        state : torch.Tensor
            Current state observation.
        action_mask : Optional[np.ndarray], optional
            Boolean validity mask per action.

        Returns
        -------
        torch.Tensor
            Action probabilities.
        """
        logits = self.forward(state)
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e9).to(logits.device))
        return F.softmax(logits, dim=-1)

    def sample_action(
        self, state: torch.Tensor, action_mask: Optional[np.ndarray] = None
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.

        Parameters
        ----------
        state : torch.Tensor
            Current state observation.
        action_mask : Optional[np.ndarray], optional
            Boolean validity mask per action.

        Returns
        -------
        tuple[int, torch.Tensor, torch.Tensor]
            Action, log probability, and entropy.
        """
        state = state.float().unsqueeze(0)
        action_probs = self.get_action_probabilities(state, action_mask)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action), distribution.entropy()
