import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional
import numpy as np # Import numpy for mask type

class PolicyNetwork(nn.Module):
    """
    Neural network that represents the agent's policy.
    Takes the environment state as input and outputs action probabilities.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initializes the Policy Network.

        Args:
            input_dim (int): Dimensionality of the input state.
            output_dim (int): Dimensionality of the output action space (number of possible actions).
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(PolicyNetwork, self).__init__()

        # Define the layers of the neural network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The current state observation from the environment.

        Returns:
            torch.Tensor: The raw output (logits) for each action.
        """
        return self.network(state)

    def get_action_probabilities(self, state: torch.Tensor, action_mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Computes the action probabilities from the network's output,
        optionally applying an action mask.

        Args:
            state (torch.Tensor): The current state observation.
            action_mask (np.ndarray, optional): A boolean array indicating valid actions.
                                                True for valid, False for invalid.
                                                Defaults to None (no masking).

        Returns:
            torch.Tensor: A tensor of probabilities for each action.
        """
        logits = self.forward(state)

        if action_mask is not None:
            # Convert mask to torch tensor, ensure it's on the same device as logits
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            # Set logits of invalid actions to a very small number (negative infinity)
            # This makes their probabilities effectively zero after softmax.
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e9).to(logits.device))

        # Apply softmax to convert logits into probabilities
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def sample_action(self, state: torch.Tensor, action_mask: Optional[np.ndarray] = None) -> tuple[int, torch.Tensor]:
        """
        Samples an action from the policy distribution and returns its log probability.
        Optionally applies action masking before sampling.

        Args:
            state (torch.Tensor): The current state observation.
            action_mask (np.ndarray, optional): A boolean array indicating valid actions.
                                                True for valid, False for invalid.
                                                Defaults to None (no masking).

        Returns:
            tuple[int, torch.Tensor]: A tuple containing the sampled action (int)
                                      and its log probability (torch.Tensor).
        """
        # Ensure state is a float tensor and has a batch dimension (even if batch size is 1)
        state = state.float().unsqueeze(0) # Add batch dimension

        # Get action probabilities, applying the mask if provided
        action_probs = self.get_action_probabilities(state, action_mask)

        # Create a categorical distribution from the action probabilities
        m = Categorical(action_probs)

        # Sample an action from the distribution
        action = m.sample()

        # Get the log probability of the sampled action
        log_prob = m.log_prob(action)

        return action.item(), log_prob