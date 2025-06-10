import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Neural network that represents the agent's policy.
    Takes the environment state as input and outputs action probabilities.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """
        Initializes the Policy Network.

        Args:
            input_dim (int): Dimensionality of the input state.
            output_dim (int): Dimensionality of the output action space (number of possible actions).
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(PolicyNetwork, self).__init__()

        # Define the layers of the neural network
        # Input Layer -> Hidden Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Hidden Layer 2 -> Output Layer (logits for action probabilities)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The current state observation from the environment.

        Returns:
            torch.Tensor: The raw output (logits) for each action.
        """
        # Apply ReLU activation function after each hidden layer
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # The output layer produces logits, which will be converted to probabilities
        # later using softmax. No activation here as softmax is applied externally.
        logits = self.fc3(x)
        return logits

    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the action probabilities from the network's output.

        Args:
            state (torch.Tensor): The current state observation.

        Returns:
            torch.Tensor: A tensor of probabilities for each action.
        """
        logits = self.forward(state)
        # Apply softmax to convert logits into probabilities
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def sample_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Samples an action from the policy distribution and returns its log probability.

        Args:
            state (torch.Tensor): The current state observation.

        Returns:
            tuple[int, torch.Tensor]: A tuple containing the sampled action (int)
                                      and its log probability (torch.Tensor).
        """
        # Ensure state is a float tensor and has a batch dimension (even if batch size is 1)
        state = state.float().unsqueeze(0) # Add batch dimension

        action_probs = self.get_action_probabilities(state)

        # Create a categorical distribution from the action probabilities
        m = Categorical(action_probs)

        # Sample an action from the distribution
        action = m.sample()

        # Get the log probability of the sampled action
        log_prob = m.log_prob(action)

        return action.item(), log_prob