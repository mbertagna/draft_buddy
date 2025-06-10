import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from policy_network import PolicyNetwork
from fantasy_draft_env import FantasyFootballDraftEnv # Import our environment
from config import Config # Import our configuration

class ReinforceAgent:
    """
    Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.
    """
    def __init__(self, env: FantasyFootballDraftEnv, config: Config):
        """
        Initializes the REINFORCE Agent.

        Args:
            env (FantasyFootballDraftEnv): The OpenAI Gym environment.
            config (Config): The project configuration object.
        """
        self.env = env
        self.config = config

        # Determine input and output dimensions for the policy network
        input_dim = len(config.ENABLED_STATE_FEATURES)
        output_dim = env.action_space.n # Number of discrete actions (QB, RB, WR, TE)

        # Initialize the Policy Network
        self.policy_network = PolicyNetwork(input_dim, output_dim, config.HIDDEN_DIM)
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config.LEARNING_RATE)

        # Store rewards and log probabilities for each episode
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = [] # Raw rewards received at each step

    def _calculate_returns(self, rewards: List[float]) -> List[float]:
        """
        Calculates the discounted returns (G_t) for an episode.
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^(T-t) * r_T
        """
        returns: List[float] = []
        G = 0.0
        # Iterate backwards through the rewards to calculate discounted sum
        for r in reversed(rewards):
            G = r + self.config.DISCOUNT_FACTOR * G
            returns.insert(0, G) # Insert at the beginning to keep original order
        return returns

    def train(self) -> Tuple[List[float], List[float]]:
        """
        Runs the REINFORCE training loop for a specified number of episodes.

        Returns:
            Tuple[List[float], List[float]]: Lists of total rewards per episode and policy losses per episode.
        """
        all_episode_rewards = []
        all_policy_losses = []

        print(f"Starting REINFORCE training for {self.config.TOTAL_EPISODES} episodes...")
        print(f"State features enabled: {self.config.ENABLED_STATE_FEATURES}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}, Discount Factor: {self.config.DISCOUNT_FACTOR}")

        for episode in range(1, self.config.TOTAL_EPISODES + 1):
            state, info = self.env.reset()
            self.episode_log_probs = []
            self.episode_rewards = []
            episode_done = False
            total_episode_reward = 0

            # --- 1. Rollout an Episode ---
            while not episode_done:
                # Convert numpy state to torch tensor
                state_tensor = torch.from_numpy(state).float()

                # Sample an action from the policy
                action, log_prob = self.policy_network.sample_action(state_tensor)

                # Take the action in the environment
                next_state, reward, done, truncated, info = self.env.step(action)

                # Store log probability and reward
                self.episode_log_probs.append(log_prob)
                self.episode_rewards.append(reward)
                total_episode_reward += reward

                state = next_state
                episode_done = done or truncated # Consider truncated also as episode end

            # --- 2. Calculate Returns ---
            returns = self._calculate_returns(self.episode_rewards)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

            # Normalize returns (standard practice for stability in REINFORCE)
            # Avoid division by zero if all returns are the same
            if len(returns_tensor) > 1 and returns_tensor.std() > 1e-6:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            else:
                # If only one or no returns, or std is zero, no normalization needed or possible
                pass

            # --- 3. Compute Policy Loss ---
            # Loss = - sum(log_prob * G_t)
            policy_loss = []
            for log_prob, G_t in zip(self.episode_log_probs, returns_tensor):
                policy_loss.append(-log_prob * G_t) # Negative sign for gradient ascent (maximize reward)

            # Combine all step losses for the episode
            policy_loss = torch.stack(policy_loss).sum()

            # --- 4. Perform Optimization Step ---
            self.optimizer.zero_grad() # Clear gradients from previous step
            policy_loss.backward()      # Compute gradients
            self.optimizer.step()       # Update network weights

            # --- 5. Logging and Reporting ---
            all_episode_rewards.append(total_episode_reward)
            all_policy_losses.append(policy_loss.item())

            if episode % 100 == 0:
                print(f"Episode {episode}/{self.config.TOTAL_EPISODES} | "
                      f"Total Reward: {total_episode_reward:.2f} | "
                      f"Policy Loss: {policy_loss.item():.4f} | "
                      f"Agent Roster Size: {len(self.env.teams_rosters[self.env.agent_team_id]['PLAYERS'])} "
                      f"(Final Score: {info.get('final_score', 'N/A')})"
                     )
                # self.env.render() # Optional: Render environment state periodically

        print("\nTraining complete!")
        return all_episode_rewards, all_policy_losses