import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import os
import signal

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

        # Initialize the Policy Network (actor) and Value Network (baseline)
        self.policy_network = PolicyNetwork(input_dim, output_dim, config.HIDDEN_DIM)
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(config.HIDDEN_DIM, 1),
        )
        # Initialize the optimizer with separate LRs for policy and value
        value_lr_multiplier = getattr(self.config, 'VALUE_LR_MULTIPLIER', 2.0)
        self.optimizer = optim.Adam(
            [
                {"params": self.policy_network.parameters(), "lr": config.LEARNING_RATE},
                {"params": self.value_network.parameters(), "lr": config.LEARNING_RATE * value_lr_multiplier},
            ]
        )

        # Store rewards and log probabilities for each episode
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = [] # Raw rewards received at each step
        self.episode_entropies: List[torch.Tensor] = []
        self.episode_states: List[torch.Tensor] = []

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

    def train(self, start_episode=1, run_version_dir=None, logs_dir=None) -> Tuple[List[float], List[float]]:
        """
        Runs the REINFORCE training loop for a specified number of episodes.

        Args:
            start_episode (int): The episode number to start training from.
            run_version_dir (str): The directory to save model checkpoints.

        Returns:
            Tuple[List[float], List[float]]: Lists of total rewards per episode and policy losses per episode.
        """
        all_episode_rewards = []
        all_policy_losses = []

        # Prepare log file paths
        rewards_data_path = os.path.join(logs_dir, 'all_episode_rewards.csv') if logs_dir else None
        losses_data_path = os.path.join(logs_dir, 'all_policy_losses.csv') if logs_dir else None
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            # Ensure files exist so external tools can see them immediately
            for path in [rewards_data_path, losses_data_path]:
                if path and not os.path.exists(path):
                    with open(path, 'w') as _f:
                        _f.write("")

        def _append_value_to_csv(path: str, value: float):
            if not path:
                return
            try:
                with open(path, 'a') as f:
                    f.write(f"{value}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                # As a fallback, try to rewrite the entire file atomically
                self._write_list_to_csv_atomic(path, all_episode_rewards if 'rewards' in path else all_policy_losses)

        # Flag to support graceful shutdown on signals
        stop_requested = {'value': False}

        def _signal_handler(signum, frame):
            print(f"\nSignal {signum} received. Finishing current episode, saving logs, and exiting gracefully...")
            stop_requested['value'] = True

        # Register signal handlers to make logging robust against Ctrl+C and terminations
        try:
            signal.signal(signal.SIGINT, _signal_handler)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            pass

        print(f"Starting REINFORCE training for {self.config.TOTAL_EPISODES} episodes...")
        print(f"State features enabled: {self.config.ENABLED_STATE_FEATURES}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}, Discount Factor: {self.config.DISCOUNT_FACTOR}")
        print(f"Action Masking Enabled: {self.config.ENABLE_ACTION_MASKING}")
        print(f"Batch episodes per update: {getattr(self.config, 'BATCH_EPISODES', 16)} | Grad clip: {getattr(self.config, 'GRAD_CLIP_NORM', 0.5)} | Value LR x{getattr(self.config, 'VALUE_LR_MULTIPLIER', 2.0)}")

        # Batch accumulators
        batch_states: List[torch.Tensor] = []
        batch_log_probs: List[torch.Tensor] = []
        batch_entropies: List[torch.Tensor] = []
        batch_returns: List[float] = []
        episodes_in_batch = 0

        batch_size = getattr(self.config, 'BATCH_EPISODES', 16)
        grad_clip_norm = getattr(self.config, 'GRAD_CLIP_NORM', 0.5)

        # Track a per-episode loss value to keep losses aligned with episodes
        last_loss_per_episode = float('nan')
        # Track last computed optimization metrics for periodic logging even when no update this episode
        last_total_loss_value = float('nan')
        last_explained_variance = float('nan')

        try:
            for episode in range(start_episode, self.config.TOTAL_EPISODES + 1):
                state, info = self.env.reset()
                current_action_mask = info.get('action_mask')

                self.episode_log_probs = []
                self.episode_rewards = []
                self.episode_entropies = []
                self.episode_states = []
                episode_done = False
                total_episode_reward = 0

                # --- 1. Rollout an Episode ---
                while not episode_done:
                    # Convert numpy state to torch tensor
                    state_tensor = torch.from_numpy(state).float()

                    # Sample an action from the policy, passing the action mask if enabled
                    if self.config.ENABLE_ACTION_MASKING:
                        action, log_prob, entropy = self.policy_network.sample_action(state_tensor, action_mask=current_action_mask)
                    else:
                        action, log_prob, entropy = self.policy_network.sample_action(state_tensor)

                    # Take the action in the environment
                    next_state, reward, done, truncated, info = self.env.step(action)

                    # Store state, log probability, entropy, and reward
                    self.episode_states.append(state_tensor)
                    self.episode_log_probs.append(log_prob)
                    self.episode_entropies.append(entropy)
                    self.episode_rewards.append(reward)
                    total_episode_reward += reward

                    state = next_state
                    episode_done = done or truncated
                    current_action_mask = info.get('action_mask')

                # --- 2. Calculate Returns and stash episode into batch ---
                returns = self._calculate_returns(self.episode_rewards)
                batch_states.extend(self.episode_states)
                batch_log_probs.extend(self.episode_log_probs)
                batch_entropies.extend(self.episode_entropies)
                batch_returns.extend(returns)
                episodes_in_batch += 1

                # --- 3. If batch is ready, perform one optimization step ---
                did_update = False
                episodes_in_batch_before_update = episodes_in_batch
                if episodes_in_batch >= batch_size or episode == self.config.TOTAL_EPISODES:
                    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)
                    states_tensor = torch.stack(batch_states)
                    values = self.value_network(states_tensor).squeeze(-1)
                    advantages = returns_tensor - values.detach()
                    if len(advantages) > 1 and advantages.std() > 1e-6:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Losses
                    policy_terms = []
                    for log_prob, adv in zip(batch_log_probs, advantages):
                        policy_terms.append(-log_prob * adv)
                    policy_loss = torch.stack(policy_terms).sum()

                    entropy_coeff = getattr(self.config, 'ENTROPY_COEFFICIENT', 0.01)
                    entropy_loss = -entropy_coeff * torch.stack(batch_entropies).sum()

                    value_coeff = getattr(self.config, 'VALUE_LOSS_COEFFICIENT', 0.5)
                    value_loss = value_coeff * torch.nn.functional.mse_loss(values, returns_tensor)

                    total_loss = policy_loss + value_loss + entropy_loss

                    # Optimization with gradient clipping
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), grad_clip_norm)
                        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), grad_clip_norm)
                    except Exception:
                        pass
                    self.optimizer.step()

                    # Explained variance of value predictions
                    with torch.no_grad():
                        var_returns = returns_tensor.var(unbiased=False)
                        if var_returns.item() > 1e-8:
                            ev = 1.0 - (returns_tensor - values).var(unbiased=False) / var_returns
                            explained_variance = ev.item()
                        else:
                            explained_variance = 0.0
                    # Store for logging even in future episodes without updates
                    last_total_loss_value = float(total_loss.item())
                    last_explained_variance = float(explained_variance)

                    # Reset batch
                    batch_states.clear()
                    batch_log_probs.clear()
                    batch_entropies.clear()
                    batch_returns.clear()
                    episodes_in_batch = 0
                    did_update = True

                # --- 4. Logging and Reporting ---
                all_episode_rewards.append(total_episode_reward)
                _append_value_to_csv(rewards_data_path, total_episode_reward)

                # Keep losses list aligned per episode using last known per-episode loss
                all_policy_losses.append(last_loss_per_episode)
                # Persist the placeholder for this episode's loss (may be corrected below)
                if losses_data_path is not None:
                    try:
                        with open(losses_data_path, 'a') as f:
                            f.write(f"{last_loss_per_episode}\n")
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception:
                        pass

                # If we updated this episode, distribute the batch loss across the last N episodes
                if did_update:
                    num_episodes_to_update = max(episodes_in_batch_before_update, 1)
                    loss_per_episode_value = (total_loss.item() / float(num_episodes_to_update))
                    last_loss_per_episode = loss_per_episode_value
                    # Overwrite last N entries with the new value
                    all_policy_losses[-num_episodes_to_update:] = [loss_per_episode_value] * num_episodes_to_update
                    # Snapshot the losses CSV atomically so values are corrected
                    try:
                        self._write_list_to_csv_atomic(losses_data_path, all_policy_losses)
                    except Exception:
                        pass

                # Calculate actual (unweighted) final projected points for logging clarity
                actual_final_projected_points = sum(
                    p.projected_points for p in self.env.teams_rosters[self.env.agent_team_id]['PLAYERS']
                )

                if episode % 1000 == 0:
                    self.save_checkpoint(run_version_dir, logs_dir, episode, all_episode_rewards, all_policy_losses)

                # Print periodic progress regardless of whether an update happened this episode.
                # Uses last known loss/EV from the most recent optimization step.
                if episode % 100 == 0:
                    loss_for_log = last_total_loss_value
                    ev_for_log = last_explained_variance
                    print(
                        f"Episode {episode}/{self.config.TOTAL_EPISODES} | "
                        f"Total Reward (Weighted): {total_episode_reward:.2f} | "
                        f"Loss (total): {loss_for_log if not np.isnan(loss_for_log) else float('nan'):.4f} | "
                        f"EV(Value): {ev_for_log if not np.isnan(ev_for_log) else float('nan'):.3f} | "
                        f"Agent Roster Size: {len(self.env.teams_rosters[self.env.agent_team_id]['PLAYERS'])} "
                        f"| Actual Final Score (Unweighted): {actual_final_projected_points:.2f}"
                    )

                # If a stop was requested (e.g., Ctrl+C), finish after saving this episode
                if stop_requested['value']:
                    try:
                        self._write_list_to_csv_atomic(rewards_data_path, all_episode_rewards)
                        self._write_list_to_csv_atomic(losses_data_path, all_policy_losses)
                    except Exception:
                        pass
                    try:
                        self.save_checkpoint(run_version_dir, logs_dir, episode, all_episode_rewards, all_policy_losses)
                    except Exception:
                        pass
                    break

            print("\nTraining complete!")
        except KeyboardInterrupt:
            # Graceful shutdown: ensure logs and a checkpoint are saved
            print("\nKeyboardInterrupt detected. Saving progress...")
        finally:
            try:
                # Always snapshot logs atomically at the end
                if rewards_data_path:
                    self._write_list_to_csv_atomic(rewards_data_path, all_episode_rewards)
                if losses_data_path:
                    self._write_list_to_csv_atomic(losses_data_path, all_policy_losses)
            except Exception:
                pass
            try:
                # Save a final checkpoint with current progress
                current_episode = len(all_episode_rewards) if all_episode_rewards else start_episode
                self.save_checkpoint(run_version_dir, logs_dir, current_episode, all_episode_rewards, all_policy_losses)
            except Exception:
                pass
        return all_episode_rewards, all_policy_losses

    def save_model(self, filepath: str):
        """
        Saves the policy network's state dictionary to a file.

        Args:
            filepath (str): The path to save the model file.
        """
        torch.save(self.policy_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Loads the policy network's state dictionary from a file.

        Args:
            filepath (str): The path to the model file.
        """
        self.policy_network.load_state_dict(torch.load(filepath))
        self.policy_network.eval() # Set the network to evaluation mode
        print(f"Model loaded from {filepath}")

    def save_value_network(self, filepath: str):
        """Saves the value network state to a file (kept separate from the policy model)."""
        torch.save(self.value_network.state_dict(), filepath)
        print(f"Value network saved to {filepath}")

    def load_value_network(self, filepath: str):
        """Loads the value network state from a file, if available."""
        self.value_network.load_state_dict(torch.load(filepath))
        self.value_network.eval()
        print(f"Value network loaded from {filepath}")

    def save_optimizer(self, filepath: str):
        """Saves the optimizer state dict to a file separate from the model file."""
        torch.save(self.optimizer.state_dict(), filepath)
        print(f"Optimizer state saved to {filepath}")

    def load_optimizer(self, filepath: str):
        """Loads the optimizer state dict from file, if available."""
        state = torch.load(filepath)
        self.optimizer.load_state_dict(state)
        print(f"Optimizer state loaded from {filepath}")

    def save_checkpoint(self, run_version_dir, logs_dir, episode, all_episode_rewards, all_policy_losses):
        """Saves training artifacts and ensures training data are persisted atomically.

        Artifacts saved:
        - Policy network (model) to checkpoint_episode_{episode}.pth (unchanged for inference users)
        - Value network to value_episode_{episode}.pth
        - Optimizer to optimizer_episode_{episode}.pt
        - Rewards/Losses CSVs written atomically
        """
        if run_version_dir:
            checkpoint_path = os.path.join(run_version_dir, f'checkpoint_episode_{episode}.pth')
            self.save_model(checkpoint_path)
            # Save additional training states for smooth resume
            value_path = os.path.join(run_version_dir, f'value_episode_{episode}.pth')
            self.save_value_network(value_path)
            optimizer_path = os.path.join(run_version_dir, f'optimizer_episode_{episode}.pt')
            self.save_optimizer(optimizer_path)

        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            rewards_data_path = os.path.join(logs_dir, 'all_episode_rewards.csv')
            losses_data_path = os.path.join(logs_dir, 'all_policy_losses.csv')
            # Rewrite full CSVs atomically to avoid duplication and partial writes
            self._write_list_to_csv_atomic(rewards_data_path, all_episode_rewards)
            self._write_list_to_csv_atomic(losses_data_path, all_policy_losses)
            print(f"Raw training data saved to {rewards_data_path} and {losses_data_path}")

    def _write_list_to_csv_atomic(self, path: str, values: List[float]):
        """Writes an entire list of floats to a CSV file atomically."""
        if not path:
            return
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, 'w') as f:
                for v in values:
                    f.write(f"{v}\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            # Best-effort cleanup
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
