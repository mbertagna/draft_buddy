import os
import signal
import numpy as np
import torch
import torch.optim as optim
from typing import List, Optional, Tuple, Dict, Any

from draft_buddy.models.checkpoint_manager import CheckpointManager
from draft_buddy.config import Config
from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
from draft_buddy.training.metrics_logger import MetricsLogger
from draft_buddy.models.policy_network import PolicyNetwork


class ReinforceAgent:
    """
    Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.
    """
    def __init__(
        self,
        env: FantasyFootballDraftEnv,
        config: Config,
        metrics_logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ReinforceAgent using device: {self.device}")

        input_dim = len(config.training.ENABLED_STATE_FEATURES)
        output_dim = env.action_space.n

        self.policy_network = PolicyNetwork(input_dim, output_dim, config.training.HIDDEN_DIM).to(self.device)
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.training.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(config.training.HIDDEN_DIM, 1),
        ).to(self.device)
        
        value_lr_multiplier = getattr(self.config.training, "VALUE_LR_MULTIPLIER", 2.0)
        self.optimizer = optim.Adam(
            [
                {"params": self.policy_network.parameters(), "lr": config.training.LEARNING_RATE},
                {"params": self.value_network.parameters(), "lr": config.training.LEARNING_RATE * value_lr_multiplier},
            ]
        )

        self._metrics_logger = metrics_logger
        self._checkpoint_manager = checkpoint_manager or CheckpointManager(
            self.policy_network, self.value_network, self.optimizer, self.device
        )

    def _calculate_returns(self, rewards: List[float]) -> List[float]:
        """Calculates discounted returns (G_t) for an episode."""
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.config.training.DISCOUNT_FACTOR * G
            returns.insert(0, G)
        return returns

    def _rollout_episode(self) -> Dict[str, Any]:
        """Executes one episode rollout and returns the data."""
        state, info = self.env.reset()
        current_action_mask = info.get('action_mask')

        episode_data = {
            'states': [],
            'log_probs': [],
            'entropies': [],
            'rewards': [],
            'total_reward': 0.0,
            'actual_points': 0.0
        }
        
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            
            if self.config.training.ENABLE_ACTION_MASKING:
                action, log_prob, entropy = self.policy_network.sample_action(state_tensor, action_mask=current_action_mask)
            else:
                action, log_prob, entropy = self.policy_network.sample_action(state_tensor)

            next_state, reward, terminated, truncated, info = self.env.step(action)

            episode_data['states'].append(state_tensor)
            episode_data['log_probs'].append(log_prob)
            episode_data['entropies'].append(entropy)
            episode_data['rewards'].append(reward)
            episode_data['total_reward'] += reward

            state = next_state
            done = terminated or truncated
            current_action_mask = info.get('action_mask')

        # Final points from agent roster
        agent_roster = self.env.teams_rosters[self.env.agent_team_id]['PLAYERS']
        episode_data['actual_points'] = sum(p.projected_points for p in agent_roster)
        
        return episode_data

    def _update_networks(self, batch_data: Dict[str, List]) -> Tuple[float, float]:
        """Performs one optimization step using batch data."""
        returns_tensor = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)
        states_tensor = torch.stack(batch_data['states']).to(self.device)
        
        # 1. Compute Value Loss and Advantages
        values = self.value_network(states_tensor).squeeze(-1)
        advantages = returns_tensor - values.detach()
        if len(advantages) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Policy Loss (REINFORCE with Baseline)
        policy_terms = [-log_prob * adv for log_prob, adv in zip(batch_data['log_probs'], advantages)]
        policy_loss = torch.stack(policy_terms).sum()

        # 3. Entropy Loss (to encourage exploration)
        entropy_coeff = getattr(self.config.training, 'ENTROPY_COEFFICIENT', 0.01)
        entropy_loss = -entropy_coeff * torch.stack(batch_data['entropies']).sum()

        # 4. Value Network Loss (MSE)
        value_coeff = getattr(self.config.training, 'VALUE_LOSS_COEFFICIENT', 0.5)
        value_loss = value_coeff * torch.nn.functional.mse_loss(values, returns_tensor)

        total_loss = policy_loss + value_loss + entropy_loss

        # 5. Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_clip = getattr(self.config.training, 'GRAD_CLIP_NORM', 0.5)
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), grad_clip)
        self.optimizer.step()

        # 6. Explained Variance
        var_returns = returns_tensor.var(unbiased=False).item()
        ev = 1.0 - (returns_tensor - values).var(unbiased=False).item() / var_returns if var_returns > 1e-8 else 0.0
        
        return float(total_loss.item()), float(ev)

    def train(self, start_episode=1, run_version_dir=None, logs_dir=None) -> Tuple[List[float], List[float]]:
        """Main training loop."""
        all_episode_rewards, all_policy_losses, all_actual_points = [], [], []
        metrics_logger = self._metrics_logger or MetricsLogger(logs_dir)
        interval = self.config.training.LOG_SAVE_INTERVAL_EPISODES
        batch_size = self.config.training.BATCH_EPISODES

        stop_requested = {'value': False}
        def _handler(sig, frame): stop_requested['value'] = True
        for s in (signal.SIGINT, signal.SIGTERM): 
            try: signal.signal(s, _handler) 
            except Exception: pass

        batch_data = {'states': [], 'log_probs': [], 'entropies': [], 'returns': []}
        episodes_in_batch = 0
        last_loss_per_episode = float('nan')
        last_metrics = {'loss': float('nan'), 'ev': float('nan')}

        print(f"Starting training for {self.config.training.TOTAL_EPISODES} episodes...")

        try:
            for ep in range(start_episode, self.config.training.TOTAL_EPISODES + 1):
                # 1. Rollout
                ep_data = self._rollout_episode()
                returns = self._calculate_returns(ep_data['rewards'])
                
                # 2. Accumulate Batch
                batch_data['states'].extend(ep_data['states'])
                batch_data['log_probs'].extend(ep_data['log_probs'])
                batch_data['entropies'].extend(ep_data['entropies'])
                batch_data['returns'].extend(returns)
                episodes_in_batch += 1
                
                all_episode_rewards.append(ep_data['total_reward'])
                all_actual_points.append(ep_data['actual_points'])
                all_policy_losses.append(last_loss_per_episode)

                # 3. Update
                if episodes_in_batch >= batch_size or ep == self.config.training.TOTAL_EPISODES:
                    loss_val, ev_val = self._update_networks(batch_data)
                    last_metrics.update({'loss': loss_val, 'ev': ev_val})
                    
                    # Distribute batch loss
                    avg_loss = loss_val / max(episodes_in_batch, 1)
                    last_loss_per_episode = avg_loss
                    all_policy_losses[-episodes_in_batch:] = [avg_loss] * episodes_in_batch
                    
                    # Reset Batch
                    batch_data = {k: [] for k in batch_data}
                    episodes_in_batch = 0

                # 4. Periodic Logging & Saving
                if ep % interval == 0:
                    self.save_checkpoint(run_version_dir, logs_dir, ep, all_episode_rewards, all_policy_losses)
                    self._log_progress(ep, interval, all_episode_rewards, all_actual_points, last_metrics)

                if stop_requested['value']: break

        finally:
            metrics_logger.write_rewards(all_episode_rewards)
            metrics_logger.write_losses(all_policy_losses)
            self.save_checkpoint(run_version_dir, logs_dir, ep if 'ep' in locals() else start_episode, all_episode_rewards, all_policy_losses)

        return all_episode_rewards, all_policy_losses

    def _log_progress(self, ep, interval, rewards, points, last_metrics):
        """Helper to print training progress."""
        avg_rew = np.mean(rewards[-interval:])
        avg_pts = np.mean(points[-interval:])
        print(f"Episode {ep:<10} | avg_rew={avg_rew:<10.2f} | avg_pts={avg_pts:<10.2f} | loss={last_metrics['loss']:<10.3f} | EV={last_metrics['ev']:<10.3f}")

    def load_checkpoint(self, filepath: str, is_training: bool) -> int:
        """Loads a consolidated checkpoint using the checkpoint manager."""
        return self._checkpoint_manager.load_checkpoint(filepath, self.config, is_training)

    def save_checkpoint(self, run_version_dir, logs_dir, episode, all_episode_rewards, all_policy_losses):
        """Saves a consolidated checkpoint and training metrics."""
        if run_version_dir:
            self._checkpoint_manager.save_checkpoint(run_version_dir, episode, self.config)
        if logs_dir:
            metrics = self._metrics_logger or MetricsLogger(logs_dir)
            metrics.write_rewards(all_episode_rewards)
            metrics.write_losses(all_policy_losses)
