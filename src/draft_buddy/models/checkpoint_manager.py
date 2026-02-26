import os
from typing import Optional
import torch

class CheckpointManager:
    """
    Handles saving, loading, and validation of model checkpoints.
    Supports inference-only mode with value_network=None and optimizer=None.
    """
    def __init__(
        self,
        policy_network: torch.nn.Module,
        value_network: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self._policy_network = policy_network
        self._value_network = value_network
        self._optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_config(self, loaded_config: dict, current_config: "Config", is_training: bool):
        """Validates the loaded config against the current config, raising ValueError on mismatch."""
        
        # Critical check: feature list must be identical
        saved_features = loaded_config.get('training', {}).get('ENABLED_STATE_FEATURES', [])
        current_features = current_config.training.ENABLED_STATE_FEATURES
        if set(saved_features) != set(current_features):
            raise ValueError(
                f"Configuration mismatch: ENABLED_STATE_FEATURES are different. "
                f"Checkpoint has {len(saved_features)} features, current config has {len(current_features)}."
            )

        # Soft check for inference, critical for training
        saved_teams = loaded_config.get("draft", {}).get("NUM_TEAMS", 0)
        current_teams = current_config.draft.NUM_TEAMS
        if saved_teams != current_teams:
            if is_training:
                raise ValueError(
                    f"Configuration mismatch: NUM_TEAMS is different. "
                    f"Checkpoint trained on {saved_teams}, current config has {current_teams}."
                )
            else:
                print(f"Warning: NUM_TEAMS mismatch. Model trained for {saved_teams}, but draft is for {current_teams}. "
                      "AI results may be suboptimal.")

        # Critical checks for training only
        if is_training:
            mismatched_params = []
            for param in ['LEARNING_RATE', 'DISCOUNT_FACTOR', 'HIDDEN_DIM']:
                saved_val = loaded_config.get('training', {}).get(param)
                current_val = getattr(current_config.training, param)
                if saved_val is not None and saved_val != current_val:
                    mismatched_params.append(f"{param}: Checkpoint={saved_val}, Current={current_val}")
            
            if mismatched_params:
                details = ", ".join(mismatched_params)
                raise ValueError(f"Configuration mismatch: Key training hyperparameters differ. Details: {details}")

    def save_checkpoint(self, run_version_dir: str, episode: int, config: "Config") -> str:
        """Saves a consolidated checkpoint file."""
        os.makedirs(run_version_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_version_dir, f"checkpoint_episode_{episode}.pth")
        
        checkpoint_data = {
            "episode": episode,
            "policy_state_dict": self._policy_network.state_dict(),
            "value_state_dict": self._value_network.state_dict() if self._value_network is not None else None,
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer is not None else None,
            "config": config.to_dict(),
        }
            
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, filepath: str, config: "Config", is_training: bool = False) -> int:
        """
        Loads, validates, and applies a checkpoint.

        Returns
        -------
        int
            The episode number from the checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if not isinstance(checkpoint, dict) or "policy_state_dict" not in checkpoint:
            # Fallback for old style checkpoints
            if is_training:
                raise ValueError("Old-style checkpoint format is not supported for resumed training.")
            print(f"Warning: Loading older model format from {filepath} without config validation.")
            self._policy_network.load_state_dict(checkpoint)
            self._policy_network.eval()
            return checkpoint.get('episode', 0)

        # New style checkpoint validation and loading
        if 'config' in checkpoint:
            self._validate_config(checkpoint['config'], config, is_training)
        else:
            if is_training:
                raise ValueError("Cannot resume training without configuration in checkpoint.")
            print(f"Warning: Checkpoint {filepath} has no embedded config. Cannot validate.")

        self._policy_network.load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint and self._value_network:
            self._value_network.load_state_dict(checkpoint["value_state_dict"])
        if "optimizer_state_dict" in checkpoint and self._optimizer and is_training:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        self._policy_network.eval()
        if self._value_network:
            self._value_network.eval()
            
        return checkpoint.get("episode", 0)
