"""
Metrics logger for reinforcement learning training.

Handles disk I/O for rewards and losses with atomic CSV writes.
Isolates file operations from the agent for testability.
"""

import os
from typing import List, Optional


class MetricsLogger:
    """
    Handles atomic CSV writes for training metrics (rewards, losses).
    """

    def __init__(self, logs_dir: Optional[str] = None):
        """
        Initialize the metrics logger.

        Parameters
        ----------
        logs_dir : str, optional
            Directory for log files. If None, logging is disabled.
        """
        self._logs_dir = logs_dir
        self._rewards_path: Optional[str] = None
        self._losses_path: Optional[str] = None
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            self._rewards_path = os.path.join(logs_dir, "all_episode_rewards.csv")
            self._losses_path = os.path.join(logs_dir, "all_policy_losses.csv")
            self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        """Creates empty log files if they do not exist."""
        for path in [self._rewards_path, self._losses_path]:
            if path and not os.path.exists(path):
                with open(path, "w") as _f:
                    _f.write("")

    def write_rewards(self, values: List[float]) -> None:
        """
        Writes the rewards list to CSV atomically.

        Parameters
        ----------
        values : List[float]
            List of episode rewards to persist.
        """
        if self._rewards_path:
            self._write_list_to_csv_atomic(self._rewards_path, values)

    def write_losses(self, values: List[float]) -> None:
        """
        Writes the losses list to CSV atomically.

        Parameters
        ----------
        values : List[float]
            List of policy losses to persist.
        """
        if self._losses_path:
            self._write_list_to_csv_atomic(self._losses_path, values)

    def _write_list_to_csv_atomic(self, path: str, values: List[float]) -> None:
        """
        Writes a list of floats to a file atomically using a temp file and replace.

        Parameters
        ----------
        path : str
            Target file path.
        values : List[float]
            Values to write (one per line).
        """
        if not path:
            return
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                for v in values:
                    f.write(f"{v}\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_rewards_path(self) -> Optional[str]:
        """Returns the rewards file path, or None if logging disabled."""
        return self._rewards_path

    def get_losses_path(self) -> Optional[str]:
        """Returns the losses file path, or None if logging disabled."""
        return self._losses_path
