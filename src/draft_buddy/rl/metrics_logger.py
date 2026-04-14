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
        """Create empty log files when absent."""
        for path in [self._rewards_path, self._losses_path]:
            if path and not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as file_obj:
                    file_obj.write("")

    def write_rewards(self, values: List[float]) -> None:
        """
        Write rewards list to CSV atomically.

        Parameters
        ----------
        values : List[float]
            Episode rewards to persist.
        """
        if self._rewards_path:
            self._write_list_to_csv_atomic(self._rewards_path, values)

    def write_losses(self, values: List[float]) -> None:
        """
        Write losses list to CSV atomically.

        Parameters
        ----------
        values : List[float]
            Policy losses to persist.
        """
        if self._losses_path:
            self._write_list_to_csv_atomic(self._losses_path, values)

    def _write_list_to_csv_atomic(self, path: str, values: List[float]) -> None:
        """
        Write values to CSV atomically.

        Parameters
        ----------
        path : str
            Target file path.
        values : List[float]
            Values to write one per line.
        """
        if not path:
            return
        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as file_obj:
                for value in values:
                    file_obj.write(f"{value}\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, path)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def get_rewards_path(self) -> Optional[str]:
        """Return rewards file path or None when disabled."""
        return self._rewards_path

    def get_losses_path(self) -> Optional[str]:
        """Return losses file path or None when disabled."""
        return self._losses_path
