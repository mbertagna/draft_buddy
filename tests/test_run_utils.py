"""Tests for metrics logging and run utility helpers."""

from pathlib import Path

from draft_buddy.rl.metrics_logger import MetricsLogger


def test_metrics_logger_write_rewards_persists_values_and_cleans_tmp_file(tmp_path):
    """Verify write_rewards writes CSV and leaves no temporary artifact."""
    logger = MetricsLogger(str(tmp_path))
    logger.write_rewards([1.0, 2.5, 3.75])
    rewards_path = Path(logger.get_rewards_path())
    tmp_path_file = Path(f"{logger.get_rewards_path()}.tmp")
    content = rewards_path.read_text(encoding="utf-8").strip().splitlines()

    assert content == ["1.0", "2.5", "3.75"] and tmp_path_file.exists() is False
