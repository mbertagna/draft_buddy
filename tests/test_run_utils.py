"""Tests for metrics logging utilities."""

from __future__ import annotations

from pathlib import Path

from draft_buddy.rl.metrics_logger import MetricsLogger


def test_metrics_logger_write_rewards_persists_values(tmp_path: Path) -> None:
    """Verify reward logs are written atomically."""
    logger = MetricsLogger(str(tmp_path))
    logger.write_rewards([1.0, 2.5, 3.75])
    rewards_path = Path(logger.get_rewards_path())

    assert rewards_path.read_text(encoding="utf-8").strip().splitlines() == ["1.0", "2.5", "3.75"]
