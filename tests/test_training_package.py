"""Tests for the training run helpers package (metrics, run directories)."""


def test_metrics_logger_class_is_defined_in_rl_metrics_logger_module():
    """Training metrics I/O lives under draft_buddy.rl.metrics_logger."""
    from draft_buddy.rl.metrics_logger import MetricsLogger

    assert MetricsLogger.__module__ == "draft_buddy.rl.metrics_logger"


def test_get_run_name_is_defined_in_rl_run_utils_module():
    """Run directory helpers live under draft_buddy.rl.run_utils."""
    from draft_buddy.rl.run_utils import get_run_name

    assert get_run_name.__module__ == "draft_buddy.rl.run_utils"
