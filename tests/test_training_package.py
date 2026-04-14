"""Tests for the training run helpers package (metrics, run directories)."""


def test_metrics_logger_class_is_defined_in_training_metrics_logger_module():
    """Training metrics I/O lives under draft_buddy.training.metrics_logger."""
    from draft_buddy.training.metrics_logger import MetricsLogger

    assert MetricsLogger.__module__ == "draft_buddy.training.metrics_logger"


def test_get_run_name_is_defined_in_training_run_utils_module():
    """Run directory helpers live under draft_buddy.training.run_utils."""
    from draft_buddy.training.run_utils import get_run_name

    assert get_run_name.__module__ == "draft_buddy.training.run_utils"
