"""Tests for configuration behavior."""

from __future__ import annotations

from pathlib import Path

from draft_buddy.config import Config, PathsConfig


def test_config_to_dict_contains_expected_sections() -> None:
    """Verify config serialization includes all top-level sections."""
    payload = Config().to_dict()

    assert set(payload) == {"paths", "draft", "training", "reward", "opponent"}


def test_paths_config_post_init_creates_directories(tmp_path: Path) -> None:
    """Verify path initialization creates derived directories."""
    paths = PathsConfig()
    paths.BASE_DIR = str(tmp_path)
    paths.__post_init__()

    assert Path(paths.DATA_DIR).is_dir() and Path(paths.MODELS_DIR).is_dir() and Path(paths.LOGS_DIR).is_dir()
