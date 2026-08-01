"""Tests for configuration behavior."""

from __future__ import annotations

from pathlib import Path
import os

from draft_buddy.config import Config, DataConfig, PathsConfig, repository_root


def test_config_to_dict_contains_expected_sections() -> None:
    """Verify config serialization includes all top-level sections."""
    payload = Config().to_dict()

    assert set(payload) == {
        "league",
        "paths",
        "draft",
        "data",
        "training",
        "reward",
        "opponent",
        "scoring",
        "season",
    }


def test_data_config_legacy_stats_start_year_uses_lookback_window() -> None:
    """Verify legacy stats start year is draft year minus lookback seasons."""
    data_config = DataConfig(LEGACY_STATS_LOOKBACK_SEASONS=2)

    assert data_config.legacy_stats_start_year(2026) == 2024


def test_paths_config_defaults_to_repository_root() -> None:
    """Verify derived paths are rooted at the repository, not ``src/``."""
    paths = PathsConfig()

    assert Path(paths.BASE_DIR).name != "src"
    assert paths.MODELS_DIR == os.path.join(paths.BASE_DIR, "models")
    assert paths.LOGS_DIR == os.path.join(paths.BASE_DIR, "logs")
    assert paths.DATA_DIR == os.path.join(paths.BASE_DIR, "data")
    assert paths.DRAFT_STATE_PREV_FILE == os.path.join(paths.DATA_DIR, "draft_state.prev.json")
    assert paths.SAVED_STATES_DIR == os.path.join(paths.BASE_DIR, "saved_states")


def test_paths_config_post_init_creates_directories(tmp_path: Path) -> None:
    """Verify path initialization creates derived directories."""
    paths = PathsConfig()
    paths.BASE_DIR = str(tmp_path)
    paths.__post_init__()

    assert Path(paths.DATA_DIR).is_dir() and Path(paths.MODELS_DIR).is_dir() and Path(paths.LOGS_DIR).is_dir()
    assert Path(paths.SAVED_STATES_DIR).is_dir()
