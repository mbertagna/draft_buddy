"""Tests for configuration serialization and path setup."""

from pathlib import Path

from draft_buddy.config import Config, PathsConfig


def test_config_to_dict_contains_expected_top_level_sections():
    """Verify config serialization includes each config subsection."""
    payload = Config().to_dict()

    assert set(payload) == {"paths", "draft", "training", "reward", "opponent"}


def test_config_save_and_from_file_round_trip(tmp_path):
    """Verify configs can be saved and loaded from disk."""
    config = Config()
    config.draft.NUM_TEAMS = 8
    target = Path(tmp_path) / "config.json"

    config.save(str(target))
    loaded = Config.from_file(str(target))

    assert loaded.draft.NUM_TEAMS == 8


def test_config_from_dict_updates_only_known_sections():
    """Verify unknown sections are ignored during deserialization."""
    config = Config.from_dict({"draft": {"NUM_TEAMS": 6}, "missing": {"value": 1}})

    assert config.draft.NUM_TEAMS == 6 and not hasattr(config, "missing")


def test_paths_config_post_init_creates_expected_directories(tmp_path):
    """Verify path config derives directories from the patched base dir."""
    paths = PathsConfig()
    paths.BASE_DIR = str(tmp_path)
    paths.__post_init__()

    assert Path(paths.DATA_DIR).is_dir() and Path(paths.MODELS_DIR).is_dir() and Path(paths.LOGS_DIR).is_dir()
