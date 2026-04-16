"""Tests for RL helper modules and lazy exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import draft_buddy.rl as rl_package
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.rl.checkpoint_manager import CheckpointManager
from draft_buddy.rl.metrics_logger import MetricsLogger
from draft_buddy.rl.policy_network import PolicyNetwork
from draft_buddy.rl.run_utils import find_latest_checkpoint, get_next_version, save_run_metadata, setup_run_directories
from draft_buddy.rl.state_normalizer import StateNormalizer


def test_rl_package_exposes_checkpoint_manager_lazily() -> None:
    """Verify checkpoint manager is exposed through the package root."""
    from draft_buddy.rl.checkpoint_manager import CheckpointManager as CheckpointManagerType

    assert rl_package.CheckpointManager is CheckpointManagerType


def test_agent_model_bot_returns_none_without_callbacks(player_catalog) -> None:
    """Verify agent bot exits early when required callbacks are missing."""
    bot = AgentModelBotGM(model=object(), action_to_position={0: "QB"})

    assert bot.execute_pick(1, {1}, player_catalog, None, {}, {}, lambda *_args: True, lambda *_args: (True, None)) is None


def test_agent_model_bot_falls_back_to_best_adp_on_invalid_model_choice(player_catalog) -> None:
    """Verify agent bot falls back to the best ADP eligible player."""

    class FakeModel:
        def get_action_probabilities(self, *_args, **_kwargs):
            return torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    bot = AgentModelBotGM(FakeModel(), {0: "QB", 1: "RB", 2: "WR", 3: "TE"})
    player = bot.execute_pick(
        1,
        {1, 5},
        player_catalog,
        None,
        {},
        {},
        lambda *_args: True,
        lambda *_args: (False, None),
        build_state_fn=lambda _team_id: np.array([1.0], dtype=np.float32),
        get_action_mask_fn=lambda _team_id: np.array([True, True, True, True]),
    )

    assert player.player_id == 1


def test_checkpoint_manager_validates_feature_mismatch(tiny_training_config) -> None:
    """Verify differing ENABLED_STATE_FEATURES raise a validation error."""
    manager = CheckpointManager(torch.nn.Linear(1, 1))
    loaded_config = {"training": {"ENABLED_STATE_FEATURES": ["a"]}, "draft": {"NUM_TEAMS": tiny_training_config.draft.NUM_TEAMS}}

    import pytest

    with pytest.raises(ValueError, match="ENABLED_STATE_FEATURES"):
        manager._validate_config(loaded_config, tiny_training_config, is_training=False)


def test_checkpoint_manager_saves_checkpoint_with_config(tmp_path: Path, tiny_training_config) -> None:
    """Verify saved checkpoints include the serialized config payload."""
    policy = torch.nn.Linear(1, 1)
    manager = CheckpointManager(policy)
    checkpoint_path = manager.save_checkpoint(str(tmp_path), 3, tiny_training_config)
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert payload["episode"] == 3 and "config" in payload


def test_checkpoint_manager_loads_old_style_checkpoint_for_inference(tmp_path: Path, tiny_training_config) -> None:
    """Verify old-style checkpoints are accepted in inference mode."""
    policy = torch.nn.Linear(1, 1)
    manager = CheckpointManager(policy)
    checkpoint_path = tmp_path / "old.pth"
    torch.save(policy.state_dict(), checkpoint_path)

    assert manager.load_checkpoint(str(checkpoint_path), tiny_training_config, is_training=False) == 0


def test_policy_network_masks_invalid_actions() -> None:
    """Verify masked actions receive zero probability mass."""
    network = PolicyNetwork(3, 2, hidden_dim=4)
    probabilities = network.get_action_probabilities(torch.zeros(1, 3), action_mask=np.array([True, False]))

    assert float(probabilities[0, 1]) == 0.0


def test_state_normalizer_z_score_returns_centered_vector(config) -> None:
    """Verify z-score normalization returns centered output."""
    config.training.STATE_NORMALIZATION_METHOD = "z_score"
    config.training.ENABLED_STATE_FEATURES = ["a", "b"]
    normalized = StateNormalizer(config).normalize({"a": 1.0, "b": 3.0})

    assert np.isclose(float(normalized.mean()), 0.0)


def test_metrics_logger_returns_none_paths_when_disabled() -> None:
    """Verify metrics logger exposes no file paths when logging is disabled."""
    logger = MetricsLogger(None)

    assert logger.get_rewards_path() is None and logger.get_losses_path() is None


def test_get_next_version_returns_v1_for_missing_directory(tmp_path: Path) -> None:
    """Verify next version starts at v1 when the run directory is absent."""
    assert get_next_version(str(tmp_path / "missing")) == "v1"


def test_setup_run_directories_creates_model_and_logs_paths(tiny_training_config) -> None:
    """Verify setup_run_directories creates both model and log folders."""
    run_name, version, run_version_dir, logs_dir = setup_run_directories(tiny_training_config)

    assert run_name and version.startswith("v") and Path(run_version_dir).is_dir() and Path(logs_dir).is_dir()


def test_save_run_metadata_writes_json_file(tmp_path: Path, tiny_training_config) -> None:
    """Verify save_run_metadata persists metadata.json."""
    save_run_metadata(tiny_training_config, "run", "v1", str(tmp_path))
    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert payload["run_name"] == "run"


def test_find_latest_checkpoint_returns_none_when_no_episode_match(tiny_training_config) -> None:
    """Verify checkpoint discovery ignores unrelated filenames."""
    run_name = f"{tiny_training_config.draft.NUM_TEAMS}_teams_pos_{tiny_training_config.draft.AGENT_START_POSITION}"
    run_dir = Path(tiny_training_config.paths.MODELS_DIR) / run_name / "v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "not_a_checkpoint.txt").write_text("x", encoding="utf-8")

    assert find_latest_checkpoint(tiny_training_config) is None
