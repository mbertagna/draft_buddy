"""Tests for RL helper modules and lazy exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import draft_buddy.rl as rl_package
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.rl.checkpoint_manager import CheckpointManager
from draft_buddy.rl.metrics_logger import MetricsLogger
from draft_buddy.rl.policy_network import PolicyNetwork
from draft_buddy.rl.run_utils import (
    find_latest_checkpoint,
    get_next_version,
    get_run_name,
    save_run_metadata,
    setup_run_directories,
)
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

    with pytest.raises(ValueError, match="ENABLED_STATE_FEATURES"):
        manager._validate_config(loaded_config, tiny_training_config, is_training=False)


def test_checkpoint_manager_warns_on_team_mismatch_in_inference_mode(
    tiny_training_config, capsys
) -> None:
    """Verify NUM_TEAMS mismatch only warns when not resuming training."""
    manager = CheckpointManager(torch.nn.Linear(1, 1))
    loaded_config = {
        "training": {"ENABLED_STATE_FEATURES": tiny_training_config.training.ENABLED_STATE_FEATURES},
        "draft": {"NUM_TEAMS": tiny_training_config.draft.NUM_TEAMS + 1},
    }

    manager._validate_config(loaded_config, tiny_training_config, is_training=False)

    assert "NUM_TEAMS mismatch" in capsys.readouterr().out


def test_checkpoint_manager_rejects_training_hyperparameter_mismatch(
    tiny_training_config,
) -> None:
    """Verify resumed training rejects mismatched key hyperparameters."""
    manager = CheckpointManager(torch.nn.Linear(1, 1))
    loaded_config = {
        "training": {
            "ENABLED_STATE_FEATURES": tiny_training_config.training.ENABLED_STATE_FEATURES,
            "LEARNING_RATE": tiny_training_config.training.LEARNING_RATE * 2,
            "DISCOUNT_FACTOR": tiny_training_config.training.DISCOUNT_FACTOR,
            "HIDDEN_DIM": tiny_training_config.training.HIDDEN_DIM,
        },
        "draft": {"NUM_TEAMS": tiny_training_config.draft.NUM_TEAMS},
    }

    with pytest.raises(ValueError, match="Key training hyperparameters differ"):
        manager._validate_config(loaded_config, tiny_training_config, is_training=True)


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


def test_checkpoint_manager_rejects_old_style_checkpoint_for_training(
    tmp_path: Path, tiny_training_config
) -> None:
    """Verify resumed training rejects old-style checkpoints."""
    policy = torch.nn.Linear(1, 1)
    manager = CheckpointManager(policy)
    checkpoint_path = tmp_path / "old.pth"
    torch.save(policy.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="Old-style checkpoint format"):
        manager.load_checkpoint(str(checkpoint_path), tiny_training_config, is_training=True)


def test_checkpoint_manager_loads_full_checkpoint_into_value_net_and_optimizer(
    tmp_path: Path, tiny_training_config
) -> None:
    """Verify full checkpoints restore policy, value, optimizer, and episode."""
    policy = torch.nn.Linear(1, 1)
    value = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(
        list(policy.parameters()) + list(value.parameters()),
        lr=tiny_training_config.training.LEARNING_RATE,
    )
    manager = CheckpointManager(policy, value, optimizer, torch.device("cpu"))
    checkpoint_path = manager.save_checkpoint(str(tmp_path), 5, tiny_training_config)
    reloaded_policy = torch.nn.Linear(1, 1)
    reloaded_value = torch.nn.Linear(1, 1)
    reloaded_optimizer = torch.optim.SGD(
        list(reloaded_policy.parameters()) + list(reloaded_value.parameters()),
        lr=tiny_training_config.training.LEARNING_RATE,
    )
    reloaded_manager = CheckpointManager(
        reloaded_policy, reloaded_value, reloaded_optimizer, torch.device("cpu")
    )

    episode = reloaded_manager.load_checkpoint(
        str(checkpoint_path), tiny_training_config, is_training=True
    )

    assert episode == 5 and not reloaded_policy.training and not reloaded_value.training


def test_checkpoint_manager_rejects_training_without_embedded_config(
    tmp_path: Path, tiny_training_config
) -> None:
    """Verify resumed training requires config data in the checkpoint."""
    policy = torch.nn.Linear(1, 1)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"episode": 2, "policy_state_dict": policy.state_dict()}, checkpoint_path)
    manager = CheckpointManager(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="Cannot resume training without configuration"):
        manager.load_checkpoint(str(checkpoint_path), tiny_training_config, is_training=True)


def test_policy_network_masks_invalid_actions() -> None:
    """Verify masked actions receive zero probability mass."""
    network = PolicyNetwork(3, 2, hidden_dim=4)
    probabilities = network.get_action_probabilities(torch.zeros(1, 3), action_mask=np.array([True, False]))

    assert float(probabilities[0, 1]) == 0.0


def test_policy_network_sample_action_returns_valid_choice() -> None:
    """Verify sample_action returns a valid action and scalar tensors."""
    network = PolicyNetwork(3, 2, hidden_dim=4)

    action, log_prob, entropy = network.sample_action(torch.zeros(3), action_mask=np.array([True, False]))

    assert action == 0 and log_prob.ndim == 1 and entropy.ndim == 1


def test_state_normalizer_z_score_returns_centered_vector(config) -> None:
    """Verify z-score normalization returns centered output."""
    config.training.STATE_NORMALIZATION_METHOD = "z_score"
    config.training.ENABLED_STATE_FEATURES = ["a", "b"]
    normalized = StateNormalizer(config).normalize({"a": 1.0, "b": 3.0})

    assert np.isclose(float(normalized.mean()), 0.0)


def test_state_normalizer_min_max_uses_feature_bounds(config) -> None:
    """Verify min-max normalization uses configured feature maxima."""
    config.training.STATE_NORMALIZATION_METHOD = "min_max"
    config.training.ENABLED_STATE_FEATURES = ["qb_available_flag", "current_pick_number"]

    normalized = StateNormalizer(config).normalize({"qb_available_flag": 1.0, "current_pick_number": 4.0})

    assert normalized.tolist()[0] == 1.0 and 0.0 < normalized.tolist()[1] < 1.0


def test_state_normalizer_returns_zero_centered_values_when_std_is_zero(config) -> None:
    """Verify z-score normalization handles constant vectors without division errors."""
    config.training.STATE_NORMALIZATION_METHOD = "z_score"
    config.training.ENABLED_STATE_FEATURES = ["a", "b"]

    normalized = StateNormalizer(config).normalize({"a": 2.0, "b": 2.0})

    assert normalized.tolist() == [0.0, 0.0]


def test_metrics_logger_returns_none_paths_when_disabled() -> None:
    """Verify metrics logger exposes no file paths when logging is disabled."""
    logger = MetricsLogger(None)

    assert logger.get_rewards_path() is None and logger.get_losses_path() is None


def test_metrics_logger_write_losses_persists_values(tmp_path: Path) -> None:
    """Verify policy losses are written through the atomic logger."""
    logger = MetricsLogger(str(tmp_path))
    logger.write_losses([0.5, 1.25])

    assert (tmp_path / "all_policy_losses.csv").read_text(encoding="utf-8").strip().splitlines() == ["0.5", "1.25"]


def test_metrics_logger_atomic_writer_ignores_empty_path(tmp_path: Path) -> None:
    """Verify the atomic writer exits early for an empty target path."""
    logger = MetricsLogger(str(tmp_path))

    logger._write_list_to_csv_atomic("", [1.0])

    assert True


def test_get_next_version_returns_v1_for_missing_directory(tmp_path: Path) -> None:
    """Verify next version starts at v1 when the run directory is absent."""
    assert get_next_version(str(tmp_path / "missing")) == "v1"


def test_get_next_version_increments_highest_existing_version(tmp_path: Path) -> None:
    """Verify next version uses the highest numbered v-directory."""
    (tmp_path / "v1").mkdir()
    (tmp_path / "v3").mkdir()
    (tmp_path / "misc").mkdir()

    assert get_next_version(str(tmp_path)) == "v4"


def test_get_run_name_uses_random_start_flag(config) -> None:
    """Verify run names switch format when random start is enabled."""
    config.draft.RANDOMIZE_AGENT_START_POSITION = True

    assert get_run_name(config) == f"{config.draft.NUM_TEAMS}_teams_random_start"


def test_setup_run_directories_creates_model_and_logs_paths(tiny_training_config) -> None:
    """Verify setup_run_directories creates both model and log folders."""
    run_name, version, run_version_dir, logs_dir = setup_run_directories(tiny_training_config)

    assert run_name and version.startswith("v") and Path(run_version_dir).is_dir() and Path(logs_dir).is_dir()


def test_save_run_metadata_writes_json_file(tmp_path: Path, tiny_training_config) -> None:
    """Verify save_run_metadata persists metadata.json."""
    save_run_metadata(tiny_training_config, "run", "v1", str(tmp_path))
    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert payload["run_name"] == "run"


def test_save_run_metadata_falls_back_to_class_attributes(tmp_path: Path) -> None:
    """Verify metadata serialization handles config objects without to_dict."""

    class BareConfig:
        ANSWER = 42
        ENABLED = True

    save_run_metadata(BareConfig(), "run", "v1", str(tmp_path))
    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert payload["config"] == {"ANSWER": 42, "ENABLED": True}


def test_find_latest_checkpoint_returns_none_when_no_episode_match(tiny_training_config) -> None:
    """Verify checkpoint discovery ignores unrelated filenames."""
    run_name = f"{tiny_training_config.draft.NUM_TEAMS}_teams_pos_{tiny_training_config.draft.AGENT_START_POSITION}"
    run_dir = Path(tiny_training_config.paths.MODELS_DIR) / run_name / "v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "not_a_checkpoint.txt").write_text("x", encoding="utf-8")

    assert find_latest_checkpoint(tiny_training_config) is None


def test_find_latest_checkpoint_returns_highest_episode_number(
    tiny_training_config, capsys
) -> None:
    """Verify latest-checkpoint discovery prefers the highest episode number."""
    run_name = f"{tiny_training_config.draft.NUM_TEAMS}_teams_pos_{tiny_training_config.draft.AGENT_START_POSITION}"
    run_dir = Path(tiny_training_config.paths.MODELS_DIR) / run_name / "v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_a = run_dir / "checkpoint_episode_3.pth"
    checkpoint_b = run_dir / "checkpoint_episode_12.pth"
    checkpoint_a.write_text("a", encoding="utf-8")
    checkpoint_b.write_text("b", encoding="utf-8")

    latest = find_latest_checkpoint(tiny_training_config)

    assert latest == str(checkpoint_b) and "episode 12" in capsys.readouterr().out


def test_find_latest_checkpoint_ignores_malformed_checkpoint_names(tiny_training_config) -> None:
    """Verify malformed checkpoint filenames do not prevent valid discovery."""
    run_name = get_run_name(tiny_training_config)
    run_dir = Path(tiny_training_config.paths.MODELS_DIR) / run_name / "v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint_episode_bad.pth").write_text("x", encoding="utf-8")
    valid = run_dir / "checkpoint_episode_2.pth"
    valid.write_text("x", encoding="utf-8")

    assert find_latest_checkpoint(tiny_training_config) == str(valid)
