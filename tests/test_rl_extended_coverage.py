"""Additional RL and draft-env coverage tests."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from draft_buddy.config import Config
from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
from draft_buddy.rl.checkpoint_manager import CheckpointManager
from draft_buddy.rl.metrics_logger import MetricsLogger
from draft_buddy.rl.policy_network import PolicyNetwork
from draft_buddy.rl.reinforce_agent import ReinforceAgent
from draft_buddy.rl.run_utils import (
    find_latest_checkpoint,
    get_next_version,
    get_run_name,
    save_run_metadata,
    setup_run_directories,
)
import draft_buddy.rl as rl_pkg
from draft_buddy.draft_env.state_normalizer import StateNormalizer
from draft_buddy.rl.reward_calculator import RewardCalculator
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.domain.entities import Player


def test_policy_network_get_action_probabilities_respects_masked_actions():
    """Verify masked actions receive near-zero probability."""
    model = PolicyNetwork(input_dim=4, output_dim=4, hidden_dim=8)
    state = torch.zeros((1, 4), dtype=torch.float32)
    probs = model.get_action_probabilities(state, action_mask=np.array([True, False, True, True]))

    assert float(probs[0, 1]) < 1e-6


def test_policy_network_sample_action_returns_int_action():
    """Verify sampled action is returned as integer."""
    model = PolicyNetwork(input_dim=4, output_dim=4, hidden_dim=8)
    action, _, _ = model.sample_action(torch.zeros(4, dtype=torch.float32), np.array([True, True, True, True]))

    assert isinstance(action, int)


def test_metrics_logger_write_losses_persists_loss_values(tmp_path):
    """Verify write_losses writes provided values to file."""
    logger = MetricsLogger(str(tmp_path))
    logger.write_losses([0.1, 0.2])
    content = open(logger.get_losses_path(), "r", encoding="utf-8").read().strip().splitlines()

    assert content == ["0.1", "0.2"]


def test_metrics_logger_atomic_writer_returns_early_for_empty_path():
    """Verify atomic writer exits when no output path is provided."""
    logger = MetricsLogger(None)
    logger._write_list_to_csv_atomic("", [1.0])

    assert True


def test_get_run_name_uses_random_start_suffix_when_enabled():
    """Verify run name format for randomized start position."""
    cfg = Config()
    cfg.RANDOMIZE_AGENT_START_POSITION = True

    assert get_run_name(cfg) == f"{cfg.draft.NUM_TEAMS}_teams_random_start"


def test_get_next_version_returns_v1_for_missing_run_dir(tmp_path):
    """Verify versioning starts at v1 when directory is absent."""
    missing = tmp_path / "missing"

    assert get_next_version(str(missing)) == "v1"


def test_get_next_version_increments_from_existing_version_directories(tmp_path):
    """Verify version helper increments from highest existing vN directory."""
    run_dir = tmp_path / "run"
    (run_dir / "v1").mkdir(parents=True, exist_ok=True)
    (run_dir / "v2").mkdir(parents=True, exist_ok=True)

    assert get_next_version(str(run_dir)) == "v3"


def test_get_next_version_returns_v1_when_no_version_dirs_exist(tmp_path):
    """Verify non-version directories do not affect version seed."""
    run_dir = tmp_path / "run2"
    (run_dir / "misc").mkdir(parents=True, exist_ok=True)

    assert get_next_version(str(run_dir)) == "v1"


def test_setup_run_directories_creates_versioned_paths(tmp_path):
    """Verify setup creates run and logs directories."""
    cfg = Config()
    cfg.paths.MODELS_DIR = str(tmp_path / "models")
    cfg.paths.LOGS_DIR = str(tmp_path / "logs")
    _, _, run_dir, logs_dir = setup_run_directories(cfg)

    assert (run_dir and logs_dir and (tmp_path / "models").exists() and (tmp_path / "logs").exists())


def test_save_run_metadata_writes_metadata_file(tmp_path):
    """Verify metadata JSON is written to run version directory."""
    cfg = Config()
    run_dir = str(tmp_path / "run")
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, "test_run", "v1", run_dir)

    assert (tmp_path / "run" / "metadata.json").exists()


def test_find_latest_checkpoint_returns_none_when_no_run_directory(tmp_path):
    """Verify latest checkpoint finder returns None without base run dir."""
    cfg = Config()
    cfg.paths.MODELS_DIR = str(tmp_path / "models")

    assert find_latest_checkpoint(cfg) is None


def test_find_latest_checkpoint_returns_none_when_no_checkpoint_files(tmp_path):
    """Verify checkpoint finder returns None when no files match pattern."""
    cfg = Config()
    cfg.paths.MODELS_DIR = str(tmp_path / "models")
    run_name = f"{cfg.draft.NUM_TEAMS}_teams_pos_{cfg.draft.AGENT_START_POSITION}"
    (tmp_path / "models" / run_name / "v1").mkdir(parents=True, exist_ok=True)

    assert find_latest_checkpoint(cfg) is None


def test_find_latest_checkpoint_picks_highest_episode_file(tmp_path):
    """Verify highest episode checkpoint path is selected."""
    cfg = Config()
    cfg.paths.MODELS_DIR = str(tmp_path / "models")
    run_name = f"{cfg.draft.NUM_TEAMS}_teams_pos_{cfg.draft.AGENT_START_POSITION}"
    base = tmp_path / "models" / run_name / "v1"
    base.mkdir(parents=True, exist_ok=True)
    (base / "checkpoint_episode_1.pth").write_text("x", encoding="utf-8")
    (base / "checkpoint_episode_3.pth").write_text("x", encoding="utf-8")

    assert str(find_latest_checkpoint(cfg)).endswith("checkpoint_episode_3.pth")


def test_rl_package_lazy_exports_policy_network_symbol():
    """Verify rl package lazily resolves PolicyNetwork symbol."""
    assert rl_pkg.PolicyNetwork.__name__ == "PolicyNetwork"


def test_rl_package_raises_attribute_error_for_unknown_symbol():
    """Verify unknown lazy attribute names raise AttributeError."""
    raised = False
    try:
        _ = rl_pkg.UNKNOWN_SYMBOL
    except AttributeError:
        raised = True

    assert raised is True


def test_checkpoint_manager_validate_config_raises_on_feature_mismatch():
    """Verify config validation fails when feature sets differ."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    cfg = Config()
    bad_loaded = {"training": {"ENABLED_STATE_FEATURES": ["x"]}, "draft": {"NUM_TEAMS": cfg.draft.NUM_TEAMS}}

    try:
        manager._validate_config(bad_loaded, cfg, True)
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_checkpoint_manager_validate_config_warn_path_allows_team_mismatch_for_inference():
    """Verify team mismatch is tolerated in inference validation mode."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    cfg = Config()
    loaded = {
        "training": {"ENABLED_STATE_FEATURES": cfg.training.ENABLED_STATE_FEATURES},
        "draft": {"NUM_TEAMS": cfg.draft.NUM_TEAMS + 1},
    }
    manager._validate_config(loaded, cfg, False)

    assert True


def test_checkpoint_manager_validate_config_raises_on_hparam_mismatch_in_training():
    """Verify training validation fails when key hparams differ."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    cfg = Config()
    loaded = {
        "training": {
            "ENABLED_STATE_FEATURES": cfg.training.ENABLED_STATE_FEATURES,
            "LEARNING_RATE": cfg.training.LEARNING_RATE + 1.0,
            "DISCOUNT_FACTOR": cfg.training.DISCOUNT_FACTOR,
            "HIDDEN_DIM": cfg.training.HIDDEN_DIM,
        },
        "draft": {"NUM_TEAMS": cfg.draft.NUM_TEAMS},
    }
    raised = False
    try:
        manager._validate_config(loaded, cfg, True)
    except ValueError:
        raised = True

    assert raised is True


def test_checkpoint_manager_save_checkpoint_creates_pth_file(tmp_path):
    """Verify checkpoint save writes checkpoint file."""
    net = PolicyNetwork(4, 4, 8)
    val = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(val.parameters()), lr=0.001)
    manager = CheckpointManager(net, val, opt)
    cfg = Config()
    path = manager.save_checkpoint(str(tmp_path), 7, cfg)

    assert path.endswith("checkpoint_episode_7.pth")


def test_checkpoint_manager_load_checkpoint_restores_episode(tmp_path):
    """Verify load_checkpoint returns stored episode number."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    cfg = Config()
    save_path = manager.save_checkpoint(str(tmp_path), 11, cfg)
    episode = manager.load_checkpoint(save_path, cfg, is_training=False)

    assert episode == 11


def test_checkpoint_manager_load_checkpoint_old_format_path_raises_for_training(tmp_path):
    """Verify old-format checkpoints are rejected for resumed training."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    old_path = str(tmp_path / "old.pth")
    torch.save(net.state_dict(), old_path)
    raised = False
    try:
        manager.load_checkpoint(old_path, Config(), is_training=True)
    except ValueError:
        raised = True

    assert raised is True


def test_checkpoint_manager_load_checkpoint_old_format_supported_for_inference(tmp_path):
    """Verify old-format checkpoints load in inference mode."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    old_path = str(tmp_path / "old_infer.pth")
    torch.save({"weight": torch.tensor([1.0])}, old_path)
    with patch.object(net, "load_state_dict", return_value=None):
        episode = manager.load_checkpoint(old_path, Config(), is_training=False)

    assert episode == 0


def test_checkpoint_manager_load_checkpoint_raises_when_training_without_embedded_config(tmp_path):
    """Verify training load fails when checkpoint lacks embedded config."""
    net = PolicyNetwork(4, 4, 8)
    manager = CheckpointManager(net)
    path = str(tmp_path / "no_cfg.pth")
    torch.save({"policy_state_dict": net.state_dict(), "episode": 1}, path)
    raised = False
    try:
        manager.load_checkpoint(path, Config(), is_training=True)
    except ValueError:
        raised = True

    assert raised is True


def test_reinforce_agent_calculate_returns_applies_discount(mock_config):
    """Verify discounted returns are computed from reward list."""
    mock_config.training.DISCOUNT_FACTOR = 0.5
    env = Mock()
    env.action_space.n = 4
    agent = ReinforceAgent(env, mock_config, metrics_logger=Mock(), checkpoint_manager=Mock())
    returns = agent._calculate_returns([2.0, 2.0])

    assert returns == [3.0, 2.0]


def test_reinforce_agent_save_checkpoint_calls_manager_and_metrics(mock_config):
    """Verify save_checkpoint delegates to manager and metrics logger."""
    env = Mock()
    env.action_space.n = 4
    metrics = Mock()
    ckpt = Mock()
    agent = ReinforceAgent(env, mock_config, metrics_logger=metrics, checkpoint_manager=ckpt)
    agent.save_checkpoint("/tmp/run", "/tmp/logs", 5, [1.0], [0.5])

    assert ckpt.save_checkpoint.called


def test_reinforce_agent_load_checkpoint_delegates_to_checkpoint_manager(mock_config):
    """Verify load_checkpoint delegates and returns manager value."""
    env = Mock()
    env.action_space.n = 4
    ckpt = Mock()
    ckpt.load_checkpoint.return_value = 3
    agent = ReinforceAgent(env, mock_config, metrics_logger=Mock(), checkpoint_manager=ckpt)
    episode = agent.load_checkpoint("checkpoint.pth", True)

    assert episode == 3


def test_state_normalizer_z_score_handles_zero_standard_deviation():
    """Verify z-score normalization handles zero-variance vectors."""
    cfg = Config()
    normalizer = StateNormalizer(cfg)
    values = normalizer._normalize_z_score(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    assert np.allclose(values, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_state_normalizer_z_score_normalizes_non_constant_vector():
    """Verify z-score normalization returns centered values."""
    cfg = Config()
    normalizer = StateNormalizer(cfg)
    values = normalizer._normalize_z_score(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert round(float(values.mean()), 7) == 0.0


def test_state_normalizer_fallback_returns_raw_state_for_unknown_method():
    """Verify unknown normalization method returns raw state array."""
    cfg = Config()
    cfg.training.ENABLED_STATE_FEATURES = ["current_pick_number"]
    cfg.training.STATE_NORMALIZATION_METHOD = "none"
    normalizer = StateNormalizer(cfg)
    output = normalizer.normalize({"current_pick_number": 5.0})

    assert output[0] == 5.0


def test_state_normalizer_normalize_z_score_branch_executes():
    """Verify normalize dispatches to z-score mode."""
    cfg = Config()
    cfg.training.ENABLED_STATE_FEATURES = ["current_pick_number"]
    cfg.training.STATE_NORMALIZATION_METHOD = "z_score"
    normalizer = StateNormalizer(cfg)
    output = normalizer.normalize({"current_pick_number": 5.0})

    assert float(output[0]) == 0.0


def test_state_normalizer_min_max_handles_equal_bounds_as_zero(monkeypatch):
    """Verify min-max path returns zero when max equals min."""
    cfg = Config()
    cfg.training.ENABLED_STATE_FEATURES = ["custom_feature"]
    normalizer = StateNormalizer(cfg)
    normalizer._feature_max_bounds["custom_feature"] = 0.0
    output = normalizer.normalize({"custom_feature": 0.0})

    assert output[0] == 0.0


def test_env_save_and_load_state_round_trip(mock_config, tmp_path):
    """Verify environment can persist and reload state file."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    state_file = str(tmp_path / "state.json")
    env.save_state(state_file)
    env.load_state(state_file)

    assert env.current_pick_number >= 1


def test_env_generate_snake_draft_order_handles_no_players(mock_config):
    """Verify snake order generation returns empty when no players available."""
    env = FantasyFootballDraftEnv(mock_config)
    env.all_players_data = []
    draft_order = env._generate_snake_draft_order(4, 10)

    assert draft_order == []


def test_env_get_info_contains_expected_keys(mock_config):
    """Verify info payload includes pick and pool context keys."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    info = env._get_info()

    assert "current_pick_number" in info


def test_env_get_best_available_player_by_pos_returns_player(mock_config):
    """Verify best available helper returns a player object."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    best_qb = env._get_best_available_player_by_pos("QB")

    assert best_qb is not None


def test_env_try_select_player_returns_false_when_position_unavailable(mock_config):
    """Verify select helper fails when no players exist for chosen position."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    ok, _ = env._try_select_player_for_team(env.agent_team_id, "K", env.available_players_ids)

    assert ok is False


def test_agent_model_bot_returns_none_when_state_callbacks_missing():
    """Verify AgentModelBotGM returns None without state/mask callbacks."""
    bot = AgentModelBotGM(model=Mock(), action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"})
    picked = bot.execute_pick(
        team_id=1,
        available_player_ids=set(),
        player_map={},
        roster_counts={},
        roster_structure={},
        bench_maxes={},
        can_draft_position_fn=lambda *args, **kwargs: True,
        try_select_player_fn=lambda *args, **kwargs: (False, None),
        build_state_fn=None,
        get_action_mask_fn=None,
    )

    assert picked is None


def test_agent_model_bot_falls_back_to_lowest_adp_when_model_pick_invalid():
    """Verify AgentModelBotGM falls back to best ADP eligible player."""
    class _MockModel:
        def get_action_probabilities(self, state_tensor, action_mask=None):
            del state_tensor, action_mask
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    qb = Player(1, "QB", "QB", 200.0, adp=5.0)
    rb = Player(2, "RB", "RB", 150.0, adp=3.0)
    bot = AgentModelBotGM(_MockModel(), {0: "QB", 1: "RB", 2: "WR", 3: "TE"})
    picked = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map={1: qb, 2: rb},
        roster_counts={},
        roster_structure={},
        bench_maxes={},
        can_draft_position_fn=lambda team_id, position, is_manual=False: True,
        try_select_player_fn=lambda team_id, position, available_ids: (False, None),
        build_state_fn=lambda team_id: np.zeros(4, dtype=np.float32),
        get_action_mask_fn=lambda team_id: np.array([True, True, True, True]),
    )

    assert picked.player_id == 2


def test_reward_calculator_step_reward_includes_pick_shaping_and_vorp_components():
    """Verify step reward includes pick-shaping delta and VORP shaping."""
    config = Config()
    config.reward.ENABLE_INTERMEDIATE_REWARD = False
    config.reward.ENABLE_PICK_SHAPING_REWARD = True
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = True
    config.reward.PICK_SHAPING_STARTER_DELTA_WEIGHT = 0.5
    config.reward.ENABLE_VORP_PICK_SHAPING = True
    config.reward.VORP_PICK_SHAPING_WEIGHT = 0.1
    config.reward.ENABLE_STACKING_REWARD = False
    env = SimpleNamespace(
        teams_rosters={1: {"PLAYERS": [Player(1, "QB", "QB", 200.0)]}},
        agent_team_id=1,
        _calculate_vorp=lambda position: 10.0,
    )
    drafted = Player(2, "RB", "RB", 150.0)
    reward, _ = RewardCalculator.calculate_step_reward(config, env, drafted, prev_starter_points=150.0)

    assert reward > 0.0


def test_reward_calculator_final_reward_handles_season_sim_exception_gracefully():
    """Verify final reward calculation tolerates simulation failures."""
    config = Config()
    config.reward.ENABLE_FINAL_BASE_REWARD = False
    config.reward.ENABLE_COMPETITIVE_REWARD = False
    config.reward.ENABLE_SEASON_SIM_REWARD = True
    env = SimpleNamespace(
        agent_team_id=1,
        teams_rosters={1: {"PLAYERS": [Player(1, "QB", "QB", 100.0)]}},
        total_roster_size_per_team=10,
        weekly_projections={},
    )
    matchups = pd.DataFrame([{"Week": 1, "Away Manager(s)": "A", "Home Manager(s)": "B"}])
    with patch("draft_buddy.rl.reward_calculator.simulate_season_fast", side_effect=RuntimeError("boom")):
        reward, _ = RewardCalculator.calculate_final_reward(config, env, matchups)

    assert isinstance(reward, float)


def test_reward_calculator_step_reward_static_mode_uses_constant_value():
    """Verify static intermediate reward mode returns configured constant."""
    config = Config()
    config.reward.ENABLE_INTERMEDIATE_REWARD = True
    config.reward.INTERMEDIATE_REWARD_MODE = "STATIC"
    config.reward.INTERMEDIATE_REWARD_VALUE = 7.5
    config.reward.ENABLE_PICK_SHAPING_REWARD = False
    config.reward.ENABLE_VORP_PICK_SHAPING = False
    config.reward.ENABLE_STACKING_REWARD = False
    env = SimpleNamespace(teams_rosters={1: {"PLAYERS": []}}, agent_team_id=1)
    reward, _ = RewardCalculator.calculate_step_reward(config, env, Player(1, "X", "WR", 100.0), 0.0)

    assert reward == 7.5


def test_reward_calculator_final_reward_adds_full_roster_bonus_when_roster_is_full():
    """Verify final reward includes full roster bonus when eligible."""
    config = Config()
    config.reward.ENABLE_FINAL_BASE_REWARD = False
    config.reward.ENABLE_COMPETITIVE_REWARD = False
    config.reward.ENABLE_SEASON_SIM_REWARD = False
    config.reward.BONUS_FOR_FULL_ROSTER = 4.0
    env = SimpleNamespace(
        agent_team_id=1,
        teams_rosters={1: {"PLAYERS": [Player(1, "QB", "QB", 100.0)]}},
        total_roster_size_per_team=1,
        weekly_projections={},
    )
    reward, _ = RewardCalculator.calculate_final_reward(config, env, pd.DataFrame())

    assert reward == 4.0


def test_reward_calculator_final_reward_avg_opponent_mode_sets_target_score():
    """Verify AVG_OPPONENT_DIFFERENCE mode records average opponent target."""
    config = Config()
    config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = False
    config.reward.ENABLE_FINAL_BASE_REWARD = False
    config.reward.ENABLE_COMPETITIVE_REWARD = True
    config.reward.COMPETITIVE_REWARD_MODE = "AVG_OPPONENT_DIFFERENCE"
    config.reward.ENABLE_SEASON_SIM_REWARD = False
    env = SimpleNamespace(
        agent_team_id=1,
        teams_rosters={
            1: {"PLAYERS": [Player(1, "A", "QB", 100.0)]},
            2: {"PLAYERS": [Player(2, "B", "QB", 50.0)]},
            3: {"PLAYERS": [Player(3, "C", "QB", 70.0)]},
        },
        total_roster_size_per_team=10,
        weekly_projections={},
    )
    _, info = RewardCalculator.calculate_final_reward(config, env, pd.DataFrame())

    assert "target_opponent_score" in info


def test_env_init_uses_random_schedule_when_flag_enabled(mock_config):
    """Verify environment builds round-robin schedule when random matchups enabled."""
    mock_config.USE_RANDOM_MATCHUPS = True
    mock_config.NUM_REGULAR_SEASON_WEEKS = 2
    env = FantasyFootballDraftEnv(mock_config)

    assert isinstance(env.matchups_df, pd.DataFrame)


def test_env_property_setters_update_underlying_state(mock_config):
    """Verify env property setters write through DraftState delegates."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.current_pick_idx = 2
    env.current_pick_number = 3
    env.agent_team_id = 1
    env._overridden_team_id = 2

    assert (env.current_pick_idx, env.current_pick_number, env.agent_team_id, env._overridden_team_id) == (2, 3, 1, 2)


def test_env_set_current_team_picking_sets_override(mock_config):
    """Verify manual override setter stores team id."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.set_current_team_picking(1)

    assert env._overridden_team_id == 1


def test_env_get_ai_suggestion_returns_error_when_model_missing(mock_config):
    """Verify AI suggestion API reports missing model."""
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    env.agent_model = None
    result = env.get_ai_suggestion()

    assert "error" in result


def test_env_get_draft_summary_reports_total_picks_after_manual_draft(mock_config):
    """Verify draft summary total picks increments after one manual draft."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    pid = next(iter(env.available_players_ids))
    env.draft_player(pid)
    summary = env.get_draft_summary()

    assert summary["total_picks"] == 1


def test_env_undo_last_pick_reverts_pick_number(mock_config):
    """Verify undo restores pick number to previous value."""
    mock_config.draft.AGENT_START_POSITION = 1
    mock_config.draft.RANDOMIZE_AGENT_START_POSITION = False
    env = FantasyFootballDraftEnv(mock_config)
    env.reset()
    pid = next(iter(env.available_players_ids))
    env.draft_player(pid)
    env.undo_last_pick()

    assert env.current_pick_number == 1
