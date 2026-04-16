"""Shared fixtures for canonical Draft Buddy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from draft_buddy.config import Config
from draft_buddy.core import (
    DraftController,
    DraftState,
    FantasyRulesEngine,
    Pick,
    Player,
    PlayerCatalog,
)
from draft_buddy.data import load_player_catalog


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Return a test configuration rooted in a temporary directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary test directory.

    Returns
    -------
    Config
        Configuration with isolated paths and deterministic draft settings.
    """
    config = Config()
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    models_dir.mkdir()
    logs_dir.mkdir()

    config.paths.DATA_DIR = str(data_dir)
    config.paths.MODELS_DIR = str(models_dir)
    config.paths.LOGS_DIR = str(logs_dir)
    config.paths.PLAYER_DATA_CSV = str(data_dir / "generated_player_data.csv")
    config.paths.DRAFT_STATE_FILE = str(data_dir / "draft_state.json")

    config.draft.NUM_TEAMS = 4
    config.draft.AGENT_START_POSITION = 1
    config.draft.RANDOMIZE_AGENT_START_POSITION = False
    config.draft.MANUAL_DRAFT_TEAMS = []
    config.draft.ROSTER_STRUCTURE = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1}
    config.draft.BENCH_MAXES = {"QB": 0, "RB": 1, "WR": 1, "TE": 0}
    config.draft.TOTAL_BENCH_SIZE = 2
    config.draft.TEAM_MANAGER_MAPPING = {1: "Team 1", 2: "Team 2", 3: "Team 3", 4: "Team 4"}
    config.training.MODEL_PATH_TO_LOAD = ""
    config.training.RESUME_TRAINING = False
    config.reward.ENABLE_SEASON_SIM_REWARD = False
    config.reward.ENABLE_COMPETITIVE_REWARD = False
    config.reward.USE_RANDOM_MATCHUPS = False
    return config


@pytest.fixture
def player_dataframe() -> pd.DataFrame:
    """Return a small deterministic player pool.

    Returns
    -------
    pd.DataFrame
        Draftable player records with bye weeks and teams.
    """
    return pd.DataFrame(
        [
            {"player_id": 1, "name": "QB One", "position": "QB", "projected_points": 300.0, "adp": 1.0, "bye_week": 7, "team": "BUF"},
            {"player_id": 2, "name": "RB One", "position": "RB", "projected_points": 240.0, "adp": 2.0, "bye_week": 7, "team": "BUF"},
            {"player_id": 3, "name": "WR One", "position": "WR", "projected_points": 230.0, "adp": 3.0, "bye_week": 7, "team": "BUF"},
            {"player_id": 4, "name": "TE One", "position": "TE", "projected_points": 170.0, "adp": 4.0, "bye_week": 7, "team": "BUF"},
            {"player_id": 5, "name": "QB Two", "position": "QB", "projected_points": 280.0, "adp": 5.0, "bye_week": 10, "team": "KC"},
            {"player_id": 6, "name": "RB Two", "position": "RB", "projected_points": 220.0, "adp": 6.0, "bye_week": 10, "team": "KC"},
            {"player_id": 7, "name": "WR Two", "position": "WR", "projected_points": 210.0, "adp": 7.0, "bye_week": 10, "team": "KC"},
            {"player_id": 8, "name": "TE Two", "position": "TE", "projected_points": 150.0, "adp": 8.0, "bye_week": 10, "team": "KC"},
            {"player_id": 9, "name": "QB Three", "position": "QB", "projected_points": 260.0, "adp": 9.0, "bye_week": 11, "team": "PHI"},
            {"player_id": 10, "name": "RB Three", "position": "RB", "projected_points": 205.0, "adp": 10.0, "bye_week": 11, "team": "PHI"},
            {"player_id": 11, "name": "WR Three", "position": "WR", "projected_points": 200.0, "adp": 11.0, "bye_week": 11, "team": "PHI"},
            {"player_id": 12, "name": "TE Three", "position": "TE", "projected_points": 140.0, "adp": 12.0, "bye_week": 11, "team": "PHI"},
            {"player_id": 13, "name": "QB Four", "position": "QB", "projected_points": 250.0, "adp": 13.0, "bye_week": 14, "team": "DAL"},
            {"player_id": 14, "name": "RB Four", "position": "RB", "projected_points": 190.0, "adp": 14.0, "bye_week": 14, "team": "DAL"},
            {"player_id": 15, "name": "WR Four", "position": "WR", "projected_points": 185.0, "adp": 15.0, "bye_week": 14, "team": "DAL"},
            {"player_id": 16, "name": "TE Four", "position": "TE", "projected_points": 130.0, "adp": 16.0, "bye_week": 14, "team": "DAL"},
        ]
    )


@pytest.fixture
def player_catalog(config: Config, player_dataframe: pd.DataFrame) -> PlayerCatalog:
    """Return a loaded player catalog from the temporary CSV.

    Parameters
    ----------
    config : Config
        Test configuration.
    player_dataframe : pd.DataFrame
        Player rows to persist.

    Returns
    -------
    PlayerCatalog
        Loaded catalog.
    """
    player_dataframe.to_csv(config.paths.PLAYER_DATA_CSV, index=False)
    return load_player_catalog(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)


@pytest.fixture
def rules_engine(config: Config) -> FantasyRulesEngine:
    """Return the fantasy rules engine for tests.

    Parameters
    ----------
    config : Config
        Test configuration.

    Returns
    -------
    FantasyRulesEngine
        Rules engine using the configured roster limits.
    """
    total_roster_size = sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE
    return FantasyRulesEngine(
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_roster_size_per_team=total_roster_size,
    )


@pytest.fixture
def draft_state(config: Config, player_catalog: PlayerCatalog) -> DraftState:
    """Return a fresh ID-based draft state.

    Parameters
    ----------
    config : Config
        Test configuration.
    player_catalog : PlayerCatalog
        Shared player catalog.

    Returns
    -------
    DraftState
        Initialized draft state.
    """
    total_roster_size = sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE
    return DraftState(
        all_player_ids=set(player_catalog.player_ids),
        draft_order=[1, 2, 3, 4],
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_roster_size_per_team=total_roster_size,
        agent_team_id=config.draft.AGENT_START_POSITION,
    )


@pytest.fixture
def draft_controller(
    config: Config,
    draft_state: DraftState,
    player_catalog: PlayerCatalog,
    rules_engine: FantasyRulesEngine,
) -> DraftController:
    """Return a shared draft controller for tests.

    Parameters
    ----------
    config : Config
        Test configuration.
    draft_state : DraftState
        Mutable draft state.
    player_catalog : PlayerCatalog
        Shared player catalog.
    rules_engine : FantasyRulesEngine
        Draft rules engine.

    Returns
    -------
    DraftController
        Shared draft workflow coordinator.
    """
    return DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
    )


@pytest.fixture
def player_factory():
    """Return a helper for constructing lightweight Player objects."""

    def _build(
        player_id: int,
        position: str,
        projected_points: float = 100.0,
        team: str | None = "BUF",
        name: str | None = None,
        adp: float = 10.0,
        bye_week: int | None = 7,
    ) -> Player:
        return Player(
            player_id=player_id,
            name=name or f"{position}-{player_id}",
            position=position,
            projected_points=projected_points,
            adp=adp,
            bye_week=bye_week,
            team=team,
        )

    return _build


@pytest.fixture
def fake_session(player_catalog: PlayerCatalog):
    """Return a lightweight fake session for web route tests."""
    pick = Pick(pick_number=1, team_id=1, player_id=1)
    state = SimpleNamespace()
    session = SimpleNamespace(
        current_pick_number=2,
        draft_history=[pick],
        player_catalog=player_catalog,
        available_player_ids=set(player_catalog.player_ids),
        _state=state,
        weekly_projections=player_catalog.to_weekly_projections(),
        team_manager_mapping={1: "Team 1", 2: "Team 2", 3: "Team 3", 4: "Team 4"},
        get_ui_state=lambda: {"ok": True},
        draft_player=lambda player_id: None,
        undo_last_pick=lambda: None,
        set_current_team_picking=lambda team_id: None,
        simulate_single_pick=lambda: None,
        simulate_scheduled_picks_remaining=lambda: None,
        get_ai_suggestion=lambda: {"QB": 0.7},
        get_ai_suggestions_all=lambda: {1: {"QB": 0.7}},
        get_ai_suggestion_for_team=lambda team_id, ignore_player_ids=None: {
            "team_id": team_id,
            "ignore": ignore_player_ids or [],
        },
        save_state=lambda file_path: None,
        get_positional_baselines=lambda: {"QB": 250.0, "RB": 200.0, "WR": 180.0, "TE": 120.0},
    )
    return session


@pytest.fixture
def fake_env(player_factory):
    """Return a configurable fake RL env for reward and agent tests."""
    roster_map = {
        1: SimpleNamespace(player_ids=[1, 2]),
        2: SimpleNamespace(player_ids=[3, 4]),
    }
    resolved = {
        1: [player_factory(1, "QB", 200.0, "BUF"), player_factory(2, "WR", 150.0, "BUF")],
        2: [player_factory(3, "QB", 180.0, "KC"), player_factory(4, "TE", 120.0, "KC")],
    }

    class _FakeEnv:
        agent_team_id = 1
        total_roster_size_per_team = 2
        weekly_projections = {
            1: {"pts": [10.0] * 18, "pos": "QB"},
            2: {"pts": [8.0] * 18, "pos": "WR"},
        }

        def __init__(self):
            self._resolved = resolved
            self.team_rosters = roster_map

        def resolve_roster_players(self, team_id: int):
            return list(self._resolved[team_id])

        def _calculate_vorp(self, position: str) -> float:
            return {"QB": 10.0, "RB": 5.0, "WR": 7.5, "TE": 3.0}.get(position, 0.0)

    return _FakeEnv()


@pytest.fixture
def tiny_training_config(config: Config) -> Config:
    """Return config tuned for deterministic RL unit tests."""
    config.training.ENABLED_STATE_FEATURES = ["f1", "f2", "f3"]
    config.training.HIDDEN_DIM = 4
    config.training.BATCH_EPISODES = 1
    config.training.LOG_SAVE_INTERVAL_EPISODES = 1
    config.training.TOTAL_EPISODES = 1
    config.training.LEARNING_RATE = 0.001
    config.training.DISCOUNT_FACTOR = 0.9
    config.training.ENABLE_ACTION_MASKING = True
    return config
