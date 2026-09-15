"""Tests for league profile loading."""

from __future__ import annotations

import pytest

from pathlib import Path

from draft_buddy.config import Config, load_runtime_config
from draft_buddy.config import loader as config_loader
from draft_buddy.config.loader import LEAGUE_ENV_VAR, SEASON_ENV_VAR


def _posix(path: str) -> str:
    """Normalize a filesystem path to POSIX separators for assertions."""
    return path.replace("\\", "/")


def test_red_league_profile_matches_espn_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Red League uses 10 teams, FLEX 3, full PPR, and draft slot 2."""
    monkeypatch.delenv(LEAGUE_ENV_VAR, raising=False)
    monkeypatch.delenv(SEASON_ENV_VAR, raising=False)
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.draft.NUM_TEAMS == 10
    assert config.draft.ROSTER_STRUCTURE["FLEX"] == 3
    assert config.draft.TOTAL_BENCH_SIZE == 7
    assert config.draft.BENCH_MAXES == {"QB": 1, "RB": 3, "WR": 3, "TE": 2}
    assert config.draft.PLATFORM_BENCH_MAXES == {"QB": 3, "RB": 8, "WR": 8, "TE": 4}
    assert config.draft.AGENT_START_POSITION == 2
    assert config.get_scoring_rules()["receptions"] == 1.0
    assert config.get_scoring_rules()["passing_tds"] == 6
    assert "/src/data/" not in config.paths.PLAYER_DATA_CSV.replace("\\", "/")
    assert "data/leagues/red_league_10" in config.paths.PLAYER_DATA_CSV.replace("\\", "/")
    assert config.paths.PLAYER_DATA_CSV.endswith("2026/generated_player_data.csv")


def test_redraft_nbfl_profile_matches_sleeper_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Redraft NBFL uses 12 teams, half PPR, FLEX 3, and draft slot 5."""
    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert config.draft.NUM_TEAMS == 12
    assert config.draft.ROSTER_STRUCTURE["FLEX"] == 3
    assert config.draft.AGENT_START_POSITION == 5
    assert config.get_scoring_rules()["receptions"] == 0.5
    assert config.get_scoring_rules()["passing_tds"] == 4
    assert config.league.display_name == "Redraft NBFL"
    assert "redraft_nbfl_12" in config.paths.PLAYER_DATA_CSV


def test_env_vars_override_default_league(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DRAFT_BUDDY_LEAGUE selects the active profile."""
    monkeypatch.setenv(LEAGUE_ENV_VAR, "redraft_nbfl_12")
    monkeypatch.setenv(SEASON_ENV_VAR, "2026")
    config = load_runtime_config()

    assert config.league.league_id == "redraft_nbfl_12"
    assert config.draft.NUM_TEAMS == 12


def test_default_league_is_red_league_10(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the imminent draft league is the default active profile."""
    monkeypatch.delenv(LEAGUE_ENV_VAR, raising=False)
    monkeypatch.delenv(SEASON_ENV_VAR, raising=False)
    config = load_runtime_config()

    assert config.league.league_id == "red_league_10"
    assert config.draft.NUM_TEAMS == 10


def test_season_overlay_sets_bye_weeks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify bye weeks are loaded from the season overlay."""
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.season.bye_weeks[5] == ["CAR", "KC"]
    assert config.season.bye_weeks[14] == ["ARI", "DAL"]


def test_model_path_to_load_from_season_overlay() -> None:
    """Verify the checkpoint path is season-scoped via training config."""
    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert config.training.MODEL_PATH_TO_LOAD == "models/12_teams_random_start/v1"


def test_team_manager_mapping_uses_integer_keys() -> None:
    """Verify JSON string keys are coerced to integer team ids."""
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.draft.TEAM_MANAGER_MAPPING[1] == "SKOL!"
    assert config.draft.TEAM_MANAGER_MAPPING[2] == "California Fourskin"
    assert config.draft.TEAM_MANAGER_MAPPING[10] == "Team Sully LLC"


def test_redraft_nbfl_team_manager_mapping_uses_lottery_order() -> None:
    """Verify Redraft NBFL draft order matches the 2026 lottery."""
    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert config.draft.TEAM_MANAGER_MAPPING[1] == "Joey"
    assert config.draft.TEAM_MANAGER_MAPPING[5] == "Michael"
    assert config.draft.TEAM_MANAGER_MAPPING[12] == "Frank"
    assert config.draft.SLEEPER_SYNC_ENABLED is True
    assert config.draft.SLEEPER_DRAFT_ID == "1363697271027290112"
    assert config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID[11] == 5
    assert config.draft.SLEEPER_SYNC_POLL_SECONDS == 3


def test_missing_league_profile_raises_file_not_found() -> None:
    """Verify unknown league ids fail fast."""
    with pytest.raises(FileNotFoundError):
        load_runtime_config(league_id="missing_league", season=2026)


def test_config_to_dict_includes_league_and_scoring() -> None:
    """Verify serialization includes league metadata and scoring rules."""
    config = load_runtime_config(league_id="red_league_10", season=2026)
    payload = config.to_dict()

    assert "league" in payload
    assert "scoring" in payload
    assert "season" in payload
    assert payload["scoring"]["rules"]["receptions"] == 1.0


def test_config_includes_fantasypros_market_data_defaults() -> None:
    """Verify FantasyPros market-data defaults are available from data config."""
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.data.FANTASYPROS_API_ENABLED is True
    assert config.data.FANTASYPROS_API_KEY_ENV == "FANTASYPROS_API_KEY"


def test_runtime_paths_are_scoped_per_league_and_season() -> None:
    """Verify generated data and draft state live under distinct league year dirs."""
    red = load_runtime_config(league_id="red_league_10", season=2026)
    nbfl = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert _posix(red.paths.PLAYER_DATA_CSV).endswith(
        "data/leagues/red_league_10/generated/2026/generated_player_data.csv"
    )
    assert _posix(nbfl.paths.PLAYER_DATA_CSV).endswith(
        "data/leagues/redraft_nbfl_12/generated/2026/generated_player_data.csv"
    )
    assert _posix(red.paths.DRAFT_STATE_FILE).endswith(
        "data/leagues/red_league_10/runtime/2026/draft_state.json"
    )
    assert _posix(nbfl.paths.DRAFT_STATE_FILE).endswith(
        "data/leagues/redraft_nbfl_12/runtime/2026/draft_state.json"
    )
    assert _posix(red.paths.DRAFT_STATE_PREV_FILE).endswith(
        "data/leagues/red_league_10/runtime/2026/draft_state.prev.json"
    )
    assert _posix(nbfl.paths.DRAFT_STATE_PREV_FILE).endswith(
        "data/leagues/redraft_nbfl_12/runtime/2026/draft_state.prev.json"
    )
    assert _posix(red.paths.SAVED_STATES_DIR).endswith("data/leagues/red_league_10/saved_states")
    assert _posix(nbfl.paths.SAVED_STATES_DIR).endswith(
        "data/leagues/redraft_nbfl_12/saved_states"
    )
    assert red.paths.DRAFT_STATE_FILE != nbfl.paths.DRAFT_STATE_FILE
    assert red.paths.SAVED_STATES_DIR != nbfl.paths.SAVED_STATES_DIR


def test_player_csv_derives_league_layout_when_template_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing player_data_template still uses the league-year CSV layout."""
    original_load = config_loader._load_json

    def load_without_template(path: Path) -> dict:
        payload = original_load(path)
        if path.parent.name == "leagues":
            payload = dict(payload)
            payload.pop("paths", None)
        return payload

    monkeypatch.setattr(config_loader, "_load_json", load_without_template)
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert _posix(config.paths.PLAYER_DATA_CSV).endswith(
        "data/leagues/red_league_10/generated/2026/generated_player_data.csv"
    )
