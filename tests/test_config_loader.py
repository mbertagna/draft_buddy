"""Tests for league profile loading."""

from __future__ import annotations

import pytest

from draft_buddy.config import Config, load_runtime_config
from draft_buddy.config.loader import LEAGUE_ENV_VAR, SEASON_ENV_VAR


def test_red_league_profile_matches_espn_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Red League uses 10 teams, FLEX 3, full PPR, and draft slot 2."""
    monkeypatch.delenv(LEAGUE_ENV_VAR, raising=False)
    monkeypatch.delenv(SEASON_ENV_VAR, raising=False)
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.draft.NUM_TEAMS == 10
    assert config.draft.ROSTER_STRUCTURE["FLEX"] == 3
    assert config.draft.TOTAL_BENCH_SIZE == 7
    assert config.draft.AGENT_START_POSITION == 2
    assert config.get_scoring_rules()["receptions"] == 1.0
    assert config.get_scoring_rules()["passing_tds"] == 6
    assert "/src/data/" not in config.paths.PLAYER_DATA_CSV.replace("\\", "/")
    assert "data/leagues/red_league_10" in config.paths.PLAYER_DATA_CSV.replace("\\", "/")
    assert config.paths.PLAYER_DATA_CSV.endswith("2026/generated_player_data.csv")


def test_redraft_nbfl_profile_matches_sleeper_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Redraft NBFL uses 12 teams, half PPR, and draft slot 5."""
    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert config.draft.NUM_TEAMS == 12
    assert config.draft.ROSTER_STRUCTURE["FLEX"] == 2
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


def test_position_guide_checkpoint_dir_from_season_overlay() -> None:
    """Verify position guide checkpoint directory is season-scoped."""
    config = load_runtime_config(league_id="redraft_nbfl_12", season=2026)

    assert config.season.position_guide_checkpoint_dir == "models/12_teams_random_start/v3"


def test_team_manager_mapping_uses_integer_keys() -> None:
    """Verify JSON string keys are coerced to integer team ids."""
    config = load_runtime_config(league_id="red_league_10", season=2026)

    assert config.draft.TEAM_MANAGER_MAPPING[1] == "SKOL!"
    assert config.draft.TEAM_MANAGER_MAPPING[2] == "California Fourskin"
    assert config.draft.TEAM_MANAGER_MAPPING[10] == "Team Sully LLC"


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
