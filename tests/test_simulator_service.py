"""Tests for season simulation service orchestration."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from draft_buddy.domain.entities import Player
from draft_buddy.simulator.service import SeasonSimulationService


def _build_config(data_dir: str = "/tmp/data", num_teams: int = 14, num_playoff_teams: int = 6):
    """Create a lightweight config-like object for tests."""
    return SimpleNamespace(
        draft=SimpleNamespace(NUM_TEAMS=num_teams),
        paths=SimpleNamespace(DATA_DIR=data_dir),
        reward=SimpleNamespace(REGULAR_SEASON_REWARD={"NUM_PLAYOFF_TEAMS": num_playoff_teams}),
    )


def test_simulate_season_unwraps_draft_session_rosters_to_manager_player_ids(monkeypatch):
    """Verify service maps team rosters into manager-keyed player-id lists."""
    captured = {}

    def fake_simulate(weekly_projections, matchups_df, rosters, season, output_file_prefix, save_data, num_playoff):
        captured["rosters"] = rosters
        return (
            pd.DataFrame(),
            [],
            pd.DataFrame(columns=["Week", "Matchup", "Away Manager(s)", "Away Score", "Home Score", "Home Manager(s)"]),
            "",
            "Winner",
        )

    service = SeasonSimulationService(_build_config())
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    draft_session = SimpleNamespace(
        teams_rosters={
            1: {"PLAYERS": [Player(1, "P1", "QB", 10.0), Player(2, "P2", "RB", 9.0)]},
            2: {"PLAYERS": [Player(3, "P3", "WR", 8.0)]},
        },
        team_manager_mapping={1: "Manager A", 2: "Manager B"},
        all_players_data=[],
        weekly_projections={},
    )
    service.simulate_season(draft_session)

    assert captured["rosters"] == {"Manager A": [1, 2], "Manager B": [3]}


def test_load_matchups_falls_back_to_default_when_size_specific_file_missing(monkeypatch):
    """Verify fallback to default matchups path when size-specific CSV is missing."""
    config = _build_config(data_dir="/league/data", num_teams=14)
    service = SeasonSimulationService(config)
    read_paths = []

    def fake_exists(path):
        return path.endswith("red_league_matchups_2025.csv")

    def fake_read_csv(path):
        read_paths.append(path)
        return pd.DataFrame([{"Week": 1}])

    monkeypatch.setattr("draft_buddy.simulator.service.os.path.exists", fake_exists)
    monkeypatch.setattr("draft_buddy.simulator.service.pd.read_csv", fake_read_csv)
    service._load_matchups()

    assert read_paths[-1].endswith("red_league_matchups_2025.csv")


def test_simulate_season_uses_default_weekly_projections_when_session_has_none(monkeypatch):
    """Verify service builds fallback weekly projections from all_players_data."""
    captured = {}

    def fake_simulate(weekly_projections, matchups_df, rosters, season, output_file_prefix, save_data, num_playoff):
        captured["weekly_projections"] = weekly_projections
        return (
            pd.DataFrame(),
            [],
            pd.DataFrame(columns=["Week", "Matchup", "Away Manager(s)", "Away Score", "Home Score", "Home Manager(s)"]),
            "",
            "Winner",
        )

    service = SeasonSimulationService(_build_config())
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    draft_session = SimpleNamespace(
        teams_rosters={1: {"PLAYERS": [Player(1, "P1", "QB", 10.0)]}},
        team_manager_mapping={1: "Manager A"},
        all_players_data=[Player(1, "P1", "QB", 10.0)],
        weekly_projections=None,
    )
    service.simulate_season(draft_session)

    assert captured["weekly_projections"][1]["pts"] == [10.0] * 18


def test_simulate_season_formats_playoff_nan_manager_as_none(monkeypatch):
    """Verify playoff formatter converts NaN manager names to None."""
    playoffs_df = pd.DataFrame(
        [
            {
                "Week": 15,
                "Matchup": 1,
                "Away Manager(s)": np.nan,
                "Away Score": 88.0,
                "Home Score": 90.0,
                "Home Manager(s)": "Manager A",
            }
        ]
    )

    def fake_simulate(*args, **kwargs):
        return pd.DataFrame(), [], playoffs_df, "tree", "Manager A"

    service = SeasonSimulationService(_build_config())
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    draft_session = SimpleNamespace(
        teams_rosters={1: {"PLAYERS": [Player(1, "P1", "QB", 10.0)]}},
        team_manager_mapping={1: "Manager A"},
        all_players_data=[Player(1, "P1", "QB", 10.0)],
        weekly_projections={1: {"pts": [10.0] * 18, "pos": "QB"}},
    )
    result = service.simulate_season(draft_session)

    assert result["playoff_results"][0]["away_manager"] is None


def test_simulate_season_skips_team_without_manager_mapping(monkeypatch):
    """Verify teams with missing manager mapping are omitted from roster payload."""
    captured = {}

    def fake_simulate(weekly_projections, matchups_df, rosters, season, output_file_prefix, save_data, num_playoff):
        captured["rosters"] = rosters
        return pd.DataFrame(), [], pd.DataFrame(), "", "Winner"

    service = SeasonSimulationService(_build_config())
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    draft_session = SimpleNamespace(
        teams_rosters={
            1: {"PLAYERS": [Player(1, "P1", "QB", 10.0)]},
            2: {"PLAYERS": [Player(2, "P2", "RB", 9.0)]},
        },
        team_manager_mapping={1: "Manager A"},
        all_players_data=[Player(1, "P1", "QB", 10.0), Player(2, "P2", "RB", 9.0)],
        weekly_projections={1: {"pts": [10.0], "pos": "QB"}, 2: {"pts": [9.0], "pos": "RB"}},
    )
    service.simulate_season(draft_session)

    assert captured["rosters"] == {"Manager A": [1]}


def test_load_matchups_attempts_default_csv_when_no_candidates_exist(monkeypatch):
    """Verify final default read is attempted if no candidate path exists."""
    config = _build_config(data_dir="/league/data", num_teams=12)
    service = SeasonSimulationService(config)
    read_paths = []

    monkeypatch.setattr("draft_buddy.simulator.service.os.path.exists", lambda path: False)

    def fake_read_csv(path):
        read_paths.append(path)
        return pd.DataFrame([{"Week": 1}])

    monkeypatch.setattr("draft_buddy.simulator.service.pd.read_csv", fake_read_csv)
    service._load_matchups()

    assert read_paths[-1].endswith("red_league_matchups_2025.csv")
