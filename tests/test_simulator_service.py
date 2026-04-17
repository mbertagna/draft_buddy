"""Tests for the simulator service boundary."""

from __future__ import annotations

import pandas as pd

from draft_buddy.core import Pick
from draft_buddy.simulator.service import SeasonSimulationService


def test_season_simulation_service_passes_id_rosters_and_fallback_projections(
    config,
    draft_state,
    player_catalog,
    monkeypatch,
) -> None:
    """Verify the simulator service resolves explicit draft inputs without a session object."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.add_player_to_roster(1, player_catalog.require(2))
    draft_state.append_pick(Pick(pick_number=1, team_id=1, player_id=1))
    draft_state.append_pick(Pick(pick_number=2, team_id=1, player_id=2))
    captured = {}

    def fake_simulate(weekly_projections, matchups_df, rosters, *_args):
        captured["weekly_projections"] = weekly_projections
        captured["matchups"] = matchups_df
        captured["rosters"] = rosters
        return None, [("Team 1", {"W": 1, "L": 0, "T": 0, "pts": 123.0})], pd.DataFrame(), {}, "Team 1"

    service = SeasonSimulationService(config)
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1, "Matchup": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    result = service.simulate_season(draft_state, player_catalog, config.draft.TEAM_MANAGER_MAPPING)

    assert captured["rosters"] == {"Team 1": [1, 2]} and captured["weekly_projections"] == player_catalog.to_weekly_projections() and result["winner"] == "Team 1"


def test_season_simulation_service_uses_explicit_weekly_projections(
    config,
    draft_state,
    player_catalog,
    monkeypatch,
) -> None:
    """Verify provided weekly projections override catalog-derived defaults."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    captured = {}
    explicit_projections = {1: {"pos": "QB", "pts": [99.0]}}

    def fake_simulate(weekly_projections, matchups_df, rosters, *_args):
        captured["weekly_projections"] = weekly_projections
        _ = (matchups_df, rosters)
        return None, [], pd.DataFrame(), "", "Team 1"

    service = SeasonSimulationService(config)
    monkeypatch.setattr(service, "_load_matchups", lambda: pd.DataFrame([{"Week": 1, "Matchup": 1}]))
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)

    service.simulate_season(
        draft_state,
        player_catalog,
        config.draft.TEAM_MANAGER_MAPPING,
        weekly_projections=explicit_projections,
    )

    assert captured["weekly_projections"] == explicit_projections


def test_load_matchups_prefers_team_count_specific_file(config, tmp_path) -> None:
    """Verify matchup loading prefers the team-count-specific schedule when present."""
    config.paths.DATA_DIR = str(tmp_path)
    default_path = tmp_path / "red_league_matchups_2025.csv"
    specific_path = tmp_path / f"red_league_matchups_2025_{config.draft.NUM_TEAMS}_team.csv"
    pd.DataFrame([{"Week": 1, "Matchup": 99}]).to_csv(default_path, index=False)
    pd.DataFrame([{"Week": 1, "Matchup": 1}]).to_csv(specific_path, index=False)
    service = SeasonSimulationService(config)

    matchups = service._load_matchups()

    assert int(matchups.iloc[0]["Matchup"]) == 1


def test_load_matchups_falls_back_to_default_file(config, tmp_path) -> None:
    """Verify matchup loading falls back to the default schedule when needed."""
    config.paths.DATA_DIR = str(tmp_path)
    default_path = tmp_path / "red_league_matchups_2025.csv"
    pd.DataFrame([{"Week": 2, "Matchup": 7}]).to_csv(default_path, index=False)
    service = SeasonSimulationService(config)

    matchups = service._load_matchups()

    assert int(matchups.iloc[0]["Matchup"]) == 7


def test_format_playoff_results_converts_nan_values_to_none(config) -> None:
    """Verify playoff rows are normalized for JSON serialization."""
    service = SeasonSimulationService(config)
    playoff_results = pd.DataFrame(
        [
            {
                "Week": 15,
                "Matchup": 1,
                "Away Manager(s)": float("nan"),
                "Away Score": float("nan"),
                "Home Manager(s)": "Team 1",
                "Home Score": 120.5,
            }
        ]
    )

    formatted = service._format_playoff_results(playoff_results)

    assert formatted == [
        {
            "week": 15,
            "matchup": 1,
            "away_manager": None,
            "away_score": None,
            "home_manager": "Team 1",
            "home_score": 120.5,
        }
    ]
