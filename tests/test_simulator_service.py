"""Tests for the simulator service boundary."""

from __future__ import annotations

import pandas as pd

from draft_buddy.core import Pick
from draft_buddy.simulator.service import SeasonSimulationService, parse_team_id, team_label


def test_team_label_and_parse_team_id_helpers() -> None:
    """Verify team label helpers round-trip canonical ids."""
    assert team_label(5) == "Team 5"
    assert parse_team_id("Team 5") == 5
    assert parse_team_id("Michael Bertagna") is None


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
        regular_results = pd.DataFrame(
            [
                {
                    "Week": 1,
                    "Matchup": 1,
                    "Away Manager(s)": "Team 1",
                    "Home Manager(s)": "Team 2",
                    "Away Score": 110.0,
                    "Home Score": 95.0,
                }
            ]
        )
        return (
            regular_results,
            [("Team 1", {"W": 1, "L": 0, "T": 0, "pts": 123.0})],
            pd.DataFrame(),
            {},
            "Team 1",
        )

    service = SeasonSimulationService(config)
    monkeypatch.setattr(
        service,
        "_load_matchups",
        lambda: pd.DataFrame(
            [
                {
                    "Week": 1,
                    "Matchup": 1,
                    "Away Manager(s)": "Team 1",
                    "Home Manager(s)": "Team 2",
                }
            ]
        ),
    )
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)
    result = service.simulate_season(draft_state, player_catalog, config.draft.TEAM_MANAGER_MAPPING)

    assert captured["rosters"] == {
        "Team 1": [1, 2],
        "Team 2": [],
        "Team 3": [],
        "Team 4": [],
    }
    assert captured["weekly_projections"] == player_catalog.to_weekly_projections()
    assert result["winner"] == "Team 1"
    assert result["winner_team_id"] == 1
    assert result["regular_season_records"] == [
        {"team_id": 1, "team": "Team 1", "W": 1, "L": 0, "T": 0, "pts": 123.0}
    ]
    assert result["regular_season_matchups"] == [
        {
            "week": 1,
            "matchup": 1,
            "away_team_id": 1,
            "home_team_id": 2,
            "away_team": "Team 1",
            "home_team": "Team 2",
            "away_score": 110.0,
            "home_score": 95.0,
        }
    ]


def test_season_simulation_service_translates_matchup_manager_names(
    config,
    draft_state,
    player_catalog,
    monkeypatch,
) -> None:
    """Verify matchup CSV display names are translated to Team labels before simulation."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    captured = {}

    def fake_simulate(_weekly_projections, matchups_df, rosters, *_args):
        captured["matchups"] = matchups_df
        captured["rosters"] = rosters
        return pd.DataFrame(), [], pd.DataFrame(), "", "Team 1"

    service = SeasonSimulationService(config)
    monkeypatch.setattr(
        service,
        "_load_matchups",
        lambda: pd.DataFrame(
            [
                {
                    "Week": 1,
                    "Matchup": 1,
                    "Away Manager(s)": "Team 1",
                    "Home Manager(s)": "Team 2",
                }
            ]
        ),
    )
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)

    service.simulate_season(draft_state, player_catalog, config.draft.TEAM_MANAGER_MAPPING)

    assert captured["rosters"]["Team 1"] == [1]
    assert captured["matchups"].iloc[0]["Away Manager(s)"] == "Team 1"
    assert captured["matchups"].iloc[0]["Home Manager(s)"] == "Team 2"


def test_season_simulation_service_uses_random_matchups_with_team_labels(
    config,
    draft_state,
    player_catalog,
    monkeypatch,
) -> None:
    """Verify random schedules are generated with Team labels matching rosters."""
    config.reward.USE_RANDOM_MATCHUPS = True
    config.reward.NUM_REGULAR_SEASON_WEEKS = 2
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.add_player_to_roster(2, player_catalog.require(2))
    captured = {}

    def fake_simulate(_weekly_projections, matchups_df, rosters, *_args):
        captured["matchups"] = matchups_df
        captured["rosters"] = rosters
        return pd.DataFrame(), [], pd.DataFrame(), "", "Team 1"

    service = SeasonSimulationService(config)
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)

    service.simulate_season(draft_state, player_catalog, config.draft.TEAM_MANAGER_MAPPING)

    assert set(captured["rosters"]) == {"Team 1", "Team 2", "Team 3", "Team 4"}
    managers = set(captured["matchups"]["Away Manager(s)"]) | set(
        captured["matchups"]["Home Manager(s)"]
    )
    assert managers <= {"Team 1", "Team 2", "Team 3", "Team 4"}
    assert int(captured["matchups"]["Week"].max()) == 2


def test_translate_matchup_labels_falls_back_to_team_nickname_columns(config) -> None:
    """Verify unresolved manager cells remap via Away/Home Team nicknames."""
    service = SeasonSimulationService(config)
    label_by_name = {"Club 33": "Team 1", "Huckleberry": "Team 5"}
    matchups = pd.DataFrame(
        [
            {
                "Week": 1,
                "Matchup": 1,
                "Away Team": "Club 33",
                "Away Manager(s)": "Jake D'Alonzo",
                "Home Team": "Huckleberry",
                "Home Manager(s)": "Noah Hollander",
            }
        ]
    )

    translated = service._translate_matchup_labels(matchups, label_by_name)

    assert translated.iloc[0]["Away Manager(s)"] == "Team 1"
    assert translated.iloc[0]["Home Manager(s)"] == "Team 5"


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
        return pd.DataFrame(), [], pd.DataFrame(), "", "Team 1"

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


def test_load_matchups_returns_empty_when_files_missing(config, tmp_path) -> None:
    """Verify matchup loading returns an empty frame when no CSV exists."""
    config.paths.DATA_DIR = str(tmp_path)
    service = SeasonSimulationService(config)

    assert service._load_matchups().empty


def test_season_simulation_keeps_compute_keys_when_display_names_differ(
    config,
    draft_state,
    player_catalog,
    monkeypatch,
) -> None:
    """Verify cosmetic nicknames decorate output without changing compute keys."""
    config.draft.TEAM_MANAGER_MAPPING = {1: "Club 33", 2: "Goofy's Kitchen", 3: "Huck", 4: "Dawgs"}
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    captured = {}

    def fake_simulate(_weekly_projections, matchups_df, rosters, *_args):
        captured["matchups"] = matchups_df
        captured["rosters"] = rosters
        regular_results = pd.DataFrame(
            [
                {
                    "Week": 1,
                    "Matchup": 1,
                    "Away Manager(s)": "Team 1",
                    "Home Manager(s)": "Team 2",
                    "Away Score": 110.0,
                    "Home Score": 95.0,
                }
            ]
        )
        return (
            regular_results,
            [("Team 1", {"W": 1, "L": 0, "T": 0, "pts": 123.0})],
            pd.DataFrame(),
            {},
            "Team 1",
        )

    service = SeasonSimulationService(config)
    monkeypatch.setattr(
        service,
        "_load_matchups",
        lambda: pd.DataFrame(
            [{"Week": 1, "Matchup": 1, "Away Manager(s)": "Team 1", "Home Manager(s)": "Team 2"}]
        ),
    )
    monkeypatch.setattr("draft_buddy.simulator.service.simulate_season_fast", fake_simulate)

    result = service.simulate_season(
        draft_state, player_catalog, config.draft.TEAM_MANAGER_MAPPING
    )

    assert set(captured["rosters"]) == {"Team 1", "Team 2", "Team 3", "Team 4"}
    assert captured["matchups"].iloc[0]["Away Manager(s)"] == "Team 1"
    assert result["winner"] == "Club 33"
    assert result["winner_team_id"] == 1
    assert result["regular_season_records"][0]["team"] == "Club 33"
    assert result["regular_season_records"][0]["team_id"] == 1
    assert result["regular_season_matchups"][0]["away_team"] == "Club 33"
    assert result["regular_season_matchups"][0]["away_team_id"] == 1


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

    formatted = service._format_playoff_results(playoff_results, {})

    assert formatted == [
        {
            "week": 15,
            "matchup": 1,
            "away_team_id": None,
            "home_team_id": 1,
            "away_team": None,
            "home_team": "Team 1",
            "away_score": None,
            "home_score": 120.5,
        }
    ]
