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
