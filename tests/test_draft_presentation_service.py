"""Tests for draft presentation helpers."""

from types import SimpleNamespace

import numpy as np

from draft_buddy.domain.entities import Player
from draft_buddy.logic.draft_presentation_service import DraftPresentationService


def _build_env(*, overridden_team_id=None, current_pick_idx=0):
    """Return a minimal env stub for presentation tests."""
    team_one_players = [
        Player(1, "QB One", "QB", 200.0, bye_week=7, team="BUF"),
        Player(2, "WR One", "WR", 150.0, bye_week=7, team="BUF"),
        Player(3, "RB One", "RB", 140.0, bye_week=np.nan, team="MIA"),
    ]
    team_two_players = [
        Player(4, "QB Two", "QB", 180.0, bye_week=10, team="KC"),
    ]
    return SimpleNamespace(
        _overridden_team_id=overridden_team_id,
        draft_order=[1, 2],
        current_pick_idx=current_pick_idx,
        current_pick_number=current_pick_idx + 1,
        teams_rosters={
            1: {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "PLAYERS": team_one_players},
            2: {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "PLAYERS": team_two_players},
        },
        roster_structure={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0},
        bench_maxes={"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        total_roster_size_per_team=3,
        manual_draft_teams={2},
        num_teams=2,
        agent_team_id=1,
    )


def test_get_ui_state_prefers_override_team_for_current_team():
    """Verify override teams replace the scheduled team-on-clock value."""
    payload = DraftPresentationService.get_ui_state(_build_env(overridden_team_id=2))

    assert payload["current_team_picking"] == 2


def test_get_ui_state_uses_none_when_pick_index_is_exhausted():
    """Verify exhausted draft order reports no current team."""
    payload = DraftPresentationService.get_ui_state(_build_env(current_pick_idx=2))

    assert payload["current_team_picking"] is None


def test_get_ui_state_marks_full_teams_by_total_roster_size():
    """Verify full-team flags compare against the configured roster cap."""
    payload = DraftPresentationService.get_ui_state(_build_env())

    assert payload["team_is_full"][1] is True


def test_get_ui_state_includes_starter_and_bench_point_summaries():
    """Verify presentation payload includes roster point summaries."""
    payload = DraftPresentationService.get_ui_state(_build_env())

    assert payload["team_points_summary"][1]["starters_total"] > 0


def test_aggregate_bye_weeks_filters_nan_and_missing_bye_values():
    """Verify bye aggregation excludes invalid bye values."""
    bye_data = DraftPresentationService._aggregate_bye_weeks(_build_env())

    assert bye_data[1] == {7: {"QB": 1, "WR": 1}}
