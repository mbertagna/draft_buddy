"""Tests for draft assistant context builder."""

from __future__ import annotations

from draft_buddy.core.entities import Player
from draft_buddy.web.draft_advisor_context import (
    DEFAULT_TOP_K,
    HIGH_PRIORITY_TOP_K,
    build_advisor_context,
    build_candidate_rows,
    collect_candidate_player_ids,
    position_top_k_map,
)


def _player(player_id: int, position: str, projected: float, adp: float) -> Player:
    """Build a test player."""
    return Player(
        player_id=player_id,
        name=f"Player {player_id}",
        position=position,
        projected_points=projected,
        adp=adp,
    )


def test_position_top_k_map_assigns_seven_to_top_two_positions() -> None:
    """Verify the two highest RL positions receive K=7."""
    top_k = position_top_k_map({"QB": 0.1, "RB": 0.5, "WR": 0.3, "TE": 0.1})

    assert top_k["RB"] == HIGH_PRIORITY_TOP_K
    assert top_k["WR"] == HIGH_PRIORITY_TOP_K
    assert top_k["QB"] == DEFAULT_TOP_K
    assert top_k["TE"] == DEFAULT_TOP_K


def test_position_top_k_map_tie_breaks_with_position_order() -> None:
    """Verify equal probabilities tie-break using QB, RB, WR, TE order."""
    top_k = position_top_k_map({"QB": 0.25, "RB": 0.25, "WR": 0.25, "TE": 0.25})

    assert top_k["QB"] == HIGH_PRIORITY_TOP_K
    assert top_k["RB"] == HIGH_PRIORITY_TOP_K


def test_build_advisor_context_includes_glossary_and_roster(config) -> None:
    """Verify markdown context contains key sections."""
    players = [
        _player(1, "RB", 240.0, 2.0),
        _player(2, "WR", 230.0, 3.0),
    ]
    rows = build_candidate_rows(players, {"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0})
    top_k = position_top_k_map({"QB": 0.1, "RB": 0.6, "WR": 0.2, "TE": 0.1})
    ui_state = {
        "current_pick_number": 5,
        "num_teams": 4,
        "snake_team_on_turn": 1,
        "override_active": False,
        "total_roster_size_per_team": 16,
        "team_rosters": {
            1: {"starters": {"RB": [players[0].to_dict()]}, "bench": []},
            2: {"starters": {}, "bench": []},
        },
        "roster_counts": {
            1: {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
            2: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        },
        "team_bye_weeks": {1: {}, 2: {}},
    }
    context = build_advisor_context(
        ui_state=ui_state,
        advising_team_id=1,
        agent_team_id=1,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_bench_size=config.draft.TOTAL_BENCH_SIZE,
        team_manager_mapping=config.draft.TEAM_MANAGER_MAPPING,
        candidate_rows=rows,
        baselines={"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0},
        top_k_by_position=top_k,
        rl_probs={"QB": 0.1, "RB": 0.6, "WR": 0.2, "TE": 0.1},
        rl_degraded=False,
        insights={},
        recent_picks=[
            {
                "pick_number": 4,
                "team_id": 2,
                "player_name": "Recent RB",
                "position": "RB",
            }
        ],
    )

    assert "## Field glossary" in context
    assert "## Advising team roster" in context
    assert "## Roster targets" in context
    assert "starters_req" in context
    assert "## Recent picks" in context
    assert "Recent RB" in context
    assert "## League snapshot (other teams)" in context
    assert "RB — by VORP" in context
    assert "candidate shortlist K=7" in context


def test_format_position_targets_table_shows_remaining_room(config) -> None:
    """Verify roster target table includes per-position caps and open starter slots."""
    from draft_buddy.web.draft_advisor_context import _format_position_targets_table

    player = _player(1, "RB", 240.0, 2.0)
    roster = {"starters": {"RB": [player.to_dict()]}, "bench": [], "players_flat": [player.to_dict()]}
    table = _format_position_targets_table(
        roster=roster,
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3},
        bench_maxes={"QB": 3, "RB": 8, "WR": 8, "TE": 4},
        total_bench_size=7,
        total_roster_size=16,
    )

    assert "| RB | 1 | 2 | 1 |" in table
    assert "room_at_pos" in table


def test_collect_candidate_player_ids_unions_vorp_and_adp_tables() -> None:
    """Verify valid recommendation ids include both ranking views."""
    players = [
        _player(1, "RB", 240.0, 2.0),
        _player(2, "RB", 220.0, 6.0),
        _player(3, "RB", 210.0, 8.0),
    ]
    rows = build_candidate_rows(players, {"RB": 180.0, "WR": 170.0, "QB": 200.0, "TE": 120.0})
    top_k = {"QB": 5, "RB": 2, "WR": 5, "TE": 5}
    valid_ids = collect_candidate_player_ids(rows, top_k)

    assert valid_ids == {1, 2}
