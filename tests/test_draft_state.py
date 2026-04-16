"""Tests for DraftState core mutation behavior."""

from draft_buddy.core.draft_state import DraftState
from draft_buddy.domain.entities import Player


def test_draft_state_starts_with_pick_index_zero(core_draft_state):
    """Verify initial pick index is zero."""
    assert core_draft_state.get_current_pick_idx() == 0


def test_draft_state_starts_with_pick_number_one(core_draft_state):
    """Verify initial global pick number is one."""
    assert core_draft_state.get_current_pick_number() == 1


def test_add_player_to_roster_removes_player_id_from_available_ids(core_draft_state, core_player_map):
    """Verify drafted player is removed from available player ids."""
    core_draft_state.add_player_to_roster(1, core_player_map[2])

    assert 2 not in core_draft_state.get_available_player_ids()


def test_advance_pick_increments_pick_index(core_draft_state):
    """Verify advancing pick increments pick index."""
    core_draft_state.advance_pick()

    assert core_draft_state.get_current_pick_idx() == 1


def test_advance_pick_increments_pick_number(core_draft_state):
    """Verify advancing pick increments global pick number."""
    core_draft_state.advance_pick()

    assert core_draft_state.get_current_pick_number() == 2


def test_add_player_cascades_rb_to_flex_when_rb_slots_full():
    """Verify an extra RB fills FLEX after RB starter slots are full."""
    state = DraftState(
        all_player_ids={1, 2, 3},
        draft_order=[1, 2, 1],
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        total_roster_size_per_team=10,
    )
    state.add_player_to_roster(1, Player(1, "RB A", "RB", 200.0))
    state.add_player_to_roster(1, Player(2, "RB B", "RB", 190.0))
    state.add_player_to_roster(1, Player(3, "RB C", "RB", 180.0))

    assert state.get_rosters()[1]["FLEX"] == 1


def test_remove_player_from_roster_decrements_flex_for_flex_occupant():
    """Verify undoing a FLEX RB pick decrements FLEX count."""
    state = DraftState(
        all_player_ids={1, 2, 3},
        draft_order=[1, 2, 1],
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        total_roster_size_per_team=10,
    )
    rb_one = Player(1, "RB A", "RB", 200.0)
    rb_two = Player(2, "RB B", "RB", 190.0)
    rb_flex = Player(3, "RB C", "RB", 180.0)
    state.add_player_to_roster(1, rb_one)
    state.add_player_to_roster(1, rb_two)
    state.add_player_to_roster(1, rb_flex)
    state.remove_player_from_roster(1, rb_flex)

    assert state.get_rosters()[1]["FLEX"] == 0


def test_add_player_increments_primary_position_when_primary_and_flex_are_full():
    """Verify extra RB increments RB counter when FLEX is unavailable."""
    state = DraftState(
        all_player_ids={1, 2, 3, 4},
        draft_order=[1, 2, 1],
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        total_roster_size_per_team=10,
    )
    state.add_player_to_roster(1, Player(1, "RB A", "RB", 200.0))
    state.add_player_to_roster(1, Player(2, "RB B", "RB", 190.0))
    state.add_player_to_roster(1, Player(3, "RB FLEX", "RB", 180.0))
    state.add_player_to_roster(1, Player(4, "RB BENCH", "RB", 170.0))

    assert state.get_rosters()[1]["RB"] == 3


def test_reset_clears_draft_history(core_draft_state, core_player_map):
    """Verify reset wipes draft history."""
    core_draft_state.append_draft_history({"team_id": 1, "player_id": 2})
    core_draft_state.reset(set(core_player_map.keys()), [4, 3, 2, 1], 4)

    assert core_draft_state.get_draft_history() == []


def test_reset_clears_all_rostered_players(core_draft_state, core_player_map):
    """Verify reset clears team players from rosters."""
    core_draft_state.add_player_to_roster(1, core_player_map[2])
    core_draft_state.reset(set(core_player_map.keys()), [4, 3, 2, 1], 4)

    assert core_draft_state.get_rosters()[1]["PLAYERS"] == []


def test_reset_restores_exact_original_available_player_ids(core_draft_state, core_player_map):
    """Verify reset restores original available player id set."""
    core_draft_state.add_player_to_roster(1, core_player_map[2])
    original_ids = set(core_player_map.keys())
    core_draft_state.reset(original_ids, [4, 3, 2, 1], 4)

    assert core_draft_state.get_available_player_ids() == original_ids


def test_get_roster_structure_returns_configured_structure(core_draft_state):
    """Verify roster structure getter returns configured values."""
    assert core_draft_state.get_roster_structure()["QB"] == 1


def test_get_bench_maxes_returns_configured_bench_limits(core_draft_state):
    """Verify bench max getter returns configured values."""
    assert core_draft_state.get_bench_maxes()["RB"] == 2


def test_get_total_roster_size_per_team_returns_configured_limit(core_draft_state):
    """Verify total roster size getter returns configured value."""
    assert core_draft_state.get_total_roster_size_per_team() == 13


def test_set_current_pick_idx_updates_pick_index(core_draft_state):
    """Verify setting pick index updates state."""
    core_draft_state.set_current_pick_idx(7)

    assert core_draft_state.get_current_pick_idx() == 7


def test_set_current_pick_number_updates_pick_number(core_draft_state):
    """Verify setting pick number updates state."""
    core_draft_state.set_current_pick_number(9)

    assert core_draft_state.get_current_pick_number() == 9


def test_set_agent_team_id_updates_agent_team(core_draft_state):
    """Verify setting agent team id updates state."""
    core_draft_state.set_agent_team_id(99)

    assert core_draft_state.get_agent_team_id() == 99


def test_set_overridden_team_id_updates_override(core_draft_state):
    """Verify setting overridden team id updates state."""
    core_draft_state.set_overridden_team_id(77)

    assert core_draft_state.get_overridden_team_id() == 77


def test_pop_draft_history_returns_none_when_history_is_empty(core_draft_state):
    """Verify popping empty history returns None."""
    assert core_draft_state.pop_draft_history() is None


def test_pop_draft_history_returns_last_entry(core_draft_state):
    """Verify popping history returns the newest draft entry."""
    core_draft_state.append_draft_history({"pick_number": 1})

    assert core_draft_state.pop_draft_history() == {"pick_number": 1}


def test_load_from_serialized_populates_available_ids(core_draft_state):
    """Verify loading serialized state restores available ids."""
    serialized = {
        "available_players_ids": [2, 3],
        "teams_rosters": {},
        "draft_order": [2, 1],
        "current_pick_idx": 1,
        "current_pick_number": 2,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_available_player_ids() == {2, 3}


def test_load_from_serialized_populates_team_roster_counts_and_players(core_draft_state):
    """Verify loading serialized state reconstructs roster content."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {
            "3": {
                "QB": 1,
                "RB": 0,
                "WR": 0,
                "TE": 0,
                "FLEX": 0,
                "PLAYERS": [{"player_id": 1}],
            }
        },
        "draft_order": [],
        "current_pick_idx": 0,
        "current_pick_number": 1,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload["player_id"])

    assert core_draft_state.get_rosters()[3]["PLAYERS"] == [1]


def test_load_from_serialized_skips_invalid_team_ids(core_draft_state):
    """Verify invalid serialized team ids are ignored."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {"bad-team-id": {"QB": 1, "PLAYERS": []}},
        "draft_order": [],
        "current_pick_idx": 0,
        "current_pick_number": 1,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert "bad-team-id" not in core_draft_state.get_rosters()


def test_load_from_serialized_restores_draft_order(core_draft_state):
    """Verify loading serialized state restores draft order."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {},
        "draft_order": [9, 8, 7],
        "current_pick_idx": 0,
        "current_pick_number": 1,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_draft_order() == [9, 8, 7]


def test_load_from_serialized_restores_pick_index(core_draft_state):
    """Verify loading serialized state restores current pick index."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {},
        "draft_order": [],
        "current_pick_idx": 4,
        "current_pick_number": 1,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_current_pick_idx() == 4


def test_load_from_serialized_restores_pick_number(core_draft_state):
    """Verify loading serialized state restores current pick number."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {},
        "draft_order": [],
        "current_pick_idx": 0,
        "current_pick_number": 8,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_current_pick_number() == 8


def test_load_from_serialized_restores_draft_history(core_draft_state):
    """Verify loading serialized state restores draft history."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {},
        "draft_order": [],
        "current_pick_idx": 0,
        "current_pick_number": 1,
        "_draft_history": [{"pick_number": 2}],
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_draft_history() == [{"pick_number": 2}]


def test_load_from_serialized_restores_overridden_team_id(core_draft_state):
    """Verify loading serialized state restores overridden team id."""
    serialized = {
        "available_players_ids": [],
        "teams_rosters": {},
        "draft_order": [],
        "current_pick_idx": 0,
        "current_pick_number": 1,
        "_overridden_team_id": 12,
    }
    core_draft_state.load_from_serialized(serialized, lambda payload: payload)

    assert core_draft_state.get_overridden_team_id() == 12
