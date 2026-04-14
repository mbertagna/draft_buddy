import pytest
from draft_buddy.core.draft_state import DraftState
from draft_buddy.domain.entities import Player

@pytest.fixture
def basic_state():
    return DraftState(
        all_player_ids={1, 2, 3},
        draft_order=[1, 2, 1],
        roster_structure={'QB': 1, 'RB': 1, 'WR': 1, 'TE': 0, 'FLEX': 1},
        bench_maxes={'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0},
        total_roster_size_per_team=3
    )

def test_add_player_to_roster(basic_state):
    player = Player(1, "Player 1", "RB", 100.0)
    basic_state.add_player_to_roster(1, player)
    
    rosters = basic_state.get_rosters()
    assert len(rosters[1]['PLAYERS']) == 1
    assert rosters[1]['RB'] == 1
    assert 1 not in basic_state.get_available_player_ids()

def test_flex_logic(basic_state):
    # RB1 fills RB slot
    p1 = Player(1, "RB1", "RB", 100.0)
    basic_state.add_player_to_roster(1, p1)
    # RB2 should fill FLEX slot
    p2 = Player(2, "RB2", "RB", 90.0)
    basic_state.add_player_to_roster(1, p2)
    
    roster = basic_state.get_rosters()[1]
    assert roster['RB'] == 1
    assert roster['FLEX'] == 1

def test_undo_last_pick(basic_state):
    p1 = Player(1, "RB1", "RB", 100.0)
    basic_state.add_player_to_roster(1, p1)
    
    # Manually simulate history entry since DraftState is a container
    basic_state.append_draft_history({'player_id': 1, 'team_id': 1})
    
    # Undo
    history = basic_state.pop_draft_history()
    basic_state.remove_player_from_roster(history['team_id'], p1)
    
    roster = basic_state.get_rosters()[1]
    assert len(roster['PLAYERS']) == 0
    assert roster['RB'] == 0
    assert 1 in basic_state.get_available_player_ids()
