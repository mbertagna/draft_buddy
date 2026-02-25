import pytest
from draft_buddy.utils.data_utils import Player, calculate_stack_count

def test_calculate_stack_count():
    """Verify QB-WR/TE stacking logic."""
    p1 = Player(1, "QB1", "QB", 300.0, team="BUF")
    p2 = Player(2, "WR1", "WR", 200.0, team="BUF")
    p3 = Player(3, "WR2", "WR", 150.0, team="MIA")
    p4 = Player(4, "TE1", "TE", 100.0, team="BUF")
    
    # 1 QB + 1 WR + 1 TE from same team = 2 stacks
    assert calculate_stack_count([p1, p2, p4]) == 2
    
    # No matches
    assert calculate_stack_count([p1, p3]) == 0
    
    # Empty
    assert calculate_stack_count([]) == 0
