"""Tests for core bot GM strategy behavior."""

from draft_buddy.core.bot_gm import AdpBotGM, HeuristicBotGM, RandomBotGM
from draft_buddy.domain.entities import Player


def _allow_all(team_id: int, position: str, is_manual: bool = False) -> bool:
    """Allow all positions in tests."""
    return True


def _unused_try_select(team_id: int, position: str, available_ids: set):
    """Unused placeholder callback for strategy interface."""
    return False, None


def test_random_bot_returns_player_when_eligible_exists():
    """Random bot should choose an eligible player."""
    players = {1: Player(1, "A", "RB", 10.0), 2: Player(2, "B", "WR", 9.0)}
    bot = RandomBotGM()
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )
    assert chosen is not None


def test_adp_bot_prefers_lowest_adp_when_no_randomness():
    """ADP bot should pick lowest ADP when randomness disabled."""
    players = {
        1: Player(1, "A", "RB", 10.0, adp=20.0),
        2: Player(2, "B", "RB", 9.0, adp=10.0),
    }
    bot = AdpBotGM(randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )
    assert chosen.player_id == 2


def test_heuristic_bot_fills_priority_starter_first():
    """Heuristic bot should fill highest-priority missing starter."""
    players = {
        1: Player(1, "QB One", "QB", 20.0, adp=40.0),
        2: Player(2, "RB One", "RB", 15.0, adp=20.0),
    }
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )
    assert chosen.position == "QB"
