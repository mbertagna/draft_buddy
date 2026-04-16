"""Tests for core bot GM strategy behavior."""

from draft_buddy.core.bot_gm import AdpBotGM, HeuristicBotGM, RandomBotGM, create_bot_gm
from draft_buddy.domain.entities import Player


def _allow_all(team_id: int, position: str, is_manual: bool = False) -> bool:
    """Allow all positions in tests."""
    return True


def _block_qb_only(team_id: int, position: str, is_manual: bool = False) -> bool:
    """Reject QB and allow all other positions."""
    return position != "QB"


def _unused_try_select(team_id: int, position: str, available_ids: set):
    """Unused placeholder callback for strategy interface."""
    return False, None


def test_random_bot_returns_player_when_eligible_exists():
    """Verify random bot returns a player from an eligible pool."""
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


def test_adp_bot_picks_lowest_eligible_adp_when_best_position_is_blocked():
    """Verify ADP bot skips blocked best ADP and takes next legal option."""
    players = {
        1: Player(1, "Best But Blocked", "QB", 18.0, adp=1.0),
        2: Player(2, "Next Best Legal", "WR", 17.0, adp=3.0),
        3: Player(3, "Third Best", "RB", 16.0, adp=4.0),
    }
    bot = AdpBotGM(randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2, 3},
        player_map=players,
        roster_counts={"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 0, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_block_qb_only,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 2


def test_adp_bot_uses_next_best_adp_strategy_when_random_path_triggered(monkeypatch):
    """Verify ADP bot uses NEXT_BEST_ADP fallback under randomness path."""
    players = {
        1: Player(1, "Best ADP", "RB", 20.0, adp=2.0),
        2: Player(2, "Next ADP", "RB", 19.0, adp=5.0),
        3: Player(3, "Third ADP", "RB", 18.0, adp=8.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="NEXT_BEST_ADP")
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2, 3},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 2


def test_heuristic_bot_uses_random_eligible_strategy_when_random_path_triggered(monkeypatch):
    """Verify heuristic bot uses RANDOM_ELIGIBLE fallback under randomness path."""
    players = {
        11: Player(11, "QB Candidate", "QB", 21.0, adp=15.0),
        12: Player(12, "RB Candidate", "RB", 20.5, adp=16.0),
        13: Player(13, "WR Candidate", "WR", 20.0, adp=17.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    monkeypatch.setattr(
        "draft_buddy.core.bot_gm.random.choice",
        lambda items: next(player for player in items if player.player_id == 13),
    )
    bot = HeuristicBotGM(
        positional_priority=["QB", "RB", "WR", "TE"],
        randomness_factor=1.0,
        suboptimal_strategy="RANDOM_ELIGIBLE",
    )
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={11, 12, 13},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 13


def test_adp_bot_returns_none_when_no_eligible_players_exist():
    """Verify ADP bot returns None for empty eligible pool."""
    players = {1: Player(1, "DST", "DST", 6.0, adp=200.0)}
    bot = AdpBotGM(randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen is None


def test_adp_bot_uses_random_eligible_strategy_when_selected(monkeypatch):
    """Verify ADP bot RANDOM_ELIGIBLE strategy delegates to random.choice."""
    players = {
        1: Player(1, "ADP A", "RB", 20.0, adp=2.0),
        2: Player(2, "ADP B", "RB", 19.0, adp=5.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.choice", lambda items: items[1])
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="RANDOM_ELIGIBLE")
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


def test_adp_bot_uses_next_best_heuristic_strategy_when_selected(monkeypatch):
    """Verify ADP bot NEXT_BEST_HEURISTIC chooses non-best option."""
    players = {
        1: Player(1, "Best ADP", "RB", 20.0, adp=2.0),
        2: Player(2, "Other One", "RB", 19.0, adp=5.0),
        3: Player(3, "Other Two", "RB", 18.0, adp=8.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.choice", lambda items: items[0])
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="NEXT_BEST_HEURISTIC")
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2, 3},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 2


def test_adp_bot_falls_back_to_best_when_next_best_adp_has_single_player(monkeypatch):
    """Verify NEXT_BEST_ADP strategy returns best when no second option exists."""
    players = {1: Player(1, "Only", "RB", 20.0, adp=2.0)}
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="NEXT_BEST_ADP")
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 1


def test_heuristic_bot_returns_none_when_no_eligible_players_exist():
    """Verify heuristic bot returns None when pool has no handled positions."""
    players = {1: Player(1, "Kicker", "K", 7.0, adp=190.0)}
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen is None


def test_heuristic_bot_falls_back_to_lowest_adp_when_heuristic_returns_none():
    """Verify heuristic bot falls back to ADP when no slot can improve."""
    players = {
        1: Player(1, "RB ADP Best", "RB", 20.0, adp=3.0),
        2: Player(2, "WR ADP Worse", "WR", 21.0, adp=7.0),
    }
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map=players,
        roster_counts={"QB": 2, "RB": 4, "WR": 4, "TE": 2, "FLEX": 1},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 1


def test_heuristic_bot_uses_next_best_adp_strategy_when_random_path_triggered(monkeypatch):
    """Verify heuristic bot NEXT_BEST_ADP branch returns second ADP player."""
    players = {
        1: Player(1, "ADP First", "RB", 20.0, adp=2.0),
        2: Player(2, "ADP Second", "RB", 19.0, adp=4.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    bot = HeuristicBotGM(
        positional_priority=["RB", "WR", "QB", "TE"],
        randomness_factor=1.0,
        suboptimal_strategy="NEXT_BEST_ADP",
    )
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


def test_heuristic_bot_uses_next_best_adp_strategy_single_player_returns_best(monkeypatch):
    """Verify heuristic NEXT_BEST_ADP keeps best with one eligible player."""
    players = {1: Player(1, "Solo", "RB", 20.0, adp=2.0)}
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    bot = HeuristicBotGM(
        positional_priority=["RB", "WR", "QB", "TE"],
        randomness_factor=1.0,
        suboptimal_strategy="NEXT_BEST_ADP",
    )
    chosen = bot.execute_pick(
        team_id=1,
        available_player_ids={1},
        player_map=players,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=_allow_all,
        try_select_player_fn=_unused_try_select,
    )

    assert chosen.player_id == 1


def test_heuristic_bot_uses_other_player_when_default_suboptimal_branch_triggered(monkeypatch):
    """Verify heuristic default suboptimal branch chooses among non-best players."""
    players = {
        1: Player(1, "Best", "RB", 21.0, adp=2.0),
        2: Player(2, "Other", "RB", 20.0, adp=3.0),
    }
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.random", lambda: 0.0)
    monkeypatch.setattr("draft_buddy.core.bot_gm.random.choice", lambda items: items[0])
    bot = HeuristicBotGM(
        positional_priority=["RB", "WR", "QB", "TE"],
        randomness_factor=1.0,
        suboptimal_strategy="UNSUPPORTED_STRATEGY",
    )
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


def test_heuristic_best_heuristic_uses_flex_when_primary_and_bench_limits_are_full():
    """Verify _best_heuristic selects FLEX-eligible player when FLEX is open."""
    players = [Player(1, "Flex RB", "RB", 19.0, adp=10.0)]
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot._best_heuristic(
        eligible=players,
        roster_counts={"QB": 2, "RB": 4, "WR": 4, "TE": 2, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    )

    assert chosen.player_id == 1


def test_heuristic_best_heuristic_returns_none_when_no_slots_can_accept_player():
    """Verify _best_heuristic returns None when all limits are exhausted."""
    players = [Player(2, "No Slot", "RB", 18.0, adp=12.0)]
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot._best_heuristic(
        eligible=players,
        roster_counts={"QB": 2, "RB": 4, "WR": 4, "TE": 2, "FLEX": 1},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    )

    assert chosen is None


def test_heuristic_best_heuristic_uses_bench_position_when_starters_are_full():
    """Verify _best_heuristic can select by bench capacity when starters are full."""
    players = [Player(30, "Bench WR", "WR", 16.0, adp=25.0)]
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot._best_heuristic(
        eligible=players,
        roster_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    )

    assert chosen.player_id == 30


def test_heuristic_best_heuristic_returns_none_when_flex_open_without_flex_candidates():
    """Verify _best_heuristic returns None if FLEX is open but no RB/WR/TE exists."""
    players = [Player(31, "QB Only", "QB", 16.0, adp=25.0)]
    bot = HeuristicBotGM(positional_priority=["QB", "RB", "WR", "TE"], randomness_factor=0.0)
    chosen = bot._best_heuristic(
        eligible=players,
        roster_counts={"QB": 2, "RB": 4, "WR": 4, "TE": 2, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    )

    assert chosen is None


def test_create_bot_gm_returns_random_bot_for_random_logic():
    """Verify factory returns RandomBotGM when requested."""
    bot = create_bot_gm("RANDOM", config={})

    assert isinstance(bot, RandomBotGM)


def test_create_bot_gm_returns_adp_bot_for_adp_logic():
    """Verify factory returns AdpBotGM when requested."""
    bot = create_bot_gm("ADP", config={"randomness_factor": 0.1})

    assert isinstance(bot, AdpBotGM)


def test_create_bot_gm_returns_heuristic_bot_for_default_logic():
    """Verify factory returns HeuristicBotGM for non-random/adp logic."""
    bot = create_bot_gm("HEURISTIC", config={})

    assert isinstance(bot, HeuristicBotGM)
