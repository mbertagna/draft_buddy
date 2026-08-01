"""Additional focused tests for core helper behavior."""

from __future__ import annotations

import logging

import pytest

from draft_buddy.core import AdpBotGM, HeuristicBotGM, RandomBotGM, create_bot_gm
from draft_buddy.core.roster_utils import calculate_roster_scores, categorize_roster_by_slots
from draft_buddy.core.stacking import calculate_stack_count


def test_random_bot_returns_none_when_no_eligible_players(player_catalog) -> None:
    """Verify RandomBotGM returns None when the pool is empty."""
    bot = RandomBotGM()

    assert bot.execute_pick(1, set(), player_catalog, None, {}, {}, lambda *_args: True, None) is None


def test_adp_bot_prefers_lowest_adp_when_not_random(player_catalog) -> None:
    """Verify AdpBotGM takes the best ADP player on the deterministic path."""
    bot = AdpBotGM(randomness_factor=0.0)
    player = bot.execute_pick(
        1,
        {1, 5},
        player_catalog,
        None,
        {"QB": 1},
        {},
        lambda *_args: True,
        None,
    )

    assert player.player_id == 1


def test_adp_bot_random_eligible_strategy_can_choose_non_best(monkeypatch, player_catalog) -> None:
    """Verify RANDOM_ELIGIBLE branch delegates to random.choice."""
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="RANDOM_ELIGIBLE")
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.choice", lambda players: players[-1])

    assert bot.execute_pick(1, {1, 5}, player_catalog, None, {"QB": 1}, {}, lambda *_args: True, None).player_id == 5


def test_adp_bot_next_best_adp_returns_second_ranked_player(player_catalog) -> None:
    """Verify NEXT_BEST_ADP chooses the second-best ADP option on the random path."""
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="NEXT_BEST_ADP")

    player = bot.execute_pick(1, {1, 5}, player_catalog, None, {"QB": 1}, {}, lambda *_args: True, None)

    assert player.player_id == 5


def test_adp_bot_next_best_heuristic_chooses_non_best_player(monkeypatch, player_catalog) -> None:
    """Verify NEXT_BEST_HEURISTIC samples from the non-best eligible players."""
    bot = AdpBotGM(randomness_factor=1.0, suboptimal_strategy="NEXT_BEST_HEURISTIC")
    monkeypatch.setattr("random.choice", lambda players: players[0])

    player = bot.execute_pick(
        1,
        {1, 5, 9},
        player_catalog,
        None,
        {"QB": 1},
        {},
        lambda *_args: True,
        None,
    )

    assert player.player_id == 5


def test_heuristic_bot_falls_back_to_adp_when_no_priority_path(player_catalog) -> None:
    """Verify heuristic bot falls back to ADP when roster is full at each priority slot."""
    bot = HeuristicBotGM(positional_priority=["RB", "WR", "QB", "TE"], randomness_factor=0.0)

    class FullRoster:
        def position_count(self, _position: str) -> int:
            return 9

    player = bot.execute_pick(
        1,
        {2, 3, 4},
        player_catalog,
        FullRoster(),
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0},
        {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        lambda *_args: True,
        None,
    )

    assert player.player_id == 2


def test_heuristic_bot_prioritizes_missing_starter_slot(player_catalog) -> None:
    """Verify heuristic bot fills the highest-priority missing starter before bench value."""
    bot = HeuristicBotGM(positional_priority=["RB", "WR", "QB", "TE"], randomness_factor=0.0)

    class StarterNeedRoster:
        def position_count(self, position: str) -> int:
            return {"RB": 0, "WR": 1, "QB": 1, "TE": 1, "FLEX": 0}.get(position, 0)

    player = bot.execute_pick(
        1,
        {1, 2, 3},
        player_catalog,
        team_roster=StarterNeedRoster(),
        roster_structure={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 0, "RB": 1, "WR": 1, "TE": 0},
        can_draft_position_fn=lambda *_args: True,
        try_select_player_fn=None,
    )

    assert player.player_id == 2


def test_heuristic_bot_random_next_best_adp_returns_second_by_adp(player_catalog) -> None:
    """Verify heuristic NEXT_BEST_ADP uses the ordered ADP fallback list."""
    bot = HeuristicBotGM(
        positional_priority=["RB", "WR", "QB", "TE"],
        randomness_factor=1.0,
        suboptimal_strategy="NEXT_BEST_ADP",
    )

    class BalancedRoster:
        def position_count(self, position: str) -> int:
            return {"RB": 1, "WR": 1, "QB": 1, "TE": 1, "FLEX": 1}.get(position, 0)

    player = bot.execute_pick(
        1,
        {2, 3, 4},
        player_catalog,
        BalancedRoster(),
        {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
        {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        lambda *_args: True,
        None,
    )

    assert player.player_id == 3


def test_create_bot_gm_returns_random_bot() -> None:
    """Verify RANDOM strategy maps to RandomBotGM."""
    assert isinstance(create_bot_gm("RANDOM", {}), RandomBotGM)


def test_create_bot_gm_defaults_to_heuristic_bot() -> None:
    """Verify unknown strategy names fall back to the heuristic bot."""
    assert isinstance(create_bot_gm("UNKNOWN", {}), HeuristicBotGM)


def test_categorize_roster_by_slots_assigns_best_remaining_flex(player_factory) -> None:
    """Verify flex slots are filled from the highest-scoring remaining RB/WR/TE player."""
    roster = [
        player_factory(1, "RB", 20.0),
        player_factory(2, "RB", 18.0),
        player_factory(3, "WR", 19.0),
        player_factory(4, "WR", 17.0),
        player_factory(5, "TE", 15.0),
    ]
    starters, _bench, flex_players = categorize_roster_by_slots(
        roster,
        {"QB": 0, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
        {"QB": 0, "RB": 1, "WR": 1, "TE": 0},
    )

    assert starters["FLEX"][0].player_id == 2 and flex_players[0].player_id == 2


def test_calculate_roster_scores_logs_unassignable_bench_player(caplog, player_factory) -> None:
    """Verify an over-limit bench player emits the warning branch."""
    roster = [
        player_factory(1, "QB", 20.0),
        player_factory(2, "QB", 10.0),
        player_factory(3, "QB", 5.0),
    ]
    with caplog.at_level(logging.WARNING):
        scores = calculate_roster_scores(
            roster,
            {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
            {"QB": 1, "RB": 0, "WR": 0, "TE": 0},
        )

    assert scores["bench_total_points"] == 15.0 and "could not be optimally assigned" in caplog.text


def test_calculate_stack_count_ignores_players_without_team(player_factory) -> None:
    """Verify stack counting ignores players without a team association."""
    roster = [
        player_factory(1, "QB", team="BUF"),
        player_factory(2, "WR", team="BUF"),
        player_factory(3, "TE", team=None),
    ]

    assert calculate_stack_count(roster) == 1
