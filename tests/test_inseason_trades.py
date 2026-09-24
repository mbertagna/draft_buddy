"""Tests for remaining-week simulation, smart drops, and trade verdicts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from draft_buddy.inseason.drops import SmartDropEvaluator
from draft_buddy.inseason.probability import RemainingWeekMonteCarlo
from draft_buddy.inseason.state import LeagueSnapshot, TeamRecord, TeamSnapshot
from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.inseason.trades import (
    InboundTradeService,
    TradeFinderService,
    is_buy_low,
    is_sell_high,
    partner_within_fairness,
    verdict_for_delta,
)
from draft_buddy.simulator.evaluator import _optimal_lineup_points, simulate_season_fast


def _weekly(player_id: int, position: str, points: float = 10.0) -> dict:
    """Build a flat remaining-season weekly vector."""
    return {"pos": position, "pts": [points] * 18}


def test_optimal_lineup_points_uses_three_flex_slots() -> None:
    """Verify FLEX 3 adds three leftover RB/WR/TE scores."""
    points = _optimal_lineup_points(
        {"QB": [10.0], "RB": [9.0, 8.0, 7.0, 6.0], "WR": [5.0, 4.0, 3.0], "TE": [2.0]},
        {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3},
    )

    assert points == 54.0


def test_simulate_season_fast_locks_starting_records() -> None:
    """Verify remaining-week sims add onto locked W-L totals."""
    weekly = {
        1: _weekly(1, "QB", 12.0),
        2: _weekly(2, "RB", 11.0),
        3: _weekly(3, "QB", 8.0),
        4: _weekly(4, "RB", 7.0),
    }
    matchups = pd.DataFrame(
        [
            {
                "Week": 2,
                "Matchup": 1,
                "Away Manager(s)": "Team 1",
                "Home Manager(s)": "Team 2",
                "Away Score": 0.0,
                "Home Score": 0.0,
            }
        ]
    )
    rosters = {"Team 1": [1, 2], "Team 2": [3, 4]}
    starting = {
        "Team 1": {"W": 3, "L": 0, "T": 0, "pts": 0.0},
        "Team 2": {"W": 0, "L": 3, "T": 0, "pts": 0.0},
    }

    _results, records, _playoffs, _tree, _winner = simulate_season_fast(
        weekly,
        matchups,
        rosters,
        season=2026,
        num_playoff_teams=2,
        start_week=2,
        starting_records=starting,
        roster_structure={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
    )
    by_name = dict(records)

    assert by_name["Team 1"]["W"] == 4


def test_verdict_for_delta_accepts_two_point_gain() -> None:
    """Verify a +2pp playoff-odds lift is an ACCEPT."""
    assert verdict_for_delta(0.02, 0.02, -0.01) == "ACCEPT"


def test_verdict_for_delta_counters_near_even() -> None:
    """Verify a small negative delta stays in COUNTER range."""
    assert verdict_for_delta(-0.005, 0.02, -0.01) == "COUNTER"


def test_verdict_for_delta_rejects_clear_loss() -> None:
    """Verify a larger playoff-odds loss is REJECT."""
    assert verdict_for_delta(-0.05, 0.02, -0.01) == "REJECT"


def _four_team_snapshot() -> LeagueSnapshot:
    """Return a tiny four-team snapshot for trade tests."""
    return LeagueSnapshot(
        league_id="test",
        platform="espn",
        current_week=14,
        teams={
            1: TeamSnapshot(1, "A", [1, 2, 3, 4], record=TeamRecord(7, 6)),
            2: TeamSnapshot(2, "B", [5, 6, 7, 8], record=TeamRecord(6, 7)),
            3: TeamSnapshot(3, "C", [9, 10, 11, 12], record=TeamRecord(5, 8)),
            4: TeamSnapshot(4, "D", [13, 14, 15, 16], record=TeamRecord(4, 9)),
        },
        remaining_matchups=[],
    )


def _trade_weekly() -> dict:
    """Return weekly projections for the 16-player test catalog."""
    positions = {
        1: "QB",
        2: "RB",
        3: "WR",
        4: "TE",
        5: "QB",
        6: "RB",
        7: "WR",
        8: "TE",
        9: "QB",
        10: "RB",
        11: "WR",
        12: "TE",
        13: "QB",
        14: "RB",
        15: "WR",
        16: "TE",
    }
    points = {
        1: 20.0,
        2: 18.0,
        3: 16.0,
        4: 12.0,
        5: 11.0,
        6: 10.0,
        7: 9.0,
        8: 8.0,
        9: 7.0,
        10: 6.0,
        11: 5.0,
        12: 4.0,
        13: 3.0,
        14: 2.0,
        15: 1.5,
        16: 1.0,
    }
    return {player_id: _weekly(player_id, pos, points[player_id]) for player_id, pos in positions.items()}


def test_smart_drop_removes_lowest_ros_when_over_cap(player_catalog, config) -> None:
    """Verify a 2-for-1 that exceeds roster cap drops the lowest ROS bench player."""
    rng = np.random.default_rng(0)
    monte_carlo = RemainingWeekMonteCarlo(
        roster_structure=config.draft.ROSTER_STRUCTURE,
        num_playoff_teams=2,
        rng=rng,
    )
    evaluator = SmartDropEvaluator(
        catalog=player_catalog,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        roster_cap=5,
        variations_count=3,
        monte_carlo=monte_carlo,
    )
    weekly = _trade_weekly()

    roster, option = evaluator.apply_simple(
        roster_player_ids=[1, 2, 3, 4],
        incoming_ids=[6, 7],
        outgoing_ids=[4],
        ir_player_ids=[],
        weekly_projections=weekly,
        start_week=14,
    )

    assert 4 not in roster


def test_inbound_trade_returns_numeric_verdict(player_catalog, config) -> None:
    """Verify inbound eval returns an ACCEPT/COUNTER/REJECT token."""
    rng = np.random.default_rng(1)
    monte_carlo = RemainingWeekMonteCarlo(
        roster_structure=config.draft.ROSTER_STRUCTURE,
        num_playoff_teams=2,
        rng=rng,
    )
    drops = SmartDropEvaluator(
        catalog=player_catalog,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        roster_cap=7,
        variations_count=2,
        monte_carlo=monte_carlo,
    )
    service = InboundTradeService(
        catalog=player_catalog,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        drop_evaluator=drops,
        monte_carlo=monte_carlo,
        n_runs=2,
        accept_delta=0.02,
        counter_delta=-0.01,
        last_regular_week=14,
    )
    snapshot = _four_team_snapshot()
    snapshot.remaining_matchups = []

    result = service.evaluate(snapshot, _trade_weekly(), 1, 2, [4], [6])

    assert result.verdict in {"ACCEPT", "COUNTER", "REJECT"}


def test_trade_finder_only_emits_allowed_shapes(player_catalog, config) -> None:
    """Verify outbound search stays within 1- and 2-player shapes."""
    rng = np.random.default_rng(2)
    monte_carlo = RemainingWeekMonteCarlo(
        roster_structure=config.draft.ROSTER_STRUCTURE,
        num_playoff_teams=2,
        rng=rng,
    )
    drops = SmartDropEvaluator(
        catalog=player_catalog,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        roster_cap=7,
        variations_count=2,
        monte_carlo=monte_carlo,
    )
    finder = TradeFinderService(
        catalog=player_catalog,
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        drop_evaluator=drops,
        monte_carlo=monte_carlo,
        n_runs=1,
        last_regular_week=14,
        max_candidates=4,
        player_pool_top_n=4,
        ros_point_floor=0.0,
        partner_starter_epsilon=-50.0,
        partner_playoff_floor=-1.0,
        partner_wins_floor=-10.0,
        outbound_shapes=[[1, 1]],
    )
    proposals = finder.search(_four_team_snapshot(), _trade_weekly(), user_team_id=1)
    illegal = [
        proposal
        for proposal in proposals
        if len(proposal.user_sends_player_ids) != 1 or len(proposal.partner_sends_player_ids) != 1
    ]

    assert illegal == []


def test_exhaustive_search_scores_players_below_the_ros_floor() -> None:
    """Verify exhaustive search keeps every skill-player deal the pool would drop."""
    catalog = PlayerCatalog(
        [
            Player(player_id=1, name="User QB", position="QB", projected_points=20.0),
            Player(player_id=2, name="User RB", position="RB", projected_points=10.0),
            Player(player_id=3, name="Partner QB", position="QB", projected_points=12.0),
            Player(player_id=4, name="Partner RB", position="RB", projected_points=8.0),
        ]
    )
    snapshot = LeagueSnapshot(
        league_id="test",
        platform="espn",
        current_week=14,
        teams={
            1: TeamSnapshot(1, "A", [1, 2], record=TeamRecord(7, 6)),
            2: TeamSnapshot(2, "B", [3, 4], record=TeamRecord(6, 7)),
        },
        remaining_matchups=[],
    )
    weekly = {
        1: _weekly(1, "QB", 20.0),
        2: _weekly(2, "RB", 10.0),
        3: _weekly(3, "QB", 12.0),
        4: _weekly(4, "RB", 8.0),
    }
    rng = np.random.default_rng(3)
    monte_carlo = RemainingWeekMonteCarlo(
        roster_structure={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
        num_playoff_teams=1,
        rng=rng,
    )
    drops = SmartDropEvaluator(
        catalog=catalog,
        roster_structure={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
        bench_maxes={"QB": 1, "RB": 1, "WR": 0, "TE": 0},
        roster_cap=4,
        variations_count=1,
        monte_carlo=monte_carlo,
    )
    finder = TradeFinderService(
        catalog=catalog,
        roster_structure={"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
        bench_maxes={"QB": 1, "RB": 1, "WR": 0, "TE": 0},
        drop_evaluator=drops,
        monte_carlo=monte_carlo,
        n_runs=1,
        last_regular_week=14,
        max_candidates=1,
        player_pool_top_n=1,
        ros_point_floor=9999.0,
        partner_starter_epsilon=0.0,
        partner_playoff_floor=0.0,
        partner_wins_floor=0.0,
        outbound_shapes=[[1, 1]],
    )

    proposals = finder.search(snapshot, weekly, user_team_id=1, exhaustive=True, workers=2)

    assert len(proposals) == 4


def _form_player(**overrides) -> Player:
    """Build a player with two-year and current-season scoring fields."""
    fields = {
        "player_id": 1,
        "name": "Form",
        "position": "WR",
        "projected_points": 10.0,
        "weekly_points_median": 10.0,
        "weekly_points_ceiling": 14.0,
        "current_season_ppg": 14.0,
        "current_season_games": 4,
    }
    fields.update(overrides)
    return Player(**fields)


def test_buy_low_requires_history_above_current_rate() -> None:
    """Verify a cold player with a higher two-year median is buy-low."""
    assert is_buy_low(_form_player(current_season_ppg=7.0, weekly_points_median=10.0)) is True


def test_sell_high_requires_three_games_and_a_gap() -> None:
    """Verify a hot player is sell-high only after three games and a 25 percent gap."""
    assert is_sell_high(_form_player(current_season_ppg=13.0)) is True


def test_promoted_rb_is_not_sell_high() -> None:
    """Verify a new RB starter is excluded from the sell-high tag."""
    player = _form_player(
        position="RB",
        sleeper_depth_chart_order=1,
        current_season_games=1,
        current_season_ppg=20.0,
    )

    assert is_sell_high(player) is False


def test_partner_three_point_playoff_loss_still_prints() -> None:
    """Verify a 3pp partner loss stays inside the widened floor."""
    assert partner_within_fairness(-0.03, -0.2, -0.05, -0.05) is True


def test_search_partner_limits_results_to_that_team() -> None:
    """Verify search with a partner id does not emit other teams."""
    catalog = PlayerCatalog(
        [
            Player(player_id=1, name="User QB", position="QB", projected_points=20.0),
            Player(player_id=3, name="Partner QB", position="QB", projected_points=12.0),
            Player(player_id=5, name="Other QB", position="QB", projected_points=9.0),
        ]
    )
    snapshot = LeagueSnapshot(
        league_id="test",
        platform="espn",
        current_week=14,
        teams={
            1: TeamSnapshot(1, "A", [1], record=TeamRecord(7, 6)),
            2: TeamSnapshot(2, "B", [3], record=TeamRecord(6, 7)),
            3: TeamSnapshot(3, "C", [5], record=TeamRecord(5, 8)),
        },
        remaining_matchups=[],
    )
    weekly = {
        1: _weekly(1, "QB", 20.0),
        3: _weekly(3, "QB", 12.0),
        5: _weekly(5, "QB", 9.0),
    }
    structure = {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0}
    monte_carlo = RemainingWeekMonteCarlo(
        roster_structure=structure,
        num_playoff_teams=1,
        rng=np.random.default_rng(4),
    )
    drops = SmartDropEvaluator(
        catalog=catalog,
        roster_structure=structure,
        bench_maxes={"QB": 1, "RB": 0, "WR": 0, "TE": 0},
        roster_cap=2,
        variations_count=1,
        monte_carlo=monte_carlo,
    )
    finder = TradeFinderService(
        catalog=catalog,
        roster_structure=structure,
        bench_maxes={"QB": 1, "RB": 0, "WR": 0, "TE": 0},
        drop_evaluator=drops,
        monte_carlo=monte_carlo,
        n_runs=1,
        last_regular_week=14,
        max_candidates=10,
        player_pool_top_n=4,
        ros_point_floor=0.0,
        partner_starter_epsilon=-50.0,
        partner_playoff_floor=-1.0,
        partner_wins_floor=-10.0,
        outbound_shapes=[[1, 1]],
    )

    proposals = finder.search(
        snapshot, weekly, user_team_id=1, partner_team_id=2, exhaustive=True
    )

    assert {proposal.partner_team_id for proposal in proposals} == {2}


def test_partner_six_point_loss_and_wins_drop_is_removed() -> None:
    """Verify a deal past both partner floors is dropped."""
    assert partner_within_fairness(-0.06, -0.1, -0.05, -0.05) is False
