"""Tests for shared dynamic value calculations."""

from __future__ import annotations

import pytest

from draft_buddy.core import DraftState, Player, PlayerCatalog
from draft_buddy.core.value_service import (
    calculate_dynamic_player_value,
    get_flex_aware_positional_baselines,
    insight_alignment_from_payload,
    next_pick_number_for_team,
    survival_probability_to_pick,
)


def _player(player_id: int, position: str, points: float, adp: float = 10.0) -> Player:
    """Build a minimal test player."""
    return Player(
        player_id=player_id,
        name=f"{position} {player_id}",
        position=position,
        projected_points=points,
        adp=adp,
        stability_score=80.0,
    )


def test_baseline_uses_first_player_beyond_dedicated_demand() -> None:
    """Verify replacement rank is the first player after remaining demand."""
    players = [_player(idx + 1, "RB", points) for idx, points in enumerate([100, 90, 80, 70, 60])]
    state = DraftState(
        all_player_ids={player.player_id for player in players},
        draft_order=[1, 2],
        roster_structure={"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
        bench_maxes={"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        total_roster_size_per_team=1,
    )

    baselines = get_flex_aware_positional_baselines(state, PlayerCatalog(players), positions=("RB",))

    assert baselines["RB"] == pytest.approx(80.0)


def test_baseline_allocates_flex_demand_to_best_remaining_flex_pool() -> None:
    """Verify FLEX demand lowers the chosen position baseline after dedicated slots."""
    players = [
        _player(1, "RB", 100.0),
        _player(2, "RB", 50.0),
        _player(3, "WR", 90.0),
        _player(4, "WR", 45.0),
    ]
    state = DraftState(
        all_player_ids={player.player_id for player in players},
        draft_order=[1],
        roster_structure={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 1},
        bench_maxes={"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        total_roster_size_per_team=1,
    )

    baselines = get_flex_aware_positional_baselines(
        state,
        PlayerCatalog(players),
        positions=("RB", "WR"),
    )

    assert baselines["RB"] == pytest.approx((100.0 + 50.0 + 50.0) / 3.0)


def test_survival_probability_declines_for_later_target_pick() -> None:
    """Verify next-pick survival decreases as the target pick moves later."""
    near = survival_probability_to_pick(24.0, current_pick=10, target_pick=20)
    far = survival_probability_to_pick(24.0, current_pick=10, target_pick=40)

    assert near > far


def test_next_pick_number_for_team_uses_one_based_pick_number() -> None:
    """Verify next-pick lookup skips the current pick."""
    assert next_pick_number_for_team([1, 2, 3, 3, 2, 1], 0, 1) == 6


def test_dynamic_value_includes_wait_loss_for_unlikely_survivor() -> None:
    """Verify scarce players unlikely to survive receive wait-loss value."""
    players = [
        _player(1, "RB", 100.0, adp=2.0),
        _player(2, "RB", 50.0, adp=60.0),
        _player(3, "RB", 45.0, adp=70.0),
    ]
    state = DraftState(
        all_player_ids={player.player_id for player in players},
        draft_order=[1, 2, 2, 1],
        roster_structure={"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0},
        bench_maxes={"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        total_roster_size_per_team=1,
    )

    value = calculate_dynamic_player_value(players[0], state, PlayerCatalog(players), team_id=1)

    assert value.wait_loss_points > 0.0
    assert value.draft_now_score > value.dynamic_vorp


def test_insight_alignment_flags_confirming_and_contradicting_evidence() -> None:
    """Verify insight alignment maps structured evidence into concise labels."""
    confirm = {
        "depth_role": "starter",
        "playing_time_tier": "high",
        "overall_confidence": "high",
        "tags": [],
    }
    contradict = {
        "depth_role": "backup",
        "playing_time_tier": "low",
        "injury_risk": "high",
        "tags": ["role_reduction"],
    }

    assert insight_alignment_from_payload(confirm) == "confirm"
    assert insight_alignment_from_payload(contradict) == "contradict"
