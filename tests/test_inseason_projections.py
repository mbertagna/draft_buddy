"""Tests for sliding-scale ROS projections and catalog bye zeros."""

from __future__ import annotations

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.inseason.projections import (
    PlayerProjectionInput,
    SlidingScaleProjectionEngine,
    blend_weights_for_week,
    inputs_from_catalog,
    remaining_ros_total,
)

WEIGHTS = [
    {"min_week": 1, "max_week": 3, "w_hist": 0.70, "w_curr": 0.30},
    {"min_week": 4, "max_week": 6, "w_hist": 0.40, "w_curr": 0.60},
    {"min_week": 7, "max_week": 99, "w_hist": 0.10, "w_curr": 0.90},
]


def test_blend_weights_for_week_two_use_early_bucket() -> None:
    """Verify week 2 still uses the historical-heavy blend."""
    w_hist, w_curr = blend_weights_for_week(2, WEIGHTS)

    assert (w_hist, w_curr) == (0.70, 0.30)


def test_blend_weights_for_week_eight_use_late_bucket() -> None:
    """Verify week 8 uses current-season-heavy blend."""
    w_hist, w_curr = blend_weights_for_week(8, WEIGHTS)

    assert (w_hist, w_curr) == (0.10, 0.90)


def test_low_sample_caps_current_weight_at_early_bucket() -> None:
    """Verify sparse current-season games cannot use the week-7+ current weight."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=1,
        position="WR",
        historical_ppg=15.0,
        current_ppg=8.0,
        current_games_played=1,
    )

    w_hist, w_curr = engine.effective_weights(player, current_week=8)

    assert (w_hist, w_curr) == (0.70, 0.30)


def test_promoted_rb_caps_current_weight() -> None:
    """Verify a listed RB starter with few games caps W_curr at 0.70."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=2,
        position="RB",
        historical_ppg=8.0,
        current_ppg=18.0,
        current_games_played=1,
        depth_chart_order=1,
    )

    _w_hist, w_curr = engine.effective_weights(player, current_week=2)

    assert w_curr == 0.70


def test_rookie_uses_historical_ppg_as_baseline() -> None:
    """Verify rookies still emit a weekly vector from the preseason baseline."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=3,
        position="WR",
        historical_ppg=12.0,
        current_ppg=None,
        current_games_played=0,
        games_played_frac="R",
        bye_week=7,
    )

    weekly = engine.project_player(player, current_week=1)

    assert weekly["pts"][0] == 12.0


def test_projection_zeros_bye_week() -> None:
    """Verify bye weeks are zero in the ROS weekly vector."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=4,
        position="QB",
        historical_ppg=20.0,
        current_ppg=20.0,
        current_games_played=4,
        bye_week=7,
    )

    weekly = engine.project_player(player, current_week=1)

    assert weekly["pts"][6] == 0.0


def test_remaining_ros_total_skips_played_weeks() -> None:
    """Verify ROS totals start at the current week."""
    total = remaining_ros_total([10.0] * 18, start_week=15, last_week=17)

    assert total == 30.0


def test_player_catalog_weekly_projections_zero_bye_week(player_catalog) -> None:
    """Verify the draft-time weekly expansion zeros configured bye weeks."""
    weekly = player_catalog.to_weekly_projections()

    assert weekly[1]["pts"][6] == 0.0


def test_inputs_from_catalog_do_not_invent_current_games() -> None:
    """Verify a player with no 2026 games stays at zero current games."""
    catalog = PlayerCatalog(
        [
            Player(
                player_id=1,
                name="Quiet",
                position="WR",
                projected_points=170.0,
                weekly_points_median=10.0,
                current_season_games=0,
            )
        ]
    )

    inputs = inputs_from_catalog(catalog, current_week=2)

    assert inputs[0].current_games_played == 0


def test_inputs_from_catalog_uses_current_season_ppg() -> None:
    """Verify completed 2026 games replace the historical rate as current PPG."""
    catalog = PlayerCatalog(
        [
            Player(
                player_id=2,
                name="Hot",
                position="WR",
                projected_points=170.0,
                weekly_points_median=10.0,
                current_season_ppg=18.0,
                current_season_games=4,
            )
        ]
    )

    inputs = inputs_from_catalog(catalog, current_week=8)

    assert inputs[0].current_ppg == 18.0


def test_current_season_projection_uses_this_year_when_games_exist() -> None:
    """Verify a current-season vector ignores the historical rate when 2026 PPG exists."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=1,
        position="WR",
        historical_ppg=10.0,
        current_ppg=18.0,
        current_games_played=5,
    )

    weekly, fallback = engine.project_current_season([player], current_week=8)

    assert fallback == []
    assert weekly[1]["pts"][7] == 18.0


def test_current_season_projection_keeps_history_without_2026_games() -> None:
    """Verify a player with no 2026 games keeps the historical weekly rate."""
    engine = SlidingScaleProjectionEngine(WEIGHTS)
    player = PlayerProjectionInput(
        player_id=4,
        position="RB",
        historical_ppg=12.0,
        current_ppg=None,
        current_games_played=0,
    )

    weekly, fallback = engine.project_current_season([player], current_week=8)

    assert fallback == [4]
    assert weekly[4]["pts"][7] == 12.0
