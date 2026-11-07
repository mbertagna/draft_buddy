"""Sliding-scale rest-of-season weekly projection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

WEEKS_IN_SEASON = 18
ROOKIE_GAMES_TOKEN = "R"


@dataclass(frozen=True)
class PlayerProjectionInput:
    """Per-player inputs for a rest-of-season weekly vector.

    Parameters
    ----------
    player_id : int
        Catalog player id.
    position : str
        Skill position code.
    historical_ppg : float
        Per-game median from lookback seasons, or preseason value for rookies.
    current_ppg : float or None
        Per-game mean of completed current-season games.
    current_games_played : int
        Completed current-season games used in ``current_ppg``.
    bye_week : int or None
        NFL bye week.
    games_played_frac : float or str
        Catalog games-played fraction, or ``\"R\"`` for rookies.
    depth_chart_order : int or None
        Sleeper depth-chart order when known.
    """

    player_id: int
    position: str
    historical_ppg: float
    current_ppg: Optional[float] = None
    current_games_played: int = 0
    bye_week: Optional[int] = None
    games_played_frac: float | str = 1.0
    depth_chart_order: Optional[int] = None


class SlidingScaleProjectionEngine:
    """Blend historical and current-season scoring into weekly ROS vectors.

    Parameters
    ----------
    weights : sequence of mapping
        Week-range buckets with ``min_week``, ``max_week``, ``w_hist``, ``w_curr``.
    min_current_games_for_full_weight : int
        Games required before current-season weight may exceed weeks 1-3.
    depth_chart_curr_weight_cap : float
        Maximum ``W_curr`` for recently promoted running backs.
    depth_chart_starter_games : int
        Games as listed starter before the RB cap is removed.
    """

    def __init__(
        self,
        weights: Sequence[Mapping[str, float]],
        min_current_games_for_full_weight: int = 3,
        depth_chart_curr_weight_cap: float = 0.70,
        depth_chart_starter_games: int = 3,
    ) -> None:
        self._weights = [dict(bucket) for bucket in weights]
        self._min_current_games = int(min_current_games_for_full_weight)
        self._depth_chart_cap = float(depth_chart_curr_weight_cap)
        self._depth_chart_games = int(depth_chart_starter_games)

    def project_player(
        self,
        player: PlayerProjectionInput,
        current_week: int,
    ) -> Dict[str, object]:
        """Return one simulator-shaped weekly projection.

        Parameters
        ----------
        player : PlayerProjectionInput
            Blend inputs for one player.
        current_week : int
            First remaining NFL week.

        Returns
        -------
        dict
            Mapping with ``pos`` and 18 weekly ``pts`` values.
        """
        w_hist, w_curr = self.effective_weights(player, current_week)
        hist_ppg = float(player.historical_ppg)
        if player.current_ppg is None or player.current_games_played <= 0:
            ppg = hist_ppg
        else:
            ppg = (w_hist * hist_ppg) + (w_curr * float(player.current_ppg))
        points = _weekly_points(ppg, player.bye_week, current_week)
        return {"pos": player.position, "pts": points}

    def project_players(
        self,
        players: Iterable[PlayerProjectionInput],
        current_week: int,
    ) -> Dict[int, Dict[str, object]]:
        """Project every supplied player.

        Parameters
        ----------
        players : iterable of PlayerProjectionInput
            Blend inputs.
        current_week : int
            First remaining NFL week.

        Returns
        -------
        dict
            Weekly projections keyed by player id.
        """
        return {
            player.player_id: self.project_player(player, current_week)
            for player in players
        }

    def project_current_season(
        self,
        players: Iterable[PlayerProjectionInput],
        current_week: int,
    ) -> Tuple[Dict[int, Dict[str, object]], List[int]]:
        """Project remaining weeks from this season only.

        Players with a current-season rate use that rate. Players without one
        keep the historical rate and are listed as fallbacks.

        Parameters
        ----------
        players : iterable of PlayerProjectionInput
            Blend inputs.
        current_week : int
            First remaining NFL week.

        Returns
        -------
        tuple
            Weekly projections, then player ids that used the historical rate.
        """
        weekly: Dict[int, Dict[str, object]] = {}
        fallback_ids: List[int] = []
        for player in players:
            points, used_history = _current_season_points(player, current_week)
            weekly[player.player_id] = {"pos": player.position, "pts": points}
            if used_history:
                fallback_ids.append(int(player.player_id))
        return weekly, fallback_ids

    def effective_weights(
        self,
        player: PlayerProjectionInput,
        current_week: int,
    ) -> Tuple[float, float]:
        """Return ``(W_hist, W_curr)`` after sample and depth-chart rules.

        Parameters
        ----------
        player : PlayerProjectionInput
            Blend inputs.
        current_week : int
            Active NFL week.

        Returns
        -------
        tuple of float
            Historical and current-season weights.
        """
        w_hist, w_curr = blend_weights_for_week(current_week, self._weights)
        early_hist, early_curr = blend_weights_for_week(1, self._weights)
        if player.current_games_played < self._min_current_games and w_curr > early_curr:
            w_curr = early_curr
            w_hist = early_hist
        if (
            player.position == "RB"
            and player.depth_chart_order == 1
            and player.current_games_played < self._depth_chart_games
        ):
            w_curr = min(self._depth_chart_cap, max(w_curr, self._depth_chart_cap))
            w_hist = 1.0 - w_curr
        return w_hist, w_curr


def blend_weights_for_week(
    current_week: int,
    weights: Sequence[Mapping[str, float]],
) -> Tuple[float, float]:
    """Return hist/curr weights for an NFL week.

    Parameters
    ----------
    current_week : int
        Active NFL week.
    weights : sequence of mapping
        Configured week-range buckets.

    Returns
    -------
    tuple of float
        ``(W_hist, W_curr)``.
    """
    week = int(current_week)
    for bucket in weights:
        if int(bucket["min_week"]) <= week <= int(bucket["max_week"]):
            return float(bucket["w_hist"]), float(bucket["w_curr"])
    return 0.10, 0.90


def remaining_ros_total(
    weekly_points: Sequence[float],
    start_week: int,
    last_week: int = WEEKS_IN_SEASON,
) -> float:
    """Sum projected points from ``start_week`` through ``last_week``.

    Parameters
    ----------
    weekly_points : sequence of float
        Eighteen weekly values.
    start_week : int
        First remaining week (1-based).
    last_week : int, optional
        Last week to include.

    Returns
    -------
    float
        Rest-of-season total.
    """
    start_index = max(0, int(start_week) - 1)
    end_index = min(len(weekly_points), int(last_week))
    return float(sum(weekly_points[start_index:end_index]))


def ros_totals_from_weekly(
    weekly_projections: Mapping[int, Mapping[str, object]],
    start_week: int,
    last_week: int = WEEKS_IN_SEASON,
) -> Dict[int, float]:
    """Map player ids to remaining-season point totals.

    Parameters
    ----------
    weekly_projections : mapping
        Simulator-shaped weekly projections.
    start_week : int
        First remaining week.
    last_week : int, optional
        Last week to include.

    Returns
    -------
    dict of int to float
        Remaining ROS totals.
    """
    totals: Dict[int, float] = {}
    for player_id, row in weekly_projections.items():
        pts = row.get("pts") or []
        totals[int(player_id)] = remaining_ros_total(list(pts), start_week, last_week)
    return totals


def inputs_from_catalog(
    catalog,
    current_week: int,
) -> List[PlayerProjectionInput]:
    """Build projection inputs from catalog history and the current season.

    Historical PPG prefers the two-year weekly median. Current PPG is the
    mean of completed current-season games. Players with no current-season
    games keep ``current_games_played`` at 0.

    Parameters
    ----------
    catalog : PlayerCatalog
        Loaded players.
    current_week : int
        Unused except to keep the call signature stable.

    Returns
    -------
    list of PlayerProjectionInput
        Catalog-derived blend inputs.
    """
    _ = current_week
    games = 17.0
    inputs: List[PlayerProjectionInput] = []
    for player in catalog:
        historical_ppg = player.weekly_points_median
        if historical_ppg is None and player.historical_weekly_median is not None:
            historical_ppg = float(player.historical_weekly_median)
        if historical_ppg is None:
            historical_ppg = float(player.projected_points) / games
        current_games = int(player.current_season_games or 0)
        current_ppg = player.current_season_ppg if current_games > 0 else None
        inputs.append(
            PlayerProjectionInput(
                player_id=player.player_id,
                position=player.position,
                historical_ppg=float(historical_ppg),
                current_ppg=None if current_ppg is None else float(current_ppg),
                current_games_played=current_games,
                bye_week=player.bye_week,
                games_played_frac=player.games_played_frac,
                depth_chart_order=player.sleeper_depth_chart_order,
            )
        )
    return inputs


def _current_season_points(
    player: PlayerProjectionInput,
    current_week: int,
) -> Tuple[List[float], bool]:
    """Return a weekly vector and whether it fell back to history.

    Parameters
    ----------
    player : PlayerProjectionInput
        Blend inputs for one player.
    current_week : int
        First remaining NFL week.

    Returns
    -------
    tuple
        Eighteen weekly points, and True when the historical rate was used.
    """
    if player.current_ppg is None or int(player.current_games_played) <= 0:
        return _weekly_points(float(player.historical_ppg), player.bye_week, current_week), True
    return _weekly_points(float(player.current_ppg), player.bye_week, current_week), False


def _weekly_points(
    ppg: float,
    bye_week: Optional[int],
    current_week: int,
) -> List[float]:
    """Fill an 18-week vector with PPG from the current week onward."""
    points = [0.0] * WEEKS_IN_SEASON
    bye = _safe_bye_week(bye_week)
    start = max(1, int(current_week))
    for week in range(start, WEEKS_IN_SEASON + 1):
        if bye is not None and week == bye:
            points[week - 1] = 0.0
        else:
            points[week - 1] = float(ppg)
    return points


def _safe_bye_week(bye_week: Optional[int]) -> Optional[int]:
    """Return a valid bye week number or ``None``."""
    if bye_week is None:
        return None
    try:
        if isinstance(bye_week, (float, np.floating)) and np.isnan(bye_week):
            return None
        week = int(bye_week)
    except (TypeError, ValueError):
        return None
    if week < 1 or week > WEEKS_IN_SEASON:
        return None
    return week
