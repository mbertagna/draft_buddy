"""Shared player value calculations for drafting, UI, advisor, and RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.entities import Player, PlayerCatalog, TeamRoster

POSITIONS = ("QB", "RB", "WR", "TE")
FLEX_POSITIONS = ("RB", "WR", "TE")
DEFAULT_ADP_STD_DEV_FLOOR = 4.0


@dataclass(frozen=True, slots=True)
class PlayerDynamicValue:
    """Current-draft value components for one player.

    Parameters
    ----------
    player_id : int
        Player identifier.
    dynamic_vorp : float
        Current VORP using shared replacement baselines.
    survival_probability : float
        Probability the player survives to the team's next pick.
    wait_loss_points : float
        Expected value lost by waiting until the next pick.
    risk_penalty_points : float
        Contextual downside penalty.
    draft_now_score : float
        Combined point-denominated score for this pick context.
    """

    player_id: int
    dynamic_vorp: float
    survival_probability: float
    wait_loss_points: float
    risk_penalty_points: float
    draft_now_score: float


def get_flex_aware_positional_baselines(
    draft_state: DraftState,
    player_catalog: PlayerCatalog,
    positions: Sequence[str] = POSITIONS,
) -> Dict[str, float]:
    """Return smoothed replacement baselines with FLEX demand included.

    Parameters
    ----------
    draft_state : DraftState
        Current draft state.
    player_catalog : PlayerCatalog
        Player lookup.
    positions : Sequence[str], optional
        Positions to include.

    Returns
    -------
    Dict[str, float]
        Position to replacement baseline in projected points.
    """
    sorted_by_position = build_sorted_available_by_position(
        draft_state.available_player_ids, player_catalog, positions
    )
    team_ids = sorted(set(draft_state.draft_order) | set(draft_state.team_rosters.keys()))
    team_rosters = {team_id: draft_state.roster_for_team(team_id) for team_id in team_ids}
    replacement_demand = calculate_replacement_demand(
        team_rosters,
        draft_state.roster_structure,
        sorted_by_position,
    )
    return {
        position: smoothed_replacement_baseline(
            sorted_by_position.get(position, []),
            replacement_demand.get(position, 0),
        )
        for position in positions
    }


def build_sorted_available_by_position(
    available_player_ids: Iterable[int],
    player_catalog: PlayerCatalog,
    positions: Sequence[str] = POSITIONS,
) -> Dict[str, list[Player]]:
    """Group and sort available players by position."""
    grouped = {position: [] for position in positions}
    for player_id in available_player_ids:
        player = player_catalog.get(player_id)
        if player is None or player.position not in grouped:
            continue
        grouped[player.position].append(player)
    for position in grouped:
        grouped[position].sort(key=lambda player: player.projected_points, reverse=True)
    return grouped


def calculate_replacement_demand(
    team_rosters: Dict[int, TeamRoster],
    roster_structure: Dict[str, int],
    sorted_by_position: Dict[str, list[Player]],
) -> Dict[str, int]:
    """Calculate remaining starter plus FLEX demand by position."""
    demand = {position: 0 for position in POSITIONS}
    total_flex_demand = 0
    for roster in team_rosters.values():
        for position in POSITIONS:
            required = int(roster_structure.get(position, 0))
            demand[position] += max(0, required - roster.position_count(position))
        total_flex_demand += max(0, int(roster_structure.get("FLEX", 0)) - roster.flex_count)

    flex_allocations = _allocate_flex_demand(sorted_by_position, demand, total_flex_demand)
    for position, allocation in flex_allocations.items():
        demand[position] += allocation
    return demand


def smoothed_replacement_baseline(players: Sequence[Player], demand_count: int) -> float:
    """Return the first player beyond demand, smoothed with neighbors."""
    if not players:
        return 0.0
    replacement_index = min(max(0, int(demand_count)), len(players) - 1)
    before = players[max(0, replacement_index - 1)].projected_points
    current = players[replacement_index].projected_points
    after = players[min(len(players) - 1, replacement_index + 1)].projected_points
    return float((before + current + after) / 3.0)


def calculate_player_vorp(player: Player, baselines: Dict[str, float]) -> float:
    """Return a player's VORP against current positional baseline."""
    if getattr(player, "data_completeness", "full") != "full":
        return 0.0
    return float(player.projected_points - baselines.get(player.position, 0.0))


def calculate_dynamic_player_value(
    player: Player,
    draft_state: DraftState,
    player_catalog: PlayerCatalog,
    team_id: int,
    baselines: Optional[Dict[str, float]] = None,
) -> PlayerDynamicValue:
    """Calculate dynamic draft value components for one candidate."""
    resolved_baselines = baselines or get_flex_aware_positional_baselines(draft_state, player_catalog)
    dynamic_vorp = calculate_player_vorp(player, resolved_baselines)
    next_pick = next_pick_number_for_team(
        draft_state.draft_order,
        draft_state.current_pick_index,
        team_id,
    )
    survival = survival_probability_to_pick(
        player_adp=player.adp,
        current_pick=draft_state.current_pick_number,
        target_pick=next_pick,
        adp_dispersion=getattr(player, "market_adp_dispersion", None),
    )
    fallback_vorp = _fallback_vorp_for_position(
        player,
        draft_state.available_player_ids,
        player_catalog,
        resolved_baselines,
    )
    wait_loss = (1.0 - survival) * max(0.0, dynamic_vorp - fallback_vorp)
    risk_penalty = contextual_risk_penalty(player, draft_state.roster_for_team(team_id), draft_state.roster_structure)
    edge = _optional_float(getattr(player, "projection_edge_points", None), default=0.0)
    market_premium = _optional_float(getattr(player, "market_premium_points", None), default=0.0)
    upside_bonus = max(0.0, market_premium) * _upside_weight(
        player, draft_state.roster_for_team(team_id), draft_state.roster_structure
    )
    draft_now_score = dynamic_vorp + wait_loss + 0.35 * edge + upside_bonus - risk_penalty
    return PlayerDynamicValue(
        player_id=player.player_id,
        dynamic_vorp=float(dynamic_vorp),
        survival_probability=float(survival),
        wait_loss_points=float(wait_loss),
        risk_penalty_points=float(risk_penalty),
        draft_now_score=float(draft_now_score),
    )


def next_pick_number_for_team(
    draft_order: Sequence[int],
    current_pick_index: int,
    team_id: int,
) -> Optional[int]:
    """Return the next one-based pick number for ``team_id`` after this pick."""
    for index in range(current_pick_index + 1, len(draft_order)):
        if draft_order[index] == team_id:
            return index + 1
    return None


def survival_probability_to_pick(
    player_adp: float,
    current_pick: int,
    target_pick: Optional[int],
    adp_dispersion: Optional[float] = None,
) -> float:
    """Estimate probability that a player remains available at ``target_pick``."""
    if target_pick is None:
        return 0.0
    if not np.isfinite(player_adp):
        return 0.0
    std_dev = adp_dispersion if adp_dispersion is not None else player_adp / 4.0
    std_dev = max(DEFAULT_ADP_STD_DEV_FLOOR, float(std_dev))
    survival_at_target = 1.0 - _normal_cdf(target_pick, player_adp, std_dev)
    survival_at_current = 1.0 - _normal_cdf(current_pick, player_adp, std_dev)
    if survival_at_current <= 0:
        return 0.0
    return float(max(0.0, min(1.0, survival_at_target / survival_at_current)))


def contextual_risk_penalty(
    player: Player,
    roster: TeamRoster,
    roster_structure: Dict[str, int],
) -> float:
    """Return a context-sensitive point penalty for downside risk."""
    stability = _optional_float(getattr(player, "stability_score", None), default=65.0)
    uncertainty = max(0.0, (100.0 - stability) / 100.0)
    return float(uncertainty * _risk_weight(player, roster, roster_structure))


def insight_alignment_from_payload(insight) -> str:
    """Classify insight evidence as confirm, neutral, or contradict."""
    if insight is None:
        return "neutral"
    values = _insight_values(insight)
    tags = set(values.get("tags", []))
    if (
        values.get("injury_risk") == "high"
        or "suspension_risk" in tags
        or "role_reduction" in tags
        or "workload_concern" in tags
        or values.get("playing_time_tier") == "low"
        or values.get("depth_role") == "backup"
    ):
        return "contradict"
    if (
        values.get("depth_role") in {"starter", "co_starter"}
        and values.get("playing_time_tier") in {"high", "medium"}
        and values.get("overall_confidence") in {"high", "medium"}
    ):
        return "confirm"
    if "role_expansion" in tags or "breakout_candidate" in tags:
        return "confirm"
    return "neutral"


def _allocate_flex_demand(
    sorted_by_position: Dict[str, list[Player]],
    dedicated_demand: Dict[str, int],
    total_flex_demand: int,
) -> Dict[str, int]:
    """Allocate remaining FLEX demand to strongest eligible replacement pools."""
    allocations = {position: 0 for position in FLEX_POSITIONS}
    cursors = {position: dedicated_demand.get(position, 0) for position in FLEX_POSITIONS}
    for _ in range(max(0, int(total_flex_demand))):
        candidates = [
            (sorted_by_position[position][cursors[position]].projected_points, position)
            for position in FLEX_POSITIONS
            if cursors[position] < len(sorted_by_position.get(position, []))
        ]
        if not candidates:
            break
        _points, selected_position = max(candidates, key=lambda item: (item[0], item[1]))
        allocations[selected_position] += 1
        cursors[selected_position] += 1
    return allocations


def _fallback_vorp_for_position(
    player: Player,
    available_player_ids: Iterable[int],
    player_catalog: PlayerCatalog,
    baselines: Dict[str, float],
) -> float:
    """Return best same-position VORP behind ``player``."""
    candidates = [
        candidate
        for player_id in available_player_ids
        if (candidate := player_catalog.get(player_id)) is not None
        and candidate.position == player.position
        and candidate.player_id != player.player_id
    ]
    if not candidates:
        return 0.0
    fallback = max(candidates, key=lambda candidate: candidate.projected_points)
    return calculate_player_vorp(fallback, baselines)


def _risk_weight(player: Player, roster: TeamRoster, roster_structure: Dict[str, int]) -> float:
    """Return risk-aversion weight by roster context."""
    if roster.position_count(player.position) < int(roster_structure.get(player.position, 0)):
        return 3.0
    if player.position in FLEX_POSITIONS and roster.flex_count < int(roster_structure.get("FLEX", 0)):
        return 2.0
    return 0.75


def _upside_weight(player: Player, roster: TeamRoster, roster_structure: Dict[str, int]) -> float:
    """Return upside weight by roster context."""
    if roster.position_count(player.position) < int(roster_structure.get(player.position, 0)):
        return 0.1
    if player.position in FLEX_POSITIONS and roster.flex_count < int(roster_structure.get("FLEX", 0)):
        return 0.2
    return 0.35


def _normal_cdf(value: float, mean: float, std_dev: float) -> float:
    """Return the normal CDF for ``value``."""
    z_score = (value - mean) / (std_dev * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z_score)))


def _optional_float(value: Optional[float], default: float) -> float:
    """Return a finite float or a default value."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _insight_values(insight) -> dict:
    """Return primitive insight values from Pydantic or dict payloads."""
    if hasattr(insight, "model_dump"):
        payload = insight.model_dump(mode="json")
    elif isinstance(insight, dict):
        payload = dict(insight)
    else:
        return {}
    tags = payload.get("tags") or []
    payload["tags"] = [str(tag) for tag in tags]
    return payload
