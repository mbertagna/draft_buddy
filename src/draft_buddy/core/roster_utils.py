"""Core roster allocation and scoring logic."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def categorize_roster_by_slots(
    team_roster: List[Any],
    roster_structure: Dict[str, int],
    bench_maxes: Dict[str, int],
) -> Tuple[Dict[str, List[Any]], List[Any], List[Any]]:
    """Split a roster into starters, bench, and flex allocations.

    Parameters
    ----------
    team_roster : List[Any]
        Players to allocate.
    roster_structure : Dict[str, int]
        Starter requirements by position.
    bench_maxes : Dict[str, int]
        Bench limits by position.

    Returns
    -------
    Tuple[Dict[str, List[Any]], List[Any], List[Any]]
        Starters by slot, bench players, and flex players.
    """
    starters = defaultdict(list)
    bench: List[Any] = []
    flex_players: List[Any] = []
    sorted_roster = sorted(team_roster, key=lambda player: player.projected_points, reverse=True)
    assigned_player_ids = set()
    for position in roster_structure.keys():
        if position == "FLEX":
            continue
        position_players = [
            player
            for player in sorted_roster
            if player.position == position and getattr(player, "player_id", id(player)) not in assigned_player_ids
        ]
        required_starters = roster_structure.get(position, 0)
        for index in range(min(len(position_players), required_starters)):
            player = position_players[index]
            starters[position].append(player)
            assigned_player_ids.add(getattr(player, "player_id", id(player)))
    remaining_players = [
        player
        for player in sorted_roster
        if getattr(player, "player_id", id(player)) not in assigned_player_ids
    ]
    flex_candidates = [player for player in remaining_players if player.position in ["RB", "WR", "TE"]]
    num_flex_spots = roster_structure.get("FLEX", 0)
    flex_players = sorted(
        flex_candidates, key=lambda player: player.projected_points, reverse=True
    )[:num_flex_spots]
    for player in flex_players:
        starters["FLEX"].append(player)
        assigned_player_ids.add(getattr(player, "player_id", id(player)))
    bench = [
        player
        for player in sorted_roster
        if getattr(player, "player_id", id(player)) not in assigned_player_ids
    ]
    return starters, bench, flex_players


def calculate_roster_scores(
    team_roster: List[Any],
    roster_structure: Dict[str, int],
    bench_maxes: Dict[str, int],
) -> Dict[str, float]:
    """Calculate starter, flex, bench, and total roster scores.

    Parameters
    ----------
    team_roster : List[Any]
        Players to score.
    roster_structure : Dict[str, int]
        Starter requirements by position.
    bench_maxes : Dict[str, int]
        Bench limits by position.

    Returns
    -------
    Dict[str, float]
        Aggregate totals and averages for roster sections.
    """
    starters_dict, bench_list, flex_list = categorize_roster_by_slots(
        team_roster, roster_structure, bench_maxes
    )
    starter_points = sum(
        player.projected_points
        for position_players in starters_dict.values()
        for player in position_players
        if player not in flex_list
    )
    flex_points = sum(player.projected_points for player in flex_list)
    bench_points = sum(player.projected_points for player in bench_list)
    temp_pos_counts = defaultdict(int)
    for position, position_players in starters_dict.items():
        if position == "FLEX":
            continue
        for player in position_players:
            temp_pos_counts[player.position] += 1
    for player in starters_dict.get("FLEX", []):
        temp_pos_counts[player.position] += 1
    for player in bench_list:
        position_limit = roster_structure.get(player.position, 0) + bench_maxes.get(player.position, 0)
        if player.position in ("RB", "WR", "TE"):
            position_limit += roster_structure.get("FLEX", 0)
        if temp_pos_counts[player.position] < position_limit:
            temp_pos_counts[player.position] += 1
        else:
            logger.warning(
                "Player %s (%s) could not be optimally assigned; kept on bench.",
                getattr(player, "name", player),
                player.position,
            )
    combined_total_points = starter_points + bench_points + flex_points
    num_starters = sum(
        len(position_players)
        for position, position_players in starters_dict.items()
        if position != "FLEX"
    ) + len(flex_list)
    num_bench = len(bench_list)
    avg_starter_points = (starter_points + flex_points) / num_starters if num_starters > 0 else 0.0
    avg_bench_points = bench_points / num_bench if num_bench > 0 else 0.0
    avg_combined_points = (
        combined_total_points / (num_starters + num_bench)
        if (num_starters + num_bench) > 0
        else 0.0
    )
    return {
        "starters_total_points": starter_points + flex_points,
        "bench_total_points": bench_points,
        "flex_total_points": flex_points,
        "combined_total_points": combined_total_points,
        "starters_avg_points": avg_starter_points,
        "bench_avg_points": avg_bench_points,
        "combined_avg_points": avg_combined_points,
        "roster_size": len(team_roster),
    }
