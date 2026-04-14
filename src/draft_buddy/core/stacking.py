"""Stacking-related core domain logic."""

from __future__ import annotations

from typing import List

from draft_buddy.domain.entities import Player


def calculate_stack_count(roster_players: List[Player]) -> int:
    """Calculate QB-WR/TE stack count for a roster.

    Parameters
    ----------
    roster_players : List[Player]
        Players currently assigned to a roster.

    Returns
    -------
    int
        Number of valid QB to WR/TE stacks across NFL teams.
    """
    if not roster_players:
        return 0
    team_positions = {}
    for player in roster_players:
        if player.team is None:
            continue
        if player.team not in team_positions:
            team_positions[player.team] = {"QB": 0, "WR": 0, "TE": 0}
        if player.position in ["QB", "WR", "TE"]:
            team_positions[player.team][player.position] += 1
    total_stacks = 0
    for positions in team_positions.values():
        total_stacks += positions["QB"] * (positions["WR"] + positions["TE"])
    return total_stacks
