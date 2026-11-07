"""Positional holes and bench surplus for in-season trade context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Sequence

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.core.roster_utils import categorize_roster_by_slots
from draft_buddy.inseason.state import SKILL_POSITIONS, TeamSnapshot


@dataclass
class TeamNeedsProfile:
    """One team's starter holes and bench rest-of-season surplus."""

    team_id: int
    display_name: str
    wins: int
    losses: int
    ties: int
    hole_slots: Dict[str, int]
    bench_ros: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        """Serialize the profile."""
        return asdict(self)


def team_needs_profile(
    team: TeamSnapshot,
    catalog: PlayerCatalog,
    ros_totals: Mapping[int, float],
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> TeamNeedsProfile:
    """Describe holes and bench surplus for one roster.

    Parameters
    ----------
    team : TeamSnapshot
        Current roster and record.
    catalog : PlayerCatalog
        Player lookup.
    ros_totals : mapping of int to float
        Remaining ROS totals.
    roster_structure : mapping
        Starter requirements by position.
    bench_maxes : mapping
        Bench limits by position.

    Returns
    -------
    TeamNeedsProfile
        Short starter slots and bench ROS by position.
    """
    players = _players_with_ros(team.player_ids, catalog, ros_totals)
    starters, bench, _flex = categorize_roster_by_slots(
        players, dict(roster_structure), dict(bench_maxes)
    )
    holes: Dict[str, int] = {}
    for position, needed in roster_structure.items():
        if position == "FLEX":
            continue
        missing = int(needed) - len(starters.get(position, []))
        if missing > 0:
            holes[position] = missing
    surplus: Dict[str, float] = {}
    for player in bench:
        if player.position not in SKILL_POSITIONS:
            continue
        surplus[player.position] = surplus.get(player.position, 0.0) + float(
            player.projected_points
        )
    rounded = {position: round(points, 1) for position, points in surplus.items() if points > 0}
    return TeamNeedsProfile(
        team_id=int(team.team_id),
        display_name=team.display_name,
        wins=int(team.record.wins),
        losses=int(team.record.losses),
        ties=int(team.record.ties),
        hole_slots=holes,
        bench_ros=rounded,
    )


def complement_score(user: TeamNeedsProfile, partner: TeamNeedsProfile) -> float:
    """Score how a partner's surplus lines up with the user's holes.

    Parameters
    ----------
    user : TeamNeedsProfile
        Searching manager.
    partner : TeamNeedsProfile
        Other manager.

    Returns
    -------
    float
        Partner bench ROS at the user's holes, plus the user's bench ROS
        at the partner's holes.
    """
    score = 0.0
    for position, slots in user.hole_slots.items():
        if slots > 0:
            score += float(partner.bench_ros.get(position, 0.0))
    for position, slots in partner.hole_slots.items():
        if slots > 0:
            score += float(user.bench_ros.get(position, 0.0))
    return score


def complement_rankings(
    profiles: Sequence[TeamNeedsProfile],
    user_team_id: int,
) -> List[Dict[str, object]]:
    """Order other teams by complement score.

    Parameters
    ----------
    profiles : sequence of TeamNeedsProfile
        One profile per league team.
    user_team_id : int
        Searching manager.

    Returns
    -------
    list of dict
        Other teams, highest complement score first.
    """
    by_id = {profile.team_id: profile for profile in profiles}
    user = by_id[int(user_team_id)]
    rows = []
    for profile in profiles:
        if profile.team_id == user.team_id:
            continue
        rows.append(
            {
                "team_id": profile.team_id,
                "display_name": profile.display_name,
                "score": round(complement_score(user, profile), 1),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def _players_with_ros(
    player_ids: Sequence[int],
    catalog: PlayerCatalog,
    ros_totals: Mapping[int, float],
) -> List[Player]:
    """Overlay remaining ROS totals onto catalog players."""
    players: List[Player] = []
    for player_id in player_ids:
        player = catalog.get(int(player_id))
        if player is None:
            continue
        points = float(ros_totals.get(player.player_id, 0.0))
        players.append(replace(player, projected_points=points))
    return players
