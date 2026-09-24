"""In-season league snapshot value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


@dataclass
class TeamRecord:
    """Regular-season record locked before remaining-week simulation."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    points: float = 0.0

    def to_sim_record(self) -> Dict[str, Any]:
        """Return the evaluator's W/L/T/pts mapping.

        Returns
        -------
        dict
            Record keyed as ``W``, ``L``, ``T``, and ``pts``.
        """
        return {
            "W": int(self.wins),
            "L": int(self.losses),
            "T": int(self.ties),
            "pts": float(self.points),
        }


@dataclass
class TeamSnapshot:
    """One team's current roster and locked standings."""

    team_id: int
    display_name: str
    player_ids: List[int] = field(default_factory=list)
    ir_player_ids: List[int] = field(default_factory=list)
    record: TeamRecord = field(default_factory=TeamRecord)


@dataclass
class ScheduledMatchup:
    """One remaining regular-season matchup."""

    week: int
    home_team_id: int
    away_team_id: int
    source: str = "round_robin"


@dataclass
class LeagueSnapshot:
    """Platform-agnostic in-season league state.

    Parameters
    ----------
    league_id : str
        Active league profile id.
    platform : str
        ``sleeper`` or ``espn``.
    current_week : int
        First NFL week that still needs to be simulated.
    teams : dict of int to TeamSnapshot
        Rosters keyed by compute team id.
    remaining_matchups : list of ScheduledMatchup
        Matchups at or after ``current_week``.
    unresolved_ids : list of str
        Platform player ids that could not be mapped into the catalog.
    """

    league_id: str
    platform: str
    current_week: int
    teams: Dict[int, TeamSnapshot] = field(default_factory=dict)
    remaining_matchups: List[ScheduledMatchup] = field(default_factory=list)
    unresolved_ids: List[str] = field(default_factory=list)

    def team(self, team_id: int) -> TeamSnapshot:
        """Return a team snapshot or raise.

        Parameters
        ----------
        team_id : int
            Compute team id.

        Returns
        -------
        TeamSnapshot
            Matching team.

        Raises
        ------
        KeyError
            If the team is missing.
        """
        if team_id not in self.teams:
            raise KeyError(f"Unknown team id: {team_id}")
        return self.teams[team_id]

    def starting_records(self) -> Dict[str, Dict[str, Any]]:
        """Return evaluator starting records keyed by ``Team {id}``.

        Returns
        -------
        dict
            Locked W/L/T/pts by team label.
        """
        from draft_buddy.simulator.team_identity import team_label

        return {
            team_label(team.team_id): team.record.to_sim_record()
            for team in self.teams.values()
        }

    def rosters_by_label(self) -> Dict[str, List[int]]:
        """Return skill rosters keyed by ``Team {id}``.

        Returns
        -------
        dict
            Player id lists by team label.
        """
        from draft_buddy.simulator.team_identity import team_label

        return {
            team_label(team.team_id): list(team.player_ids)
            for team in self.teams.values()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the snapshot for disk and API responses.

        Returns
        -------
        dict
            JSON-friendly payload.
        """
        return {
            "league_id": self.league_id,
            "platform": self.platform,
            "current_week": self.current_week,
            "unresolved_ids": list(self.unresolved_ids),
            "teams": {
                str(team_id): {
                    "team_id": team.team_id,
                    "display_name": team.display_name,
                    "player_ids": list(team.player_ids),
                    "ir_player_ids": list(team.ir_player_ids),
                    "record": asdict(team.record),
                }
                for team_id, team in self.teams.items()
            },
            "remaining_matchups": [asdict(matchup) for matchup in self.remaining_matchups],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LeagueSnapshot":
        """Restore a snapshot from serialized JSON.

        Parameters
        ----------
        payload : dict
            Stored snapshot.

        Returns
        -------
        LeagueSnapshot
            Restored state.
        """
        teams: Dict[int, TeamSnapshot] = {}
        for raw_team in payload.get("teams", {}).values():
            record_payload = raw_team.get("record") or {}
            team = TeamSnapshot(
                team_id=int(raw_team["team_id"]),
                display_name=str(raw_team.get("display_name") or ""),
                player_ids=[int(player_id) for player_id in raw_team.get("player_ids") or []],
                ir_player_ids=[int(player_id) for player_id in raw_team.get("ir_player_ids") or []],
                record=TeamRecord(
                    wins=int(record_payload.get("wins", 0)),
                    losses=int(record_payload.get("losses", 0)),
                    ties=int(record_payload.get("ties", 0)),
                    points=float(record_payload.get("points", 0.0)),
                ),
            )
            teams[team.team_id] = team
        matchups = [
            ScheduledMatchup(
                week=int(item["week"]),
                home_team_id=int(item["home_team_id"]),
                away_team_id=int(item["away_team_id"]),
                source=str(item.get("source") or "round_robin"),
            )
            for item in payload.get("remaining_matchups") or []
        ]
        return cls(
            league_id=str(payload.get("league_id") or ""),
            platform=str(payload.get("platform") or ""),
            current_week=int(payload.get("current_week") or 1),
            teams=teams,
            remaining_matchups=matchups,
            unresolved_ids=[str(item) for item in payload.get("unresolved_ids") or []],
        )


def roster_cap(roster_structure: Dict[str, int], total_bench_size: int) -> int:
    """Return maximum skill-roster size including flex and bench.

    Parameters
    ----------
    roster_structure : dict of str to int
        Starter slots including FLEX.
    total_bench_size : int
        Bench slots.

    Returns
    -------
    int
        Roster cap used for smart drops.
    """
    return int(sum(roster_structure.values()) + total_bench_size)


def resolve_sleeper_player(catalog, sleeper_id: str):
    """Map a Sleeper player id onto a catalog player when present.

    Parameters
    ----------
    catalog : PlayerCatalog
        Loaded player catalog.
    sleeper_id : str
        Sleeper identifier.

    Returns
    -------
    Player or None
        Matching catalog player.
    """
    from draft_buddy.data.sleeper_player_resolver import encode_sleeper_player_id

    token = str(sleeper_id)
    for player in catalog:
        if player.sleeper_id == token:
            return player
    return catalog.get(encode_sleeper_player_id(token))
