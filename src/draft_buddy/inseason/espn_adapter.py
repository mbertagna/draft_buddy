"""ESPN manual-paste snapshot adapter."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.data.name_matching import standardize_name
from draft_buddy.inseason.exceptions import NameResolutionError
from draft_buddy.inseason.schedule import fill_missing_weeks
from draft_buddy.inseason.state import (
    SKILL_POSITIONS,
    LeagueSnapshot,
    ScheduledMatchup,
    TeamRecord,
    TeamSnapshot,
)


class EspnSnapshotAdapter:
    """Build a ``LeagueSnapshot`` from a pasted ESPN roster payload."""

    def build_snapshot(
        self,
        league_id: str,
        payload: Mapping[str, Any],
        catalog: PlayerCatalog,
        team_names: Mapping[int, str],
        last_regular_week: int,
        team_ids: Optional[Sequence[int]] = None,
    ) -> LeagueSnapshot:
        """Resolve pasted names or ids into a league snapshot.

        Parameters
        ----------
        league_id : str
            Draft Buddy league profile id.
        payload : mapping
            ``{current_week, teams: [{team_id|team|display_name, wins, losses, player_names|player_ids}]}``.
        catalog : PlayerCatalog
            Player catalog used for name and id resolution.
        team_names : mapping of int to str
            Display names keyed by compute team id.
        last_regular_week : int
            Final regular-season week.
        team_ids : sequence of int, optional
            League team ids used when building a fallback schedule.

        Returns
        -------
        LeagueSnapshot
            Manual league state.

        Raises
        ------
        NameResolutionError
            If any pasted name is unmatched or ambiguous.
        """
        current_week = max(1, int(payload.get("current_week") or 1))
        teams: Dict[int, TeamSnapshot] = {}
        for raw_team in payload.get("teams") or []:
            team_id = resolve_team_id_from_payload(raw_team, team_names)
            player_ids = self._resolve_team_players(raw_team, catalog)
            mapped_name = team_names.get(team_id) or f"Team {team_id}"
            teams[team_id] = TeamSnapshot(
                team_id=team_id,
                display_name=str(mapped_name),
                player_ids=player_ids,
                ir_player_ids=[int(player_id) for player_id in raw_team.get("ir_player_ids") or []],
                record=TeamRecord(
                    wins=int(raw_team.get("wins") or 0),
                    losses=int(raw_team.get("losses") or 0),
                    ties=int(raw_team.get("ties") or 0),
                    points=float(raw_team.get("points") or 0.0),
                ),
            )
        schedule_ids = list(team_ids) if team_ids is not None else list(teams)
        provided = matchups_from_rows(payload.get("matchups") or [], team_names, current_week)
        matchups = fill_missing_weeks(provided, schedule_ids, current_week, last_regular_week)
        return LeagueSnapshot(
            league_id=league_id,
            platform="espn",
            current_week=current_week,
            teams=teams,
            remaining_matchups=matchups,
        )

    def _resolve_team_players(
        self,
        raw_team: Mapping[str, Any],
        catalog: PlayerCatalog,
    ) -> List[int]:
        """Resolve explicit ids first, then pasted names."""
        if raw_team.get("player_ids"):
            return [int(player_id) for player_id in raw_team["player_ids"]]
        names = [str(name) for name in raw_team.get("player_names") or []]
        return resolve_player_names(names, catalog)


def resolve_team_id_from_payload(
    raw_team: Mapping[str, Any],
    team_names: Mapping[int, str],
) -> int:
    """Resolve a pasted team object onto a configured team id.

    Parameters
    ----------
    raw_team : mapping
        Paste row with ``team_id`` and/or ``team`` / ``display_name``.
    team_names : mapping of int to str
        Configured manager names.

    Returns
    -------
    int
        Compute team id.

    Raises
    ------
    NameResolutionError
        If the team label is missing, unmatched, or ambiguous.
    """
    if raw_team.get("team_id") is not None and str(raw_team.get("team_id")).strip() != "":
        return int(raw_team["team_id"])
    label = raw_team.get("team") or raw_team.get("display_name")
    if not label:
        raise NameResolutionError(["(missing team_id or team name)"], [])
    return resolve_team_id(str(label), team_names)


def resolve_team_id(token: str, team_names: Mapping[int, str]) -> int:
    """Map a team id or manager name onto a configured team id.

    Parameters
    ----------
    token : str
        Numeric team id or manager display name.
    team_names : mapping of int to str
        Configured manager names.

    Returns
    -------
    int
        Compute team id.

    Raises
    ------
    NameResolutionError
        If the token is unmatched or matches more than one team.
    """
    stripped = str(token).strip()
    if stripped.isdigit():
        return int(stripped)
    key = standardize_name(stripped)
    matches = [
        team_id
        for team_id, name in team_names.items()
        if standardize_name(name) == key
    ]
    if not matches:
        raise NameResolutionError([stripped], [])
    if len(matches) > 1:
        raise NameResolutionError([], [stripped])
    return matches[0]


def matchups_from_rows(
    rows: Sequence[Mapping[str, Any]],
    team_names: Mapping[int, str],
    current_week: int,
) -> List[ScheduledMatchup]:
    """Resolve pasted matchup rows onto team ids.

    Parameters
    ----------
    rows : sequence of mapping
        Objects with ``week``, ``home``, and ``away``.
    team_names : mapping of int to str
        Configured manager names.
    current_week : int
        Weeks before this are ignored.

    Returns
    -------
    list of ScheduledMatchup
        File-sourced games at or after ``current_week``.

    Raises
    ------
    NameResolutionError
        If a home or away label does not match one team.
    """
    matchups: List[ScheduledMatchup] = []
    for row in rows:
        week = int(row.get("week") or 0)
        if week < int(current_week):
            continue
        matchups.append(
            ScheduledMatchup(
                week=week,
                home_team_id=resolve_team_id(str(row.get("home") or ""), team_names),
                away_team_id=resolve_team_id(str(row.get("away") or ""), team_names),
                source="file",
            )
        )
    return matchups


def resolve_player_names(names: Iterable[str], catalog: PlayerCatalog) -> List[int]:
    """Map display names onto unique catalog player ids.

    Parameters
    ----------
    names : iterable of str
        Pasted ESPN player names.
    catalog : PlayerCatalog
        Lookup catalog.

    Returns
    -------
    list of int
        Resolved player ids in input order.

    Raises
    ------
    NameResolutionError
        If any name is unmatched or matches multiple skill players.
    """
    index = _skill_name_index(catalog)
    resolved: List[int] = []
    unmatched: List[str] = []
    ambiguous: List[str] = []
    for name in names:
        key = standardize_name(name)
        matches = index.get(key or "", [])
        if not matches:
            unmatched.append(name)
            continue
        if len(matches) > 1:
            ambiguous.append(name)
            continue
        resolved.append(matches[0].player_id)
    if unmatched or ambiguous:
        raise NameResolutionError(unmatched, ambiguous)
    return resolved


def _skill_name_index(catalog: PlayerCatalog) -> Dict[str, List[Player]]:
    """Index skill players by standardized name."""
    index: Dict[str, List[Player]] = {}
    for player in catalog:
        if player.position not in SKILL_POSITIONS:
            continue
        key = standardize_name(player.name)
        if not key:
            continue
        index.setdefault(key, []).append(player)
    return index
