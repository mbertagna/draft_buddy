"""Sleeper-backed in-season league snapshot adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.sleeper_client import SleeperGateway
from draft_buddy.inseason.schedule import fill_missing_weeks
from draft_buddy.inseason.state import (
    SKILL_POSITIONS,
    LeagueSnapshot,
    ScheduledMatchup,
    TeamRecord,
    TeamSnapshot,
    resolve_sleeper_player,
)


def current_week_from_nfl_state(nfl_state: Mapping[str, object], last_regular_week: int) -> int:
    """Map Sleeper NFL state onto a regular-season week number.

    Parameters
    ----------
    nfl_state : mapping
        Sleeper ``/state/nfl`` payload.
    last_regular_week : int
        Configured regular-season length.

    Returns
    -------
    int
        Week to use as the first remaining simulation week.
    """
    season_type = str(nfl_state.get("season_type") or "regular")
    week = int(nfl_state.get("week") or 1)
    if season_type == "pre":
        return 1
    if season_type in {"post", "off"}:
        return int(last_regular_week)
    return max(1, min(week, int(last_regular_week)))


class SleeperSnapshotAdapter:
    """Build a ``LeagueSnapshot`` from live Sleeper league endpoints.

    Parameters
    ----------
    gateway : SleeperGateway
        Sleeper HTTP (or test) gateway.
    """

    def __init__(self, gateway: SleeperGateway) -> None:
        self._gateway = gateway

    def build_snapshot(
        self,
        league_id: str,
        sleeper_league_id: str,
        roster_id_to_team_id: Mapping[int, int],
        catalog: PlayerCatalog,
        team_names: Mapping[int, str],
        last_regular_week: int,
        current_week: Optional[int] = None,
    ) -> LeagueSnapshot:
        """Fetch rosters, records, and remaining matchups.

        Parameters
        ----------
        league_id : str
            Draft Buddy league profile id.
        sleeper_league_id : str
            Sleeper league identifier.
        roster_id_to_team_id : mapping of int to int
            Sleeper roster id to compute team id.
        catalog : PlayerCatalog
            Player catalog used to resolve Sleeper ids.
        team_names : mapping of int to str
            Display names keyed by compute team id.
        last_regular_week : int
            Final regular-season week.
        current_week : int, optional
            Override for the Sleeper NFL week.

        Returns
        -------
        LeagueSnapshot
            Live league state.
        """
        if current_week is None:
            nfl_state = self._gateway.fetch_nfl_state()
            resolved_week = current_week_from_nfl_state(nfl_state, last_regular_week)
        else:
            resolved_week = int(current_week)
        payloads = self._gateway.fetch_league_roster_payloads(sleeper_league_id)
        teams, unresolved = self._teams_from_payloads(
            payloads, roster_id_to_team_id, catalog, team_names
        )
        matchups = self._remaining_matchups(
            sleeper_league_id,
            roster_id_to_team_id,
            list(teams),
            resolved_week,
            last_regular_week,
        )
        return LeagueSnapshot(
            league_id=league_id,
            platform="sleeper",
            current_week=resolved_week,
            teams=teams,
            remaining_matchups=matchups,
            unresolved_ids=unresolved,
        )

    def _teams_from_payloads(
        self,
        payloads: Iterable[Mapping[str, object]],
        roster_id_to_team_id: Mapping[int, int],
        catalog: PlayerCatalog,
        team_names: Mapping[int, str],
    ) -> tuple[Dict[int, TeamSnapshot], List[str]]:
        """Translate raw Sleeper rosters into team snapshots."""
        teams: Dict[int, TeamSnapshot] = {}
        unresolved: List[str] = []
        for payload in payloads:
            roster_id = int(payload.get("roster_id") or 0)
            team_id = int(roster_id_to_team_id.get(roster_id, roster_id))
            settings = payload.get("settings") or {}
            if not isinstance(settings, dict):
                settings = {}
            player_ids, ir_ids, missing = self._map_player_lists(payload, catalog)
            unresolved.extend(missing)
            teams[team_id] = TeamSnapshot(
                team_id=team_id,
                display_name=str(team_names.get(team_id) or f"Team {team_id}"),
                player_ids=player_ids,
                ir_player_ids=ir_ids,
                record=TeamRecord(
                    wins=int(settings.get("wins") or 0),
                    losses=int(settings.get("losses") or 0),
                    ties=int(settings.get("ties") or 0),
                    points=float(settings.get("fpts") or settings.get("fpts_decimal") or 0.0),
                ),
            )
        return teams, unresolved

    def _map_player_lists(
        self,
        payload: Mapping[str, object],
        catalog: PlayerCatalog,
    ) -> tuple[List[int], List[int], List[str]]:
        """Resolve Sleeper roster and reserve lists onto catalog ids."""
        player_ids: List[int] = []
        ir_ids: List[int] = []
        unresolved: List[str] = []
        for sleeper_id in payload.get("players") or []:
            player = resolve_sleeper_player(catalog, str(sleeper_id))
            if player is None or player.position not in SKILL_POSITIONS:
                if player is None:
                    unresolved.append(str(sleeper_id))
                continue
            player_ids.append(player.player_id)
        for sleeper_id in payload.get("reserve") or []:
            player = resolve_sleeper_player(catalog, str(sleeper_id))
            if player is None:
                unresolved.append(str(sleeper_id))
                continue
            ir_ids.append(player.player_id)
        return player_ids, ir_ids, unresolved

    def _remaining_matchups(
        self,
        sleeper_league_id: str,
        roster_id_to_team_id: Mapping[int, int],
        team_ids: List[int],
        start_week: int,
        last_week: int,
    ) -> List[ScheduledMatchup]:
        """Fetch remaining Sleeper matchups, filling only weeks the API omits."""
        collected: List[ScheduledMatchup] = []
        for week in range(int(start_week), int(last_week) + 1):
            try:
                raw_rows = self._gateway.fetch_league_matchups(sleeper_league_id, week)
            except Exception:
                continue
            collected.extend(self._pair_matchups(raw_rows, roster_id_to_team_id, week))
        return fill_missing_weeks(collected, team_ids, start_week, last_week)

    @staticmethod
    def _pair_matchups(
        raw_rows: Iterable[Mapping[str, object]],
        roster_id_to_team_id: Mapping[int, int],
        week: int,
    ) -> List[ScheduledMatchup]:
        """Pair Sleeper rows that share a ``matchup_id``."""
        grouped: Dict[object, List[int]] = defaultdict(list)
        for row in raw_rows:
            matchup_id = row.get("matchup_id")
            roster_id = row.get("roster_id")
            if matchup_id is None or roster_id is None:
                continue
            team_id = int(roster_id_to_team_id.get(int(roster_id), int(roster_id)))
            grouped[matchup_id].append(team_id)
        games: List[ScheduledMatchup] = []
        for team_ids in grouped.values():
            unique_ids = list(dict.fromkeys(team_ids))
            if len(unique_ids) != 2:
                continue
            games.append(
                ScheduledMatchup(
                    week=int(week),
                    home_team_id=unique_ids[0],
                    away_team_id=unique_ids[1],
                    source="sleeper",
                )
            )
        return games
