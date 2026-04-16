"""Core mutable draft state stored as typed ids and counters."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from draft_buddy.core.entities import Pick, Player, TeamRoster


class DraftState:
    """Mutable source of truth for draft progression.

    Parameters
    ----------
    all_player_ids : set[int]
        Initial set of all player ids in the draft pool.
    draft_order : list[int]
        Team ids in draft order.
    roster_structure : dict[str, int]
        Required starters by position.
    bench_maxes : dict[str, int]
        Maximum bench slots by position.
    total_roster_size_per_team : int
        Maximum roster size per team.
    agent_team_id : int, optional
        Team id used as default RL perspective.
    """

    def __init__(
        self,
        all_player_ids: set[int],
        draft_order: list[int],
        roster_structure: dict[str, int],
        bench_maxes: dict[str, int],
        total_roster_size_per_team: int,
        agent_team_id: int = 1,
    ) -> None:
        self.roster_structure = dict(roster_structure)
        self.bench_maxes = dict(bench_maxes)
        self.total_roster_size_per_team = total_roster_size_per_team
        self.reset(all_player_ids, draft_order, agent_team_id)

    @property
    def team_rosters(self) -> dict[int, TeamRoster]:
        """Return rosters keyed by team id."""
        return self._team_rosters

    @property
    def available_player_ids(self) -> set[int]:
        """Return undrafted player ids."""
        return self._available_player_ids

    @available_player_ids.setter
    def available_player_ids(self, player_ids: set[int]) -> None:
        """Replace the set of available player ids."""
        self._available_player_ids = set(player_ids)

    @property
    def draft_order(self) -> list[int]:
        """Return the global draft order."""
        return self._draft_order

    @draft_order.setter
    def draft_order(self, draft_order: list[int]) -> None:
        """Set the global draft order."""
        self._draft_order = list(draft_order)

    @property
    def current_pick_index(self) -> int:
        """Return zero-based pick index."""
        return self._current_pick_index

    @current_pick_index.setter
    def current_pick_index(self, value: int) -> None:
        """Set zero-based pick index."""
        self._current_pick_index = value

    @property
    def current_pick_number(self) -> int:
        """Return one-based global pick number."""
        return self._current_pick_number

    @current_pick_number.setter
    def current_pick_number(self, value: int) -> None:
        """Set one-based global pick number."""
        self._current_pick_number = value

    @property
    def agent_team_id(self) -> int:
        """Return the configured agent team id."""
        return self._agent_team_id

    @agent_team_id.setter
    def agent_team_id(self, value: int) -> None:
        """Set the configured agent team id."""
        self._agent_team_id = value

    @property
    def override_team_id(self) -> Optional[int]:
        """Return temporary manual override team id."""
        return self._override_team_id

    @override_team_id.setter
    def override_team_id(self, value: Optional[int]) -> None:
        """Set temporary manual override team id."""
        self._override_team_id = value

    @property
    def draft_history(self) -> list[Pick]:
        """Return typed draft history."""
        return self._draft_history

    def roster_for_team(self, team_id: int) -> TeamRoster:
        """Return one team's roster, creating it on first access."""
        return self._team_rosters[team_id]

    def add_player_to_roster(self, team_id: int, player: Player) -> None:
        """Add a player id to a team roster and update counts."""
        roster = self.roster_for_team(team_id)
        roster.player_ids.append(player.player_id)
        self.available_player_ids.discard(player.player_id)
        self._update_roster_counts_for_pick(team_id, player)

    def remove_player_from_roster(self, team_id: int, player: Player) -> None:
        """Remove a player id from roster and restore availability."""
        roster = self.roster_for_team(team_id)
        roster.player_ids = [
            player_id for player_id in roster.player_ids if player_id != player.player_id
        ]
        self.available_player_ids.add(player.player_id)

    def recalculate_roster_counts(self, team_id: int, player_lookup) -> None:
        """Rebuild positional counters from roster player ids."""
        roster = self.roster_for_team(team_id)
        roster.qb_count = 0
        roster.rb_count = 0
        roster.wr_count = 0
        roster.te_count = 0
        roster.flex_count = 0
        for player_id in roster.player_ids:
            self._update_roster_counts_for_pick(team_id, player_lookup(player_id))

    def advance_pick(self) -> None:
        """Advance the draft cursor to the next pick."""
        self.current_pick_index += 1
        self.current_pick_number += 1

    def append_pick(self, pick: Pick) -> None:
        """Append one pick to draft history."""
        self._draft_history.append(pick)

    def pop_pick(self) -> Optional[Pick]:
        """Pop the latest pick from draft history when present."""
        return self._draft_history.pop() if self._draft_history else None

    def reset(self, all_player_ids: set[int], draft_order: list[int], agent_team_id: int) -> None:
        """Reset state to a fresh draft."""
        self._available_player_ids = set(all_player_ids)
        self._team_rosters = defaultdict(TeamRoster)
        self._draft_order = list(draft_order)
        self._current_pick_index = 0
        self._current_pick_number = 1
        self._agent_team_id = agent_team_id
        self._draft_history: list[Pick] = []
        self._override_team_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize state to a JSON-friendly dictionary."""
        return {
            "available_player_ids": sorted(self.available_player_ids),
            "team_rosters": {
                str(team_id): roster.to_dict() for team_id, roster in self.team_rosters.items()
            },
            "draft_order": list(self.draft_order),
            "current_pick_index": self.current_pick_index,
            "current_pick_number": self.current_pick_number,
            "draft_history": [pick.to_dict() for pick in self.draft_history],
            "override_team_id": self.override_team_id,
            "agent_team_id": self.agent_team_id,
        }

    def load_from_dict(self, payload: dict) -> None:
        """Load state from serialized data."""
        self.available_player_ids = {int(player_id) for player_id in payload.get("available_player_ids", [])}
        self._team_rosters = defaultdict(TeamRoster)
        for team_id_str, roster_payload in payload.get("team_rosters", {}).items():
            try:
                team_id = int(team_id_str)
            except (TypeError, ValueError):
                continue
            self._team_rosters[team_id] = TeamRoster.from_dict(roster_payload)
        self.draft_order = [int(team_id) for team_id in payload.get("draft_order", [])]
        self.current_pick_index = int(payload.get("current_pick_index", 0))
        self.current_pick_number = int(payload.get("current_pick_number", 1))
        self._draft_history = [
            Pick.from_dict(pick_payload) for pick_payload in payload.get("draft_history", [])
        ]
        self.override_team_id = payload.get("override_team_id")
        self.agent_team_id = int(payload.get("agent_team_id", self.agent_team_id))

    def _update_roster_counts_for_pick(self, team_id: int, player: Player) -> None:
        """Update roster counters for one drafted player."""
        roster = self.roster_for_team(team_id)
        position = player.position
        if roster.position_count(position) < self.roster_structure.get(position, 0):
            roster.set_position_count(position, roster.position_count(position) + 1)
            return
        if position in {"RB", "WR", "TE"} and roster.flex_count < self.roster_structure.get("FLEX", 0):
            roster.flex_count += 1
            return
        roster.set_position_count(position, roster.position_count(position) + 1)
