"""Draft-state data model independent from adapters."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from draft_buddy.domain.entities import Player


class DraftState:
    """Mutable source of truth for draft progression.

    Parameters
    ----------
    all_player_ids : Set[int]
        Initial set of all player IDs in the draft pool.
    draft_order : List[int]
        Team IDs in draft order.
    roster_structure : Dict[str, int]
        Required starters by position.
    bench_maxes : Dict[str, int]
        Maximum bench slots by position.
    total_roster_size_per_team : int
        Maximum players on each roster.
    agent_team_id : int, optional
        Team id used as default "agent" perspective.
    """

    def __init__(
        self,
        all_player_ids: Set[int],
        draft_order: List[int],
        roster_structure: Dict[str, int],
        bench_maxes: Dict[str, int],
        total_roster_size_per_team: int,
        agent_team_id: int = 1,
    ) -> None:
        self._available_player_ids: Set[int] = set(all_player_ids)
        self._rosters: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "PLAYERS": []}
        )
        self._draft_order = draft_order
        self._current_pick_idx = 0
        self._current_pick_number = 1
        self._agent_team_id = agent_team_id
        self._draft_history: List[Dict[str, Any]] = []
        self._overridden_team_id: Optional[int] = None
        self._roster_structure = roster_structure
        self._bench_maxes = bench_maxes
        self._total_roster_size_per_team = total_roster_size_per_team

    def get_available_player_ids(self) -> Set[int]:
        """Return undrafted player IDs."""
        return self._available_player_ids

    def replace_available_player_ids(self, player_ids: Set[int]) -> None:
        """Replace available-player IDs."""
        self._available_player_ids = set(player_ids)

    def get_rosters(self) -> Dict[int, Dict[str, Any]]:
        """Return all team rosters."""
        return self._rosters

    def get_draft_order(self) -> List[int]:
        """Return draft order."""
        return self._draft_order

    def get_current_pick_idx(self) -> int:
        """Return zero-based pick index."""
        return self._current_pick_idx

    def get_current_pick_number(self) -> int:
        """Return one-based global pick number."""
        return self._current_pick_number

    def get_agent_team_id(self) -> int:
        """Return perspective team id."""
        return self._agent_team_id

    def get_overridden_team_id(self) -> Optional[int]:
        """Return temporary manual override team id."""
        return self._overridden_team_id

    def get_draft_history(self) -> List[Dict[str, Any]]:
        """Return raw draft history entries."""
        return self._draft_history

    def get_roster_structure(self) -> Dict[str, int]:
        """Return starter roster structure."""
        return self._roster_structure

    def get_bench_maxes(self) -> Dict[str, int]:
        """Return bench slot maximums."""
        return self._bench_maxes

    def get_total_roster_size_per_team(self) -> int:
        """Return roster size cap per team."""
        return self._total_roster_size_per_team

    def add_player_to_roster(self, team_id: int, player: Player) -> None:
        """Add a player to team roster and update counts."""
        self._rosters[team_id]["PLAYERS"].append(player)
        self._available_player_ids.discard(player.player_id)
        self._update_roster_counts_for_pick(team_id, player)

    def _update_roster_counts_for_pick(self, team_id: int, player: Player) -> None:
        """Update positional counters after drafting a player."""
        roster = self._rosters[team_id]
        pos = player.position
        if roster.get(pos, 0) < self._roster_structure.get(pos, 0):
            roster[pos] = roster.get(pos, 0) + 1
            return
        if pos in ("RB", "WR", "TE") and roster.get("FLEX", 0) < self._roster_structure.get(
            "FLEX", 0
        ):
            roster["FLEX"] = roster.get("FLEX", 0) + 1
            return
        roster[pos] = roster.get(pos, 0) + 1

    def remove_player_from_roster(self, team_id: int, player: Player) -> None:
        """Remove a player from team roster and restore availability."""
        players_list = self._rosters[team_id]["PLAYERS"]
        for index, rostered_player in enumerate(players_list):
            if rostered_player.player_id == player.player_id:
                players_list.pop(index)
                break
        self._available_player_ids.add(player.player_id)
        self._decrement_roster_counts_for_pick(team_id, player)

    def _decrement_roster_counts_for_pick(self, team_id: int, player: Player) -> None:
        """Decrement positional counters after undoing a pick."""
        self._recalculate_roster_counts(team_id)

    def _recalculate_roster_counts(self, team_id: int) -> None:
        """Rebuild positional counters from rostered players.

        Notes
        -----
        This keeps FLEX/position counts consistent when removing a player that
        may have occupied a FLEX slot.
        """
        roster = self._rosters[team_id]
        players = list(roster["PLAYERS"])
        roster["QB"] = 0
        roster["RB"] = 0
        roster["WR"] = 0
        roster["TE"] = 0
        roster["FLEX"] = 0
        for rostered_player in players:
            self._update_roster_counts_for_pick(team_id, rostered_player)

    def advance_pick(self) -> None:
        """Advance to next pick."""
        self._current_pick_idx += 1
        self._current_pick_number += 1

    def set_current_pick_idx(self, idx: int) -> None:
        """Set current pick index."""
        self._current_pick_idx = idx

    def set_current_pick_number(self, number: int) -> None:
        """Set current pick number."""
        self._current_pick_number = number

    def set_agent_team_id(self, team_id: int) -> None:
        """Set perspective team id."""
        self._agent_team_id = team_id

    def set_overridden_team_id(self, team_id: Optional[int]) -> None:
        """Set manual team override."""
        self._overridden_team_id = team_id

    def append_draft_history(self, entry: Dict[str, Any]) -> None:
        """Append a draft history entry."""
        self._draft_history.append(entry)

    def pop_draft_history(self) -> Optional[Dict[str, Any]]:
        """Pop most recent draft history entry."""
        return self._draft_history.pop() if self._draft_history else None

    def load_from_serialized(self, data: dict, player_factory) -> None:
        """Load state from serialized structure."""
        self._available_player_ids = set(data.get("available_players_ids", []))
        self._rosters = defaultdict(
            lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "PLAYERS": []}
        )
        for team_id_str, roster_data in data.get("teams_rosters", {}).items():
            try:
                team_id = int(team_id_str)
                self._rosters[team_id] = {
                    "QB": roster_data.get("QB", 0),
                    "RB": roster_data.get("RB", 0),
                    "WR": roster_data.get("WR", 0),
                    "TE": roster_data.get("TE", 0),
                    "FLEX": roster_data.get("FLEX", 0),
                    "PLAYERS": [player_factory(player) for player in roster_data.get("PLAYERS", [])],
                }
            except (TypeError, ValueError):
                continue
        self._draft_order = data.get("draft_order", [])
        self._current_pick_idx = data.get("current_pick_idx", 0)
        self._current_pick_number = data.get("current_pick_number", 1)
        self._draft_history = data.get("_draft_history", [])
        self._overridden_team_id = data.get("_overridden_team_id")

    def reset(self, all_player_ids: Set[int], draft_order: List[int], agent_team_id: int) -> None:
        """Reset draft state to a new draft."""
        self._available_player_ids = set(all_player_ids)
        self._rosters = defaultdict(
            lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "PLAYERS": []}
        )
        self._draft_order = draft_order
        self._current_pick_idx = 0
        self._current_pick_number = 1
        self._agent_team_id = agent_team_id
        self._draft_history = []
        self._overridden_team_id = None
