"""
Draft state container for the fantasy football draft environment.

Maintains the mutable data structures for the draft. Exposes methods to mutate
state but contains no logic about whether a move is legal.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Any

from draft_buddy.domain.entities import Player


class DraftState:
    """
    Maintains the mutable data structures for the draft.

    Holds rosters, available players, draft order, and pick history.
    Exposes methods to mutate state but contains no validation logic.
    """

    def __init__(
        self,
        all_player_ids: Set[int],
        draft_order: List[int],
        roster_structure: Dict[str, int],
        bench_maxes: Dict[str, int],
        total_roster_size_per_team: int,
        agent_team_id: int = 1,
    ):
        """
        Initialize draft state.

        Parameters
        ----------
        all_player_ids : Set[int]
            Initial set of all player IDs in the draft pool.
        draft_order : List[int]
            List of team IDs in draft order for each pick.
        roster_structure : Dict[str, int]
            Required starters per position (QB, RB, WR, TE, FLEX).
        bench_maxes : Dict[str, int]
            Maximum bench slots per position.
        total_roster_size_per_team : int
            Total roster size (starters + bench).
        agent_team_id : int, optional
            The team ID controlled by the agent.
        """
        self._available_player_ids: Set[int] = set(all_player_ids)
        self._rosters: Dict[int, Dict] = defaultdict(
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
        """Returns the set of undrafted player IDs."""
        return self._available_player_ids

    def replace_available_player_ids(self, player_ids: Set[int]) -> None:
        """Replaces the available player IDs set (e.g., for temporary state override)."""
        self._available_player_ids = set(player_ids)

    def get_rosters(self) -> Dict[int, Dict]:
        """Returns the rosters for all teams."""
        return self._rosters

    def get_draft_order(self) -> List[int]:
        """Returns the draft order."""
        return self._draft_order

    def get_current_pick_idx(self) -> int:
        """Returns the current pick index in the draft order."""
        return self._current_pick_idx

    def get_current_pick_number(self) -> int:
        """Returns the global pick number (1-indexed)."""
        return self._current_pick_number

    def get_agent_team_id(self) -> int:
        """Returns the agent's team ID."""
        return self._agent_team_id

    def get_overridden_team_id(self) -> Optional[int]:
        """Returns the overridden team ID for manual pick override, if any."""
        return self._overridden_team_id

    def get_draft_history(self) -> List[Dict[str, Any]]:
        """Returns the draft history for undo support."""
        return self._draft_history

    def get_roster_structure(self) -> Dict[str, int]:
        """Returns the roster structure (starters per position)."""
        return self._roster_structure

    def get_bench_maxes(self) -> Dict[str, int]:
        """Returns the bench maxes per position."""
        return self._bench_maxes

    def get_total_roster_size_per_team(self) -> int:
        """Returns total roster size per team."""
        return self._total_roster_size_per_team

    def add_player_to_roster(self, team_id: int, player: Player) -> None:
        """
        Assigns a player to a team's roster and updates position counts.

        Parameters
        ----------
        team_id : int
            The team receiving the player.
        player : Player
            The player being drafted.
        """
        self._rosters[team_id]["PLAYERS"].append(player)
        self._available_player_ids.discard(player.player_id)
        self._update_roster_counts_for_pick(team_id, player)

    def _update_roster_counts_for_pick(self, team_id: int, player: Player) -> None:
        """Updates QB/RB/WR/TE/FLEX counts for a drafted player (starter, flex, or bench)."""
        roster = self._rosters[team_id]
        pos = player.position
        if roster.get(pos, 0) < self._roster_structure.get(pos, 0):
            roster[pos] = roster.get(pos, 0) + 1
            return
        if pos in ("RB", "WR", "TE") and roster.get("FLEX", 0) < self._roster_structure.get("FLEX", 0):
            roster["FLEX"] = roster.get("FLEX", 0) + 1
            return
        roster[pos] = roster.get(pos, 0) + 1

    def remove_player_from_roster(self, team_id: int, player: Player) -> None:
        """
        Removes a player from a team's roster and re-adds to available pool.

        Parameters
        ----------
        team_id : int
            The team losing the player.
        player : Player
            The player being removed (for undo).
        """
        players_list = self._rosters[team_id]["PLAYERS"]
        for i, p in enumerate(players_list):
            if p.player_id == player.player_id:
                players_list.pop(i)
                break
        self._available_player_ids.add(player.player_id)
        self._decrement_roster_counts_for_pick(team_id, player)

    def _decrement_roster_counts_for_pick(self, team_id: int, player: Player) -> None:
        """Decrements QB/RB/WR/TE/FLEX counts when undoing a pick."""
        roster = self._rosters[team_id]
        pos = player.position
        if roster.get(pos, 0) > 0:
            roster[pos] = roster.get(pos, 0) - 1
        elif pos in ("RB", "WR", "TE") and roster.get("FLEX", 0) > 0:
            roster["FLEX"] = roster.get("FLEX", 0) - 1

    def remove_player_from_pool(self, player_id: int) -> None:
        """
        Removes a player from the available pool.

        Parameters
        ----------
        player_id : int
            The player ID to remove.
        """
        self._available_player_ids.discard(player_id)

    def increment_roster_count(self, team_id: int, position: str, count: int = 1) -> None:
        """
        Increments the roster count for a position on a team.

        Parameters
        ----------
        team_id : int
            The team ID.
        position : str
            The position (QB, RB, WR, TE, FLEX).
        count : int, optional
            Amount to increment.
        """
        self._rosters[team_id][position] = self._rosters[team_id].get(position, 0) + count

    def decrement_roster_count(self, team_id: int, position: str, count: int = 1) -> None:
        """
        Decrements the roster count for a position on a team.

        Parameters
        ----------
        team_id : int
            The team ID.
        position : str
            The position (QB, RB, WR, TE, FLEX).
        count : int, optional
            Amount to decrement.
        """
        current = self._rosters[team_id].get(position, 0)
        self._rosters[team_id][position] = max(0, current - count)

    def advance_pick(self) -> None:
        """Advances the draft to the next pick."""
        self._current_pick_idx += 1
        self._current_pick_number += 1

    def set_current_pick_idx(self, idx: int) -> None:
        """Sets the current pick index."""
        self._current_pick_idx = idx

    def set_current_pick_number(self, number: int) -> None:
        """Sets the current pick number."""
        self._current_pick_number = number

    def set_agent_team_id(self, team_id: int) -> None:
        """Sets the agent's team ID."""
        self._agent_team_id = team_id

    def set_overridden_team_id(self, team_id: Optional[int]) -> None:
        """Sets the overridden team ID for manual pick."""
        self._overridden_team_id = team_id

    def append_draft_history(self, entry: Dict[str, Any]) -> None:
        """Appends an entry to the draft history."""
        self._draft_history.append(entry)

    def pop_draft_history(self) -> Optional[Dict[str, Any]]:
        """Pops and returns the last draft history entry."""
        return self._draft_history.pop() if self._draft_history else None

    def load_from_serialized(self, data: dict, player_factory) -> None:
        """
        Loads state from a serialized dict (e.g., from JSON file).

        Parameters
        ----------
        data : dict
            Dict with keys: available_players_ids, teams_rosters, draft_order,
            current_pick_idx, current_pick_number, _draft_history, _overridden_team_id.
        player_factory : callable
            Callable that takes a dict and returns a Player (e.g., lambda d: Player(**d)).
        """
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
                    "PLAYERS": [player_factory(p) for p in roster_data.get("PLAYERS", [])],
                }
            except (ValueError, TypeError):
                continue
        self._draft_order = data.get("draft_order", [])
        self._current_pick_idx = data.get("current_pick_idx", 0)
        self._current_pick_number = data.get("current_pick_number", 1)
        self._draft_history = data.get("_draft_history", [])
        self._overridden_team_id = data.get("_overridden_team_id")

    def reset(
        self,
        all_player_ids: Set[int],
        draft_order: List[int],
        agent_team_id: int,
    ) -> None:
        """
        Resets the draft state for a new episode.

        Parameters
        ----------
        all_player_ids : Set[int]
            Full set of player IDs for the new draft.
        draft_order : List[int]
            Draft order for the new episode.
        agent_team_id : int
            Agent's team ID for this episode.
        """
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
