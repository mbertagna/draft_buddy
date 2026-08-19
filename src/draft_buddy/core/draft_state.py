"""Core mutable draft state stored as typed ids and counters."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from draft_buddy.core.entities import DraftAction, Pick, Player, Swap, TeamRoster, Transfer


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
    def shelved_player_ids(self) -> set[int]:
        """Return player ids tucked away from the draftable pool."""
        return self._shelved_player_ids

    @shelved_player_ids.setter
    def shelved_player_ids(self, player_ids: set[int]) -> None:
        """Replace the set of shelved player ids."""
        self._shelved_player_ids = set(player_ids)

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

    @property
    def transfer_history(self) -> list[Transfer]:
        """Return typed transfer history."""
        return self._transfer_history

    @property
    def swap_history(self) -> list[Swap]:
        """Return typed swap history."""
        return self._swap_history

    @property
    def action_history(self) -> list[DraftAction]:
        """Return chronological undo-stack actions."""
        return self._action_history

    @property
    def visual_board(self) -> dict[int, dict[int, Optional[int]]]:
        """Return team -> round -> player_id visual placement grid."""
        return self._visual_board

    def roster_for_team(self, team_id: int) -> TeamRoster:
        """Return one team's roster, creating it on first access."""
        return self._team_rosters[team_id]

    def add_player_to_roster(self, team_id: int, player: Player) -> None:
        """Add a player id to a team roster and update counts."""
        roster = self.roster_for_team(team_id)
        roster.player_ids.append(player.player_id)
        self.available_player_ids.discard(player.player_id)
        self._update_roster_counts_for_pick(team_id, player)

    def remove_player_from_roster(
        self, team_id: int, player: Player, restore_availability: bool = True
    ) -> None:
        """Remove a player id from roster and restore availability."""
        roster = self.roster_for_team(team_id)
        roster.player_ids = [
            player_id for player_id in roster.player_ids if player_id != player.player_id
        ]
        if restore_availability:
            self.available_player_ids.add(player.player_id)

    def find_player_team_id(self, player_id: int) -> Optional[int]:
        """Return the team currently rostering a player id."""
        for team_id, roster in self.team_rosters.items():
            if player_id in roster.player_ids:
                return team_id
        return None

    def move_player_between_rosters(
        self, from_team_id: int, to_team_id: int, player: Player
    ) -> None:
        """Move a drafted player between rosters without changing availability."""
        self.remove_player_from_roster(from_team_id, player, restore_availability=False)
        self.roster_for_team(to_team_id).player_ids.append(player.player_id)

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

    def ensure_visual_board(
        self, num_teams: Optional[int] = None, rounds: Optional[int] = None
    ) -> None:
        """Ensure the visual board has empty cells for every team and round.

        Parameters
        ----------
        num_teams : int, optional
            Highest team id to include. Defaults to max draft-order team id.
        rounds : int, optional
            Number of visual rounds. Defaults to ``total_roster_size_per_team``.
        """
        team_count = num_teams
        if team_count is None:
            team_ids = self._team_ids_for_board()
            team_count = max(team_ids) if team_ids else 0
        round_count = rounds if rounds is not None else self.total_roster_size_per_team
        for team_id in range(1, team_count + 1):
            if team_id not in self._visual_board:
                self._visual_board[team_id] = {}
            for round_index in range(round_count):
                if round_index not in self._visual_board[team_id]:
                    self._visual_board[team_id][round_index] = None

    def find_player_cell(self, player_id: int) -> Optional[tuple[int, int]]:
        """Return ``(team_id, round)`` for a player on the visual board.

        Parameters
        ----------
        player_id : int
            Player to locate.

        Returns
        -------
        tuple[int, int] or None
            Team and round coordinates when found.
        """
        for team_id, rounds in self._visual_board.items():
            for round_index, cell_player_id in rounds.items():
                if cell_player_id == player_id:
                    return team_id, round_index
        return None

    def first_empty_round(self, team_id: int) -> Optional[int]:
        """Return the lowest empty round index for a team.

        Parameters
        ----------
        team_id : int
            Team column to search.

        Returns
        -------
        int or None
            First empty round, or ``None`` when the column is full.
        """
        rounds = self._visual_board.get(team_id, {})
        for round_index in sorted(rounds.keys()):
            if rounds[round_index] is None:
                return round_index
        return None

    def place_player_visual(self, team_id: int, round_index: int, player_id: int) -> None:
        """Place a player id into one visual-board cell.

        Parameters
        ----------
        team_id : int
            Destination team column.
        round_index : int
            Destination round row.
        player_id : int
            Player to place.
        """
        if team_id not in self._visual_board:
            self._visual_board[team_id] = {}
        self._visual_board[team_id][round_index] = player_id

    def clear_cell(self, team_id: int, round_index: int) -> None:
        """Clear one visual-board cell.

        Parameters
        ----------
        team_id : int
            Team column.
        round_index : int
            Round row.
        """
        if team_id in self._visual_board and round_index in self._visual_board[team_id]:
            self._visual_board[team_id][round_index] = None

    def cell_player_id(self, team_id: int, round_index: int) -> Optional[int]:
        """Return the player id in a visual-board cell.

        Parameters
        ----------
        team_id : int
            Team column.
        round_index : int
            Round row.

        Returns
        -------
        int or None
            Occupying player id, or ``None`` when empty or missing.
        """
        return self._visual_board.get(team_id, {}).get(round_index)

    def advance_pick(self) -> None:
        """Advance the draft cursor to the next pick."""
        self.current_pick_index += 1
        self.current_pick_number += 1

    def append_pick(self, pick: Pick) -> None:
        """Append one pick to draft history."""
        self._draft_history.append(pick)

    def append_transfer(self, transfer: Transfer) -> None:
        """Append one transfer to transfer history."""
        self._transfer_history.append(transfer)

    def append_swap(self, swap: Swap) -> None:
        """Append one swap to swap history."""
        self._swap_history.append(swap)

    def append_action(self, action: DraftAction) -> None:
        """Append one chronological undo action."""
        self._action_history.append(action)

    def pop_pick(self) -> Optional[Pick]:
        """Pop the latest pick from draft history when present."""
        return self._draft_history.pop() if self._draft_history else None

    def pop_transfer(self) -> Optional[Transfer]:
        """Pop the latest transfer from transfer history when present."""
        return self._transfer_history.pop() if self._transfer_history else None

    def pop_swap(self) -> Optional[Swap]:
        """Pop the latest swap from swap history when present."""
        return self._swap_history.pop() if self._swap_history else None

    def pop_action(self) -> Optional[DraftAction]:
        """Pop the latest chronological undo action when present."""
        return self._action_history.pop() if self._action_history else None

    def reset(self, all_player_ids: set[int], draft_order: list[int], agent_team_id: int) -> None:
        """Reset state to a fresh draft."""
        self._available_player_ids = set(all_player_ids)
        self._shelved_player_ids: set[int] = set()
        self._team_rosters = defaultdict(TeamRoster)
        self._draft_order = list(draft_order)
        self._current_pick_index = 0
        self._current_pick_number = 1
        self._agent_team_id = agent_team_id
        self._draft_history: list[Pick] = []
        self._transfer_history: list[Transfer] = []
        self._swap_history: list[Swap] = []
        self._action_history: list[DraftAction] = []
        self._override_team_id: Optional[int] = None
        self._visual_board: dict[int, dict[int, Optional[int]]] = {}
        self.ensure_visual_board()

    def to_dict(self) -> dict:
        """Serialize state to a JSON-friendly dictionary."""
        return {
            "available_player_ids": sorted(self.available_player_ids),
            "shelved_player_ids": sorted(self.shelved_player_ids),
            "team_rosters": {
                str(team_id): roster.to_dict() for team_id, roster in self.team_rosters.items()
            },
            "draft_order": list(self.draft_order),
            "current_pick_index": self.current_pick_index,
            "current_pick_number": self.current_pick_number,
            "draft_history": [pick.to_dict() for pick in self.draft_history],
            "transfer_history": [transfer.to_dict() for transfer in self.transfer_history],
            "swap_history": [swap.to_dict() for swap in self.swap_history],
            "action_history": [action.to_dict() for action in self.action_history],
            "override_team_id": self.override_team_id,
            "agent_team_id": self.agent_team_id,
            "visual_board": {
                str(team_id): {
                    str(round_index): player_id for round_index, player_id in rounds.items()
                }
                for team_id, rounds in self._visual_board.items()
            },
        }

    def load_from_dict(self, payload: dict) -> None:
        """Load state from serialized data."""
        self.available_player_ids = {
            int(player_id) for player_id in payload.get("available_player_ids", [])
        }
        self.shelved_player_ids = {
            int(player_id) for player_id in payload.get("shelved_player_ids", [])
        }
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
        self._transfer_history = [
            Transfer.from_dict(transfer_payload)
            for transfer_payload in payload.get("transfer_history", [])
        ]
        self._swap_history = [
            Swap.from_dict(swap_payload) for swap_payload in payload.get("swap_history", [])
        ]
        if "action_history" in payload:
            self._action_history = [
                DraftAction.from_dict(action_payload)
                for action_payload in payload.get("action_history", [])
            ]
        else:
            self._action_history = [
                DraftAction(action_type="pick", history_index=index)
                for index, _pick in enumerate(self._draft_history)
            ]
        self.override_team_id = payload.get("override_team_id")
        self.agent_team_id = int(payload.get("agent_team_id", self.agent_team_id))
        self._visual_board = {}
        if "visual_board" in payload:
            self._load_visual_board(payload.get("visual_board", {}))
        else:
            self._backfill_visual_board_from_rosters()
        self.ensure_visual_board()

    def _team_ids_for_board(self) -> list[int]:
        """Return team ids that should appear on the visual board."""
        team_ids = set(self._draft_order) | set(self._team_rosters.keys())
        return sorted(team_ids)

    def _load_visual_board(self, payload: dict) -> None:
        """Load visual board cells from a serialized payload."""
        for team_id_str, rounds_payload in payload.items():
            try:
                team_id = int(team_id_str)
            except (TypeError, ValueError):
                continue
            self._visual_board[team_id] = {}
            if not isinstance(rounds_payload, dict):
                continue
            for round_str, player_id in rounds_payload.items():
                try:
                    round_index = int(round_str)
                except (TypeError, ValueError):
                    continue
                self._visual_board[team_id][round_index] = (
                    int(player_id) if player_id is not None else None
                )

    def _backfill_visual_board_from_rosters(self) -> None:
        """Populate visual board densely from logical roster order."""
        self.ensure_visual_board()
        for team_id, roster in self._team_rosters.items():
            for round_index, player_id in enumerate(roster.player_ids):
                if round_index >= self.total_roster_size_per_team:
                    break
                self.place_player_visual(team_id, round_index, player_id)

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
