"""
Draft rules validation for the fantasy football draft environment.

Stateless service that validates moves. Receives DraftState and a proposed
action to determine legality.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from draft_buddy.draft_env.draft_state import DraftState


class DraftRules(ABC):
    """
    Defines the contract for validating draft actions.
    """

    @abstractmethod
    def can_draft_manual(
        self, state: "DraftState", team_id: int, position: str, player_map: dict
    ) -> bool:
        """
        Validates if a manual user can draft a specific position.

        Parameters
        ----------
        state : DraftState
            The current state of the draft.
        team_id : int
            The identifier of the team attempting the pick.
        position : str
            The position the team wants to draft.
        player_map : dict
            Mapping of player_id to Player for availability checks.

        Returns
        -------
        bool
            True if the pick is legal under manual constraints, False otherwise.
        """
        pass

    @abstractmethod
    def can_draft_simulated(
        self, state: "DraftState", team_id: int, position: str, player_map: dict
    ) -> bool:
        """
        Validates if an automated agent can draft a specific position.

        Parameters
        ----------
        state : DraftState
            The current state of the draft.
        team_id : int
            The identifier of the team attempting the pick.
        position : str
            The position the team wants to draft.
        player_map : dict
            Mapping of player_id to Player for availability checks.

        Returns
        -------
        bool
            True if the pick is legal under simulation constraints, False otherwise.
        """
        pass


class FantasyDraftRules(DraftRules):
    """
    Concrete implementation of draft rules for fantasy football.
    """

    def __init__(
        self,
        roster_structure: dict,
        bench_maxes: dict,
        total_roster_size_per_team: int,
    ):
        """
        Initialize with roster configuration.

        Parameters
        ----------
        roster_structure : dict
            Required starters per position.
        bench_maxes : dict
            Maximum bench slots per position.
        total_roster_size_per_team : int
            Total roster size.
        """
        self._roster_structure = roster_structure
        self._bench_maxes = bench_maxes
        self._total_roster_size = total_roster_size_per_team

    def _has_position_available(self, position: str, available_ids: set, player_map: dict) -> bool:
        """Checks if any player of the given position is available."""
        return any(
            player_map.get(pid) and player_map[pid].position == position
            for pid in available_ids
        )

    def _validate_manual_constraints(
        self, roster_counts: dict, position: str, current_bench: int, total_starters: int
    ) -> bool:
        """Validates constraints specifically for manual picks."""
        if roster_counts.get(position, 0) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and roster_counts.get("FLEX", 0) < self._roster_structure.get("FLEX", 0):
            return True
        if current_bench < (self._total_roster_size - total_starters):
            return True
        return False

    def _validate_simulated_constraints(
        self, roster_counts: dict, position: str, current_bench: int, total_starters: int
    ) -> bool:
        """Validates constraints specifically for simulated/AI picks."""
        if roster_counts.get(position, 0) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and roster_counts.get("FLEX", 0) < self._roster_structure.get("FLEX", 0):
            return True
        pos_max = self._roster_structure.get(position, 0) + self._bench_maxes.get(position, 0)
        bench_max = self._total_roster_size - total_starters
        if roster_counts.get(position, 0) < pos_max and current_bench < bench_max:
            return True
        return False

    def can_draft_manual(
        self, state: "DraftState", team_id: int, position: str, player_map: dict
    ) -> bool:
        """Validates manual pick."""
        rosters = state.get_rosters()
        roster_counts = rosters[team_id]
        current_total = len(roster_counts["PLAYERS"])
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters

        if current_total >= self._total_roster_size:
            return False

        if not self._has_position_available(
            position, state.get_available_player_ids(), player_map
        ):
            return False

        return self._validate_manual_constraints(roster_counts, position, current_bench, total_starters)

    def can_draft_simulated(
        self, state: "DraftState", team_id: int, position: str, player_map: dict
    ) -> bool:
        """Validates simulated/AI pick."""
        rosters = state.get_rosters()
        roster_counts = rosters[team_id]
        current_total = len(roster_counts["PLAYERS"])
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters

        if current_total >= self._total_roster_size:
            return False

        if not self._has_position_available(
            position, state.get_available_player_ids(), player_map
        ):
            return False

        return self._validate_simulated_constraints(roster_counts, position, current_bench, total_starters)
