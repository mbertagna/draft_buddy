"""Core draft rules abstractions and implementations."""

from abc import ABC, abstractmethod

from draft_buddy.core.draft_state import DraftState


class RulesEngine(ABC):
    """Abstraction for validating draft moves."""

    @abstractmethod
    def can_draft_manual(
        self, state: DraftState, team_id: int, position: str, player_map: dict
    ) -> bool:
        """Return whether a manual pick is legal."""

    @abstractmethod
    def can_draft_simulated(
        self, state: DraftState, team_id: int, position: str, player_map: dict
    ) -> bool:
        """Return whether an automated pick is legal."""


class FantasyRulesEngine(RulesEngine):
    """Fantasy football roster and availability validation rules."""

    def __init__(
        self,
        roster_structure: dict,
        bench_maxes: dict,
        total_roster_size_per_team: int,
    ) -> None:
        self._roster_structure = roster_structure
        self._bench_maxes = bench_maxes
        self._total_roster_size = total_roster_size_per_team

    def _has_position_available(self, position: str, available_ids: set, player_map: dict) -> bool:
        """Return whether pool contains at least one player at a position."""
        return any(player_map.get(pid) and player_map[pid].position == position for pid in available_ids)

    def _validate_manual_constraints(
        self, roster_counts: dict, position: str, current_bench: int, total_starters: int
    ) -> bool:
        """Validate manual constraints for a position."""
        if roster_counts.get(position, 0) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and roster_counts.get("FLEX", 0) < self._roster_structure.get(
            "FLEX", 0
        ):
            return True
        return current_bench < (self._total_roster_size - total_starters)

    def _validate_simulated_constraints(
        self, roster_counts: dict, position: str, current_bench: int, total_starters: int
    ) -> bool:
        """Validate simulated constraints for a position."""
        if roster_counts.get(position, 0) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and roster_counts.get("FLEX", 0) < self._roster_structure.get(
            "FLEX", 0
        ):
            return True
        pos_max = self._roster_structure.get(position, 0) + self._bench_maxes.get(position, 0)
        bench_max = self._total_roster_size - total_starters
        return roster_counts.get(position, 0) < pos_max and current_bench < bench_max

    def can_draft_manual(
        self, state: DraftState, team_id: int, position: str, player_map: dict
    ) -> bool:
        """Validate manual pick legality."""
        roster_counts = state.get_rosters()[team_id]
        current_total = len(roster_counts["PLAYERS"])
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters
        if current_total >= self._total_roster_size:
            return False
        if not self._has_position_available(position, state.get_available_player_ids(), player_map):
            return False
        return self._validate_manual_constraints(roster_counts, position, current_bench, total_starters)

    def can_draft_simulated(
        self, state: DraftState, team_id: int, position: str, player_map: dict
    ) -> bool:
        """Validate automated pick legality."""
        roster_counts = state.get_rosters()[team_id]
        current_total = len(roster_counts["PLAYERS"])
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters
        if current_total >= self._total_roster_size:
            return False
        if not self._has_position_available(position, state.get_available_player_ids(), player_map):
            return False
        return self._validate_simulated_constraints(
            roster_counts, position, current_bench, total_starters
        )
