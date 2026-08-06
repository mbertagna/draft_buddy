"""Core draft rules abstractions and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.entities import PlayerCatalog


class RulesEngine(ABC):
    """Abstraction for validating draft moves."""

    @abstractmethod
    def can_draft_manual(
        self, state: DraftState, team_id: int, position: str, player_catalog: PlayerCatalog
    ) -> bool:
        """Return whether a manual pick is legal."""

    @abstractmethod
    def can_draft_simulated(
        self, state: DraftState, team_id: int, position: str, player_catalog: PlayerCatalog
    ) -> bool:
        """Return whether an automated pick is legal."""

    @abstractmethod
    def can_accept_transfer(self, state: DraftState, team_id: int, position: str) -> bool:
        """Return whether a team can receive a transferred player."""


class FantasyRulesEngine(RulesEngine):
    """Fantasy football roster and availability validation rules.

    Parameters
    ----------
    roster_structure : dict
        Dedicated starter slots by position (including FLEX).
    bench_maxes : dict
        Per-position sim bench limits for RL and bots.
    total_roster_size_per_team : int
        Maximum players per team roster.
    platform_bench_maxes : dict, optional
        Per-position platform hard bench limits for manual picks and
        transfers. Defaults to ``bench_maxes`` when omitted.
    """

    def __init__(
        self,
        roster_structure: dict,
        bench_maxes: dict,
        total_roster_size_per_team: int,
        platform_bench_maxes: Optional[dict] = None,
    ) -> None:
        self._roster_structure = roster_structure
        self._bench_maxes = bench_maxes
        self._platform_bench_maxes = (
            dict(platform_bench_maxes) if platform_bench_maxes is not None else dict(bench_maxes)
        )
        self._total_roster_size = total_roster_size_per_team

    def _has_position_available(
        self, position: str, available_ids: set[int], player_catalog: PlayerCatalog
    ) -> bool:
        """Return whether pool contains at least one player at a position."""
        return any(
            player_catalog.get(player_id) and player_catalog.require(player_id).position == position
            for player_id in available_ids
        )

    def _position_cap(self, position: str, bench_maxes: dict) -> int:
        """Return total players allowed at a position for the given bench map."""
        return self._roster_structure.get(position, 0) + bench_maxes.get(position, 0)

    def _validate_manual_constraints(self, team_roster, position: str, current_bench: int, total_starters: int) -> bool:
        """Validate manual constraints for a position.

        Manual picks may exceed sim targets but must stay under the platform
        hard cap and within total roster capacity.
        """
        if team_roster.position_count(position) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and team_roster.position_count("FLEX") < self._roster_structure.get(
            "FLEX", 0
        ):
            return True
        platform_cap = self._position_cap(position, self._platform_bench_maxes)
        if team_roster.position_count(position) >= platform_cap:
            return False
        return current_bench < (self._total_roster_size - total_starters)

    def _validate_simulated_constraints(
        self, team_roster, position: str, current_bench: int, total_starters: int
    ) -> bool:
        """Validate simulated constraints for a position."""
        if team_roster.position_count(position) < self._roster_structure.get(position, 0):
            return True
        if position in ["RB", "WR", "TE"] and team_roster.position_count("FLEX") < self._roster_structure.get(
            "FLEX", 0
        ):
            return True
        pos_max = self._position_cap(position, self._bench_maxes)
        bench_max = self._total_roster_size - total_starters
        return team_roster.position_count(position) < pos_max and current_bench < bench_max

    def can_draft_manual(
        self, state: DraftState, team_id: int, position: str, player_catalog: PlayerCatalog
    ) -> bool:
        """Validate manual pick legality."""
        roster = state.roster_for_team(team_id)
        current_total = roster.size
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters
        if current_total >= self._total_roster_size:
            return False
        if not self._has_position_available(position, state.available_player_ids, player_catalog):
            return False
        return self._validate_manual_constraints(roster, position, current_bench, total_starters)

    def can_draft_simulated(
        self, state: DraftState, team_id: int, position: str, player_catalog: PlayerCatalog
    ) -> bool:
        """Validate automated pick legality."""
        roster = state.roster_for_team(team_id)
        current_total = roster.size
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters
        if current_total >= self._total_roster_size:
            return False
        if not self._has_position_available(position, state.available_player_ids, player_catalog):
            return False
        return self._validate_simulated_constraints(
            roster, position, current_bench, total_starters
        )

    def can_accept_transfer(self, state: DraftState, team_id: int, position: str) -> bool:
        """Validate whether a transfer can add a player to a roster."""
        roster = state.roster_for_team(team_id)
        current_total = roster.size
        total_starters = sum(self._roster_structure.values())
        current_bench = current_total - total_starters
        if current_total >= self._total_roster_size:
            return False
        return self._validate_manual_constraints(roster, position, current_bench, total_starters)
