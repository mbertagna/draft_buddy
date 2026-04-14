"""Polymorphic draft bot implementations."""

from abc import ABC, abstractmethod
import random
from typing import Dict, List, Optional

from draft_buddy.domain.entities import Player


class BotGM(ABC):
    """Abstract draft bot interface."""

    @abstractmethod
    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ) -> Optional[Player]:
        """Pick a player for the current team."""


class RandomBotGM(BotGM):
    """Select a random eligible player."""

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ) -> Optional[Player]:
        """Execute a random valid pick."""
        handled_positions = set(roster_structure.keys()) - {"FLEX"}
        eligible = [
            player_map[pid]
            for pid in available_player_ids
            if player_map.get(pid)
            and player_map[pid].position in handled_positions
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]
        return random.choice(eligible) if eligible else None


class AdpBotGM(BotGM):
    """Pick by ADP with optional randomness."""

    def __init__(self, randomness_factor: float = 0.2, suboptimal_strategy: str = "NEXT_BEST_ADP"):
        self._randomness_factor = randomness_factor
        self._suboptimal_strategy = suboptimal_strategy

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ) -> Optional[Player]:
        """Execute an ADP-based pick."""
        eligible = [
            player_map[pid]
            for pid in available_player_ids
            if player_map.get(pid)
            and player_map[pid].position in {"QB", "RB", "WR", "TE"}
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]
        if not eligible:
            return None

        sorted_by_adp = sorted(eligible, key=lambda player: player.adp)
        best = sorted_by_adp[0]
        if random.random() >= self._randomness_factor:
            return best
        if self._suboptimal_strategy == "RANDOM_ELIGIBLE":
            return random.choice(eligible)
        if self._suboptimal_strategy == "NEXT_BEST_ADP" and len(sorted_by_adp) > 1:
            return sorted_by_adp[1]
        if self._suboptimal_strategy == "NEXT_BEST_HEURISTIC" and len(eligible) > 1:
            others = [player for player in eligible if player != best]
            return random.choice(others) if others else best
        return best


class HeuristicBotGM(BotGM):
    """Pick by starter needs first, then bench value."""

    def __init__(
        self,
        positional_priority: List[str],
        randomness_factor: float = 0.2,
        suboptimal_strategy: str = "NEXT_BEST_HEURISTIC",
    ) -> None:
        self._positional_priority = positional_priority
        self._randomness_factor = randomness_factor
        self._suboptimal_strategy = suboptimal_strategy

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ) -> Optional[Player]:
        """Execute a need-aware pick."""
        eligible = [
            player_map[pid]
            for pid in available_player_ids
            if player_map.get(pid)
            and player_map[pid].position in {"QB", "RB", "WR", "TE"}
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]
        if not eligible:
            return None
        best = self._best_heuristic(eligible, roster_counts, roster_structure, bench_maxes)
        if best is None:
            best = min(eligible, key=lambda player: player.adp)
        if random.random() >= self._randomness_factor:
            return best
        if self._suboptimal_strategy == "RANDOM_ELIGIBLE":
            return random.choice(eligible)
        if self._suboptimal_strategy == "NEXT_BEST_ADP":
            ordered = sorted(eligible, key=lambda player: player.adp)
            return ordered[1] if len(ordered) > 1 else best
        others = [player for player in eligible if player != best]
        return random.choice(others) if others else best

    def _best_heuristic(
        self, eligible: List[Player], roster_counts: dict, roster_structure: dict, bench_maxes: dict
    ) -> Optional[Player]:
        """Return best player using positional priority and roster needs."""
        for position in self._positional_priority:
            if roster_counts.get(position, 0) < roster_structure.get(position, 0):
                position_players = [player for player in eligible if player.position == position]
                if position_players:
                    return max(position_players, key=lambda player: player.projected_points)
        for position in self._positional_priority:
            position_max = roster_structure.get(position, 0) + bench_maxes.get(position, 0)
            if roster_counts.get(position, 0) < position_max:
                position_players = [player for player in eligible if player.position == position]
                if position_players:
                    return max(position_players, key=lambda player: player.projected_points)
        if roster_counts.get("FLEX", 0) < roster_structure.get("FLEX", 0):
            flex_players = [player for player in eligible if player.position in {"RB", "WR", "TE"}]
            if flex_players:
                return max(flex_players, key=lambda player: player.projected_points)
        return None


def create_bot_gm(
    logic: str,
    config: dict,
    opponent_models: Optional[Dict[int, object]] = None,
    team_id: Optional[int] = None,
    action_to_position: Optional[Dict[int, str]] = None,
) -> BotGM:
    """Factory for BotGM instances."""
    if logic == "RANDOM":
        return RandomBotGM()
    if logic == "ADP":
        return AdpBotGM(
            randomness_factor=config.get("randomness_factor", 0.2),
            suboptimal_strategy=config.get("suboptimal_strategy", "NEXT_BEST_ADP"),
        )
    return HeuristicBotGM(
        positional_priority=config.get("positional_priority", ["RB", "WR", "QB", "TE"]),
        randomness_factor=config.get("randomness_factor", 0.2),
        suboptimal_strategy=config.get("suboptimal_strategy", "NEXT_BEST_HEURISTIC"),
    )


OpponentStrategy = BotGM
RandomStrategy = RandomBotGM
AdpStrategy = AdpBotGM
HeuristicStrategy = HeuristicBotGM
create_opponent_strategy = create_bot_gm
