"""
Opponent draft strategies for the fantasy football draft environment.

Polymorphic strategy pattern replaces the monolithic if/elif block in
_simulate_competing_pick. Each strategy encapsulates its decision-making logic.
"""

from abc import ABC, abstractmethod
import random
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from draft_buddy.domain.entities import Player


class OpponentStrategy(ABC):
    """
    Abstract interface for opponent draft decision making.
    """

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
    ) -> Optional["Player"]:
        """
        Selects a player to draft based on the current state.

        Parameters
        ----------
        team_id : int
            The team making the pick.
        available_player_ids : set
            Set of available player IDs.
        player_map : dict
            Mapping of player_id to Player.
        roster_counts : dict
            Current roster counts for the team.
        roster_structure : dict
            Required starters per position.
        bench_maxes : dict
            Maximum bench slots per position.
        can_draft_position_fn : callable
            Function(team_id, position, is_manual) -> bool.
        try_select_player_fn : callable
            Function(team_id, position, available_ids) -> (bool, Optional[Player]).
        **kwargs : dict
            Additional context (e.g., global_features, action_to_position).

        Returns
        -------
        Player or None
            The player selected to be drafted, or None if no valid pick.
        """
        pass


class RandomStrategy(OpponentStrategy):
    """
    Selects a random eligible player.
    """

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn=None,
        try_select_player_fn=None,
        **kwargs,
    ) -> Optional["Player"]:
        """Picks a random eligible player."""
        handled_positions = set(roster_structure.keys()) - {"FLEX"}
        flex_eligible = {"RB", "WR", "TE"}
        all_draftable = handled_positions.union(flex_eligible)

        eligible = [
            player_map[pid]
            for pid in available_player_ids
            if player_map.get(pid) and player_map[pid].position in all_draftable
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]
        return random.choice(eligible) if eligible else None


class AdpStrategy(OpponentStrategy):
    """
    Selects the player with the lowest ADP available.
    Supports randomness for suboptimal picks.
    """

    def __init__(
        self,
        randomness_factor: float = 0.2,
        suboptimal_strategy: str = "NEXT_BEST_ADP",
    ):
        """
        Parameters
        ----------
        randomness_factor : float
            Probability of making a suboptimal pick.
        suboptimal_strategy : str
            'RANDOM_ELIGIBLE', 'NEXT_BEST_ADP', or 'NEXT_BEST_HEURISTIC'.
        """
        self.randomness_factor = randomness_factor
        self.suboptimal_strategy = suboptimal_strategy

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn=None,
        try_select_player_fn=None,
        **kwargs,
    ) -> Optional["Player"]:
        """Picks by ADP, with optional randomness."""
        eligible = self._get_eligible_players(
            team_id, available_player_ids, player_map,
            roster_structure, can_draft_position_fn
        )
        if not eligible:
            return None

        sorted_by_adp = sorted(eligible, key=lambda p: p.adp)
        best = sorted_by_adp[0]

        if random.random() < self.randomness_factor:
            return self._apply_suboptimal(best, eligible, sorted_by_adp)
        return best

    def _get_eligible_players(
        self, team_id, available_player_ids, player_map,
        roster_structure, can_draft_position_fn
    ) -> List["Player"]:
        """Returns list of eligible players for the team."""
        handled = set(roster_structure.keys()) - {"FLEX"}
        all_draftable = handled.union({"RB", "WR", "TE"})
        return [
            player_map[pid] for pid in available_player_ids
            if player_map.get(pid) and player_map[pid].position in all_draftable
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]

    def _apply_suboptimal(
        self, best: "Player", eligible: List["Player"], sorted_by_adp: List["Player"]
    ) -> "Player":
        """Applies suboptimal strategy when randomness triggers."""
        if self.suboptimal_strategy == "RANDOM_ELIGIBLE":
            return random.choice(eligible)
        if self.suboptimal_strategy == "NEXT_BEST_ADP" and len(sorted_by_adp) > 1:
            return sorted_by_adp[1]
        if self.suboptimal_strategy == "NEXT_BEST_HEURISTIC" and len(eligible) > 1:
            others = [p for p in eligible if p != best]
            return random.choice(others) if others else best
        return best


class HeuristicStrategy(OpponentStrategy):
    """
    Prioritizes by position need (starters first) then projected points.
    Uses configurable positional priority and supports randomness.
    """

    def __init__(
        self,
        positional_priority: List[str],
        randomness_factor: float = 0.2,
        suboptimal_strategy: str = "NEXT_BEST_HEURISTIC",
    ):
        """
        Parameters
        ----------
        positional_priority : List[str]
            Order of positions to fill (e.g., ['RB', 'WR', 'QB', 'TE']).
        randomness_factor : float
            Probability of making a suboptimal pick.
        suboptimal_strategy : str
            How to choose when being suboptimal.
        """
        self.positional_priority = positional_priority
        self.randomness_factor = randomness_factor
        self.suboptimal_strategy = suboptimal_strategy

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn=None,
        try_select_player_fn=None,
        **kwargs,
    ) -> Optional["Player"]:
        """Picks by heuristic: fill starters first, then bench."""
        eligible = self._get_eligible_players(
            team_id, available_player_ids, player_map,
            roster_structure, can_draft_position_fn
        )
        if not eligible:
            return None

        best = self._get_best_heuristic(eligible, roster_counts, roster_structure, bench_maxes)
        if not best:
            best = min(eligible, key=lambda p: p.adp)

        if random.random() < self.randomness_factor:
            if self.suboptimal_strategy == "RANDOM_ELIGIBLE":
                return random.choice(eligible)
            if self.suboptimal_strategy == "NEXT_BEST_ADP":
                sorted_adp = sorted(eligible, key=lambda p: p.adp)
                return sorted_adp[1] if len(sorted_adp) > 1 else best
            if self.suboptimal_strategy == "NEXT_BEST_HEURISTIC" and len(eligible) > 1:
                others = [p for p in eligible if p != best]
                return random.choice(others) if others else best
        return best

    def _get_eligible_players(
        self, team_id, available_player_ids, player_map,
        roster_structure, can_draft_position_fn
    ) -> List["Player"]:
        """Returns list of eligible players."""
        handled = set(roster_structure.keys()) - {"FLEX"}
        all_draftable = handled.union({"RB", "WR", "TE"})
        return [
            player_map[pid] for pid in available_player_ids
            if player_map.get(pid) and player_map[pid].position in all_draftable
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]

    def _get_best_heuristic(
        self, eligible: List["Player"], roster_counts: dict,
        roster_structure: dict, bench_maxes: dict
    ) -> Optional["Player"]:
        """Finds best pick by positional need then projected points."""
        for pos in self.positional_priority:
            if roster_counts.get(pos, 0) < roster_structure.get(pos, 0):
                pos_eligible = [p for p in eligible if p.position == pos]
                if pos_eligible:
                    return max(pos_eligible, key=lambda p: p.projected_points)

        for pos in self.positional_priority:
            max_pos = roster_structure.get(pos, 0) + bench_maxes.get(pos, 0)
            if roster_counts.get(pos, 0) < max_pos:
                pos_eligible = [p for p in eligible if p.position == pos]
                if pos_eligible:
                    return max(pos_eligible, key=lambda p: p.projected_points)

        if roster_counts.get("FLEX", 0) < roster_structure.get("FLEX", 0):
            flex_eligible = [p for p in eligible if p.position in ["RB", "WR", "TE"]]
            if flex_eligible:
                return max(flex_eligible, key=lambda p: p.projected_points)

        return None


class AgentModelStrategy(OpponentStrategy):
    """
    Uses a trained policy network to select the pick.
    """

    def __init__(self, model, action_to_position: Dict[int, str]):
        """
        Parameters
        ----------
        model : PolicyNetwork
            Trained policy network for action selection.
        action_to_position : Dict[int, str]
            Mapping of action index to position string.
        """
        self.model = model
        self.action_to_position = action_to_position

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_map: dict,
        roster_counts: dict,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn=None,
        try_select_player_fn=None,
        build_state_fn=None,
        get_action_mask_fn=None,
        **kwargs,
    ) -> Optional["Player"]:
        """Uses the agent model to select position, then picks best player."""
        import torch

        if build_state_fn is None or get_action_mask_fn is None or try_select_player_fn is None:
            return None

        opponent_state = build_state_fn(team_id)
        action_mask = get_action_mask_fn(team_id)
        state_tensor = torch.from_numpy(opponent_state).float().unsqueeze(0)

        with torch.no_grad():
            action_probs = self.model.get_action_probabilities(
                state_tensor, action_mask=action_mask
            )
            action_chosen = torch.argmax(action_probs).item()

        selected_position = self.action_to_position[action_chosen]
        is_valid, drafted_player = try_select_player_fn(
            team_id, selected_position, available_player_ids
        )
        if is_valid:
            return drafted_player

        eligible = [
            player_map[pid] for pid in available_player_ids
            if player_map.get(pid) and player_map[pid].position in {"QB", "RB", "WR", "TE"}
            and can_draft_position_fn(team_id, player_map[pid].position, False)
        ]
        if eligible:
            return min(eligible, key=lambda p: p.adp)
        return None


def create_opponent_strategy(
    logic: str,
    config: dict,
    opponent_models: Optional[Dict[int, object]] = None,
    team_id: Optional[int] = None,
    action_to_position: Optional[Dict[int, str]] = None,
) -> OpponentStrategy:
    """
    Factory to create opponent strategy from config.

    Parameters
    ----------
    logic : str
        'RANDOM', 'ADP', 'HEURISTIC', or 'AGENT_MODEL'.
    config : dict
        Strategy config (randomness_factor, suboptimal_strategy, positional_priority).
    opponent_models : dict, optional
        Mapping of team_id to PolicyNetwork for AGENT_MODEL.
    team_id : int, optional
        Team ID (needed for AGENT_MODEL lookup).
    action_to_position : dict, optional
        Action to position mapping (needed for AGENT_MODEL).

    Returns
    -------
    OpponentStrategy
        The configured strategy instance.
    """
    if logic == "RANDOM":
        return RandomStrategy()
    if logic == "ADP":
        return AdpStrategy(
            randomness_factor=config.get("randomness_factor", 0.2),
            suboptimal_strategy=config.get("suboptimal_strategy", "NEXT_BEST_ADP"),
        )
    if logic == "HEURISTIC":
        return HeuristicStrategy(
            positional_priority=config.get("positional_priority", ["RB", "WR", "QB", "TE"]),
            randomness_factor=config.get("randomness_factor", 0.2),
            suboptimal_strategy=config.get("suboptimal_strategy", "NEXT_BEST_HEURISTIC"),
        )
    if logic == "AGENT_MODEL" and opponent_models and team_id and team_id in opponent_models and action_to_position:
        return AgentModelStrategy(opponent_models[team_id], action_to_position)
    return HeuristicStrategy(
        positional_priority=config.get("positional_priority", ["RB", "WR", "QB", "TE"]),
        randomness_factor=config.get("randomness_factor", 0.2),
        suboptimal_strategy=config.get("suboptimal_strategy", "NEXT_BEST_ADP"),
    )
