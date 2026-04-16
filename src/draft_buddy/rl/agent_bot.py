"""PyTorch-backed BotGM implementation for RL inference."""

from __future__ import annotations

from typing import Dict, Optional

from draft_buddy.core.bot_gm import BotGM
from draft_buddy.core.entities import Player, PlayerCatalog, TeamRoster


class AgentModelBotGM(BotGM):
    """Use a trained policy model to choose a draft pick.

    Parameters
    ----------
    model : object
        Trained policy model exposing ``get_action_probabilities``.
    action_to_position : Dict[int, str]
        Mapping from action index to draft position.
    """

    def __init__(self, model, action_to_position: Dict[int, str]) -> None:
        self._model = model
        self._action_to_position = action_to_position

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_catalog: PlayerCatalog,
        team_roster: TeamRoster,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        build_state_fn=None,
        get_action_mask_fn=None,
        **kwargs,
    ) -> Optional[Player]:
        """Execute one model-driven pick.

        Parameters
        ----------
        team_id : int
            Drafting team id.
        available_player_ids : set
            Available player ids.
        player_catalog : PlayerCatalog
            Shared player catalog.
        team_roster : TeamRoster
            Current team roster.
        roster_structure : dict
            Required starter counts.
        bench_maxes : dict
            Bench limits by position.
        can_draft_position_fn : callable
            Position validation callback.
        try_select_player_fn : callable
            Player selection callback.
        build_state_fn : callable, optional
            Builds model input state for team.
        get_action_mask_fn : callable, optional
            Produces valid-action mask for team.

        Returns
        -------
        Optional[Player]
            Selected player if one can be drafted.
        """
        import torch

        if build_state_fn is None or get_action_mask_fn is None:
            return None
        state = build_state_fn(team_id)
        action_mask = get_action_mask_fn(team_id)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = self._model.get_action_probabilities(state_tensor, action_mask=action_mask)
            action = torch.argmax(action_probs).item()
        selected_position = self._action_to_position[action]
        is_valid, drafted_player = try_select_player_fn(team_id, selected_position, available_player_ids)
        if is_valid:
            return drafted_player
        eligible = [
            player_catalog.require(player_id)
            for player_id in available_player_ids
            if player_catalog.get(player_id)
            and player_catalog.require(player_id).position in {"QB", "RB", "WR", "TE"}
            and can_draft_position_fn(
                team_id, player_catalog.require(player_id).position, False
            )
        ]
        return min(eligible, key=lambda player: player.adp) if eligible else None
