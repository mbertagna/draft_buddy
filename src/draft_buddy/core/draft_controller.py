"""Shared draft orchestration for web and RL runtimes."""

from __future__ import annotations

import json
import os
import random
from typing import Callable, Optional

import numpy as np

from draft_buddy.core.bot_gm import BotGM
from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.entities import Pick, Player, PlayerCatalog
from draft_buddy.core.rules_engine import RulesEngine


class DraftController:
    """Own draft workflow on top of ``DraftState`` and ``PlayerCatalog``.

    Parameters
    ----------
    state : DraftState
        Mutable draft state.
    player_catalog : PlayerCatalog
        Shared player catalog.
    rules_engine : RulesEngine
        Rules validator.
    action_to_position : dict[int, str]
        Action index to position mapping.
    bot_factory : callable, optional
        Lazy factory returning a ``BotGM`` for a team.
    """

    def __init__(
        self,
        state: DraftState,
        player_catalog: PlayerCatalog,
        rules_engine: RulesEngine,
        action_to_position: dict[int, str],
        bot_factory: Optional[Callable[[int], BotGM]] = None,
    ) -> None:
        self.state = state
        self.player_catalog = player_catalog
        self.rules_engine = rules_engine
        self.action_to_position = dict(action_to_position)
        self._bot_factory = bot_factory
        self._bots: dict[int, BotGM] = {}

    @property
    def available_player_ids(self) -> set[int]:
        """Return current available player ids."""
        return self.state.available_player_ids

    @property
    def team_rosters(self):
        """Return typed team rosters."""
        return self.state.team_rosters

    @property
    def draft_order(self) -> list[int]:
        """Return global draft order."""
        return self.state.draft_order

    @property
    def current_pick_index(self) -> int:
        """Return current pick index."""
        return self.state.current_pick_index

    @property
    def current_pick_number(self) -> int:
        """Return current pick number."""
        return self.state.current_pick_number

    @property
    def team_on_clock(self) -> Optional[int]:
        """Return the team currently on the clock."""
        if self.state.override_team_id is not None:
            return self.state.override_team_id
        if self.current_pick_index >= len(self.draft_order):
            return None
        return self.draft_order[self.current_pick_index]

    def reset(self, draft_order: list[int], agent_team_id: int) -> None:
        """Reset draft state and clear cached bots."""
        self.state.reset(set(self.player_catalog.player_ids), draft_order, agent_team_id)
        self._bots = {}

    def save_state(self, file_path: str) -> None:
        """Persist state atomically to disk."""
        payload = self.state.to_dict()
        temp_file_path = f"{file_path}.tmp"
        with open(temp_file_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
        os.replace(temp_file_path, file_path)

    def load_state(self, file_path: str) -> None:
        """Load state from disk when a saved state exists."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as file_obj:
            self.state.load_from_dict(json.load(file_obj))
        for team_id in list(self.team_rosters.keys()):
            self.state.recalculate_roster_counts(team_id, self.player_catalog.require)

    def can_draft_position(self, team_id: int, position: str, is_manual: bool = False) -> bool:
        """Return whether a team can draft a position."""
        if is_manual:
            return self.rules_engine.can_draft_manual(self.state, team_id, position, self.player_catalog)
        return self.rules_engine.can_draft_simulated(self.state, team_id, position, self.player_catalog)

    def get_action_mask_for_team(self, team_id: int) -> np.ndarray:
        """Return valid-action mask for one team."""
        mask = np.zeros(len(self.action_to_position), dtype=bool)
        for action, position in self.action_to_position.items():
            mask[action] = self.can_draft_position(team_id, position, is_manual=False)
        return mask

    def try_select_player_for_team(
        self,
        team_id: int,
        position_choice: str,
        available_player_ids: Optional[set[int]] = None,
    ) -> tuple[bool, Optional[Player]]:
        """Return the best available player for a valid position choice."""
        candidate_ids = available_player_ids or self.available_player_ids
        eligible_players = [
            self.player_catalog.require(player_id)
            for player_id in candidate_ids
            if self.player_catalog.require(player_id).position == position_choice
        ]
        if not eligible_players:
            return False, None
        if not self.can_draft_position(team_id, position_choice, is_manual=False):
            return False, None
        return True, max(eligible_players, key=lambda player: player.projected_points)

    def draft_player(self, player_id: int, is_manual_pick: bool = True) -> None:
        """Apply a pick for the team currently on the clock."""
        team_id = self.team_on_clock
        if team_id is None:
            raise ValueError("The draft has already concluded. No more picks can be made.")
        player = self.player_catalog.get(player_id)
        if player is None or player_id not in self.available_player_ids:
            self.state.override_team_id = None
            raise ValueError(f"Player with ID {player_id} is not available to be drafted.")
        if not self.can_draft_position(team_id, player.position, is_manual=is_manual_pick):
            self.state.override_team_id = None
            raise ValueError(f"Team {team_id} cannot draft a {player.position}.")
        self.apply_pick(
            team_id=team_id,
            player_id=player_id,
            is_manual_pick=is_manual_pick,
            previous_override_team_id=self.state.override_team_id,
        )

    def undo_last_pick(self) -> None:
        """Undo the most recent pick."""
        last_pick = self.state.pop_pick()
        if last_pick is None:
            raise ValueError("No picks to undo.")
        player = self.player_catalog.require(last_pick.player_id)
        self.state.remove_player_from_roster(last_pick.team_id, player)
        self.state.recalculate_roster_counts(last_pick.team_id, self.player_catalog.require)
        self.state.current_pick_index = last_pick.previous_pick_index
        self.state.current_pick_number = last_pick.pick_number
        self.state.override_team_id = last_pick.previous_override_team_id

    def set_override_team(self, team_id: int) -> None:
        """Override the next team on the clock."""
        self.state.override_team_id = team_id

    def simulate_single_pick(
        self,
        manual_draft_teams: set[int],
        build_state_fn=None,
        get_action_mask_fn=None,
    ) -> Player:
        """Simulate a single non-manual team pick."""
        team_id = self.team_on_clock
        if team_id is None:
            raise ValueError("The draft has already concluded. No more picks can be made.")
        self.state.override_team_id = None
        if team_id in manual_draft_teams:
            raise ValueError("It is a manual team's turn. Cannot simulate pick.")
        selected_player = self._select_bot_pick(
            team_id=team_id,
            build_state_fn=build_state_fn,
            get_action_mask_fn=get_action_mask_fn,
        )
        if selected_player is None:
            raise ValueError(f"Team {team_id} could not make a valid pick.")
        self.apply_pick(team_id=team_id, player_id=selected_player.player_id, is_manual_pick=False)
        return selected_player

    def simulate_remaining(
        self,
        manual_draft_teams: set[int],
        build_state_fn=None,
        get_action_mask_fn=None,
    ) -> None:
        """Simulate remaining scheduled picks."""
        while self.current_pick_index < len(self.draft_order):
            self.simulate_single_pick(
                manual_draft_teams=manual_draft_teams,
                build_state_fn=build_state_fn,
                get_action_mask_fn=get_action_mask_fn,
            )

    def resolve_roster_players(self, team_id: int) -> list[Player]:
        """Return resolved players for one team roster."""
        return self.player_catalog.resolve(self.state.roster_for_team(team_id).player_ids)

    def apply_pick(
        self,
        team_id: int,
        player_id: int,
        is_manual_pick: bool,
        previous_override_team_id: Optional[int] = None,
    ) -> None:
        """Apply a known legal pick for a specific team."""
        player = self.player_catalog.require(player_id)
        self.state.append_pick(
            Pick(
                pick_number=self.current_pick_number,
                team_id=team_id,
                player_id=player_id,
                is_manual_pick=is_manual_pick,
                previous_pick_index=self.current_pick_index,
                previous_override_team_id=previous_override_team_id,
            )
        )
        self.state.add_player_to_roster(team_id, player)
        self.state.advance_pick()
        self.state.override_team_id = None

    def get_positional_baselines(self) -> dict[str, float]:
        """Return smoothed replacement baselines by position."""
        baselines: dict[str, float] = {}
        team_ids = sorted(set(self.draft_order) | set(self.team_rosters.keys()))
        for position in self.action_to_position.values():
            available = sorted(
                [
                    self.player_catalog.require(player_id)
                    for player_id in self.available_player_ids
                    if self.player_catalog.require(player_id).position == position
                ],
                key=lambda player: player.projected_points,
                reverse=True,
            )
            if not available:
                baselines[position] = 0.0
                continue
            needed_starters = 0
            required = self.state.roster_structure.get(position, 0)
            for team_id in team_ids:
                roster = self.state.roster_for_team(team_id)
                needed_starters += max(0, required - roster.position_count(position))
            replacement_index = min(max(0, needed_starters + 1), len(available) - 1)
            before = available[max(0, replacement_index - 1)].projected_points
            current = available[replacement_index].projected_points
            after = available[min(len(available) - 1, replacement_index + 1)].projected_points
            baselines[position] = (before + current + after) / 3.0
        return baselines

    def _select_bot_pick(self, team_id: int, build_state_fn=None, get_action_mask_fn=None) -> Optional[Player]:
        """Return the simulated pick for a bot team."""
        strategy = self._get_bot(team_id)
        if strategy is not None:
            chosen = strategy.execute_pick(
                team_id=team_id,
                available_player_ids=self.available_player_ids,
                player_catalog=self.player_catalog,
                team_roster=self.state.roster_for_team(team_id),
                roster_structure=self.state.roster_structure,
                bench_maxes=self.state.bench_maxes,
                can_draft_position_fn=self.can_draft_position,
                try_select_player_fn=self.try_select_player_for_team,
                build_state_fn=build_state_fn,
                get_action_mask_fn=get_action_mask_fn or self.get_action_mask_for_team,
            )
            if chosen is not None:
                return chosen
        eligible_players = [
            self.player_catalog.require(player_id)
            for player_id in self.available_player_ids
            if self.player_catalog.require(player_id).position in {"QB", "RB", "WR", "TE"}
            and self.can_draft_position(
                team_id, self.player_catalog.require(player_id).position, is_manual=False
            )
        ]
        return random.choice(eligible_players) if eligible_players else None

    def _get_bot(self, team_id: int) -> Optional[BotGM]:
        """Return or lazily build a bot for a team."""
        if team_id in self._bots:
            return self._bots[team_id]
        if self._bot_factory is None:
            return None
        self._bots[team_id] = self._bot_factory(team_id)
        return self._bots[team_id]
