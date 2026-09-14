"""Shared draft orchestration for web and RL runtimes."""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Callable, Optional

import numpy as np

from draft_buddy.core.bot_gm import BotGM
from draft_buddy.core.draft_invariants import assert_invariants
from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.draft_state_store import (
    iter_load_candidates,
    read_json_dict,
    save_draft_state,
)
from draft_buddy.core.entities import DraftAction, Pick, Player, PlayerCatalog, Swap, Transfer
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
        self._load_warning: Optional[str] = None

    @property
    def load_warning(self) -> Optional[str]:
        """Return the most recent load-recovery warning, if any."""
        return self._load_warning

    def consume_load_warning(self) -> Optional[str]:
        """Return and clear the most recent load-recovery warning.

        Returns
        -------
        str or None
            Warning text when recovery was used on the last load.
        """
        warning = self._load_warning
        self._load_warning = None
        return warning

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

    def reset(
        self,
        draft_order: list[int],
        agent_team_id: int,
        available_player_ids: Optional[set[int]] = None,
    ) -> None:
        """Reset draft state and clear cached bots.

        Parameters
        ----------
        draft_order : list[int]
            Team ids in snake draft order.
        agent_team_id : int
            Default agent perspective team id.
        available_player_ids : set[int], optional
            Subset of catalog ids that start available. Defaults to the full
            catalog. Shelved ids are always cleared.
        """
        pool_ids = (
            set(available_player_ids)
            if available_player_ids is not None
            else set(self.player_catalog.player_ids)
        )
        self.state.reset(pool_ids, draft_order, agent_team_id)
        self._bots = {}

    def save_state(self, file_path: str, prev_path: Optional[str] = None) -> None:
        """Persist state atomically to disk with an optional rolling previous copy.

        Parameters
        ----------
        file_path : str
            Primary draft state path.
        prev_path : str, optional
            Rolling previous-file path updated before overwrite.
        """
        save_draft_state(file_path, self.state.to_dict(), prev_path=prev_path)

    def load_state(
        self,
        file_path: str,
        prev_path: Optional[str] = None,
        saved_states_dir: Optional[str] = None,
    ) -> None:
        """Load state from disk with primary → prev → archive recovery.

        Parameters
        ----------
        file_path : str
            Primary draft state path.
        prev_path : str, optional
            Rolling previous-file path.
        saved_states_dir : str, optional
            Timestamped archive directory.

        Raises
        ------
        ValueError
            When candidate files exist but none load with valid invariants.
        """
        self._load_warning = None
        candidates = [
            (path, is_primary)
            for path, is_primary in iter_load_candidates(file_path, prev_path, saved_states_dir)
            if os.path.isfile(path)
        ]
        if not candidates:
            return

        errors: list[str] = []
        for path, is_primary in candidates:
            try:
                payload = read_json_dict(path)
                self.state.load_from_dict(payload)
                for team_id in list(self.team_rosters.keys()):
                    self.state.recalculate_roster_counts(team_id, self.player_catalog.require)
                assert_invariants(self.state)
                if not is_primary:
                    warning = (
                        "Draft state primary unreadable or invalid; "
                        f"recovered from {os.path.basename(path)}."
                    )
                    print(f"WARNING: {warning}")
                    self._load_warning = warning
                return
            except (ValueError, OSError, TypeError, KeyError) as error:
                errors.append(f"{path}: {error}")
                continue

        detail = "; ".join(errors) if errors else "no readable candidates"
        message = f"Could not load draft state from any candidate ({detail})."
        print(f"WARNING: {message}")
        self._load_warning = message
        raise ValueError(message)

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

    def shelve_players(self, player_ids: list[int] | set[int]) -> list[int]:
        """Move currently available players into the shelved sink.

        Parameters
        ----------
        player_ids : list[int] or set[int]
            Candidate player ids to shelve.

        Returns
        -------
        list of int
            Player ids that were actually shelved.
        """
        shelved: list[int] = []
        for player_id in player_ids:
            player_id = int(player_id)
            if player_id not in self.available_player_ids:
                continue
            if self.state.find_player_team_id(player_id) is not None:
                continue
            self.state.available_player_ids.discard(player_id)
            self.state.shelved_player_ids.add(player_id)
            shelved.append(player_id)
        return shelved

    def unshelve_players(self, player_ids: list[int] | set[int]) -> list[int]:
        """Restore currently shelved players to the available pool.

        Parameters
        ----------
        player_ids : list[int] or set[int]
            Candidate player ids to unshelve.

        Returns
        -------
        list of int
            Player ids that were actually unshelved.
        """
        restored: list[int] = []
        for player_id in player_ids:
            player_id = int(player_id)
            if player_id not in self.state.shelved_player_ids:
                continue
            if self.state.find_player_team_id(player_id) is not None:
                continue
            self.state.shelved_player_ids.discard(player_id)
            self.state.available_player_ids.add(player_id)
            restored.append(player_id)
        return restored

    def shelve_inactive_players(
        self,
        roster_statuses: list[str] | set[str],
        injury_statuses: list[str] | set[str],
    ) -> list[int]:
        """Shelve available players matching inactive roster or injury statuses.

        Parameters
        ----------
        roster_statuses : list[str] or set[str]
            Sleeper roster statuses treated as unavailable.
        injury_statuses : list[str] or set[str]
            Sleeper injury designations treated as unavailable.

        Returns
        -------
        list of int
            Player ids that were shelved.
        """
        from draft_buddy.data.player_filter import iter_inactive_player_ids

        inactive_ids = [
            player_id
            for player_id in iter_inactive_player_ids(
                self.player_catalog, roster_statuses, injury_statuses
            )
            if player_id in self.available_player_ids
        ]
        return self.shelve_players(inactive_ids)

    def shelve_players_above_adp(self, max_adp: float) -> list[int]:
        """Shelve available players with finite ADP strictly above a threshold.

        Parameters
        ----------
        max_adp : float
            Players with finite ``adp > max_adp`` are shelved. Non-finite ADP
            values are left untouched.

        Returns
        -------
        list of int
            Player ids that were shelved.
        """
        to_shelve: list[int] = []
        for player_id in list(self.available_player_ids):
            player = self.player_catalog.get(player_id)
            if player is None:
                continue
            if not np.isfinite(player.adp):
                continue
            if float(player.adp) > float(max_adp):
                to_shelve.append(player_id)
        return self.shelve_players(to_shelve)

    def undo_last_pick(self) -> None:
        """Undo the most recent draft action."""
        if not self.state.action_history:
            raise ValueError("No actions to undo.")
        last_action = self.state.action_history[-1]
        if last_action.action_type == "pick":
            self._undo_latest_pick_action(last_action)
            return
        if last_action.action_type == "transfer":
            self._undo_latest_transfer_action(last_action)
            return
        if last_action.action_type == "swap":
            self._undo_latest_swap_action(last_action)
            return
        raise ValueError(f"Cannot undo unknown action type: {last_action.action_type}.")

    def transfer_player(
        self, player_id: int, to_team_id: int, to_round: Optional[int] = None
    ) -> Transfer:
        """Transfer or visually reposition one drafted player.

        Parameters
        ----------
        player_id : int
            Drafted player to move.
        to_team_id : int
            Destination team id.
        to_round : int, optional
            Destination visual-board round. Defaults to the first empty slot.

        Returns
        -------
        Transfer
            Applied transfer record.
        """
        player = self.player_catalog.get(player_id)
        if player is None:
            raise ValueError(f"Unknown player id: {player_id}.")
        from_team_id = self.state.find_player_team_id(player_id)
        if from_team_id is None:
            raise ValueError(f"Player with ID {player_id} is not currently rostered.")
        if not self._is_valid_team_id(to_team_id):
            raise ValueError(f"Invalid team ID: {to_team_id}.")

        cell = self.state.find_player_cell(player_id)
        if cell is None:
            raise ValueError(f"Player with ID {player_id} has no visual board placement.")
        from_round = cell[1]

        resolved_round = to_round
        if resolved_round is None:
            resolved_round = self.state.first_empty_round(to_team_id)
            if resolved_round is None:
                raise ValueError(f"Team {to_team_id} has no empty visual board slots.")
        if not self._is_valid_round(to_team_id, resolved_round):
            raise ValueError(f"Invalid round index: {resolved_round}.")
        if from_team_id == to_team_id and from_round == resolved_round:
            raise ValueError("Player is already in the destination slot.")
        if self.state.cell_player_id(to_team_id, resolved_round) is not None:
            raise ValueError("Destination cell is occupied. Use swap to exchange players.")

        if from_team_id != to_team_id:
            if not self.rules_engine.can_accept_transfer(self.state, to_team_id, player.position):
                raise ValueError(f"Team {to_team_id} cannot receive a {player.position}.")
            self.state.move_player_between_rosters(from_team_id, to_team_id, player)
            self.state.recalculate_roster_counts(from_team_id, self.player_catalog.require)
            self.state.recalculate_roster_counts(to_team_id, self.player_catalog.require)

        self.state.clear_cell(from_team_id, from_round)
        self.state.place_player_visual(to_team_id, resolved_round, player_id)

        transfer = Transfer(
            player_id=player_id,
            from_team_id=from_team_id,
            to_team_id=to_team_id,
            from_round=from_round,
            to_round=resolved_round,
            previous_override_team_id=self.state.override_team_id,
        )
        self.state.append_transfer(transfer)
        self.state.append_action(
            DraftAction(action_type="transfer", history_index=len(self.state.transfer_history) - 1)
        )
        return transfer

    def swap_players(self, player_id_1: int, player_id_2: int) -> Swap:
        """Swap two drafted players' teams and visual-board slots.

        Parameters
        ----------
        player_id_1 : int
            First drafted player.
        player_id_2 : int
            Second drafted player.

        Returns
        -------
        Swap
            Applied swap record.
        """
        if player_id_1 == player_id_2:
            raise ValueError("Cannot swap a player with themselves.")
        player_1 = self.player_catalog.get(player_id_1)
        player_2 = self.player_catalog.get(player_id_2)
        if player_1 is None or player_2 is None:
            raise ValueError("Unknown player id in swap request.")
        cell_1 = self.state.find_player_cell(player_id_1)
        cell_2 = self.state.find_player_cell(player_id_2)
        if cell_1 is None or cell_2 is None:
            raise ValueError("Both players must have visual board placements to swap.")
        team_id_1, round_1 = cell_1
        team_id_2, round_2 = cell_2

        if team_id_1 != team_id_2:
            self.state.remove_player_from_roster(team_id_1, player_1, restore_availability=False)
            self.state.remove_player_from_roster(team_id_2, player_2, restore_availability=False)
            if not self.rules_engine.can_accept_transfer(self.state, team_id_1, player_2.position):
                self.state.roster_for_team(team_id_1).player_ids.append(player_id_1)
                self.state.roster_for_team(team_id_2).player_ids.append(player_id_2)
                self.state.recalculate_roster_counts(team_id_1, self.player_catalog.require)
                self.state.recalculate_roster_counts(team_id_2, self.player_catalog.require)
                raise ValueError(f"Team {team_id_1} cannot receive a {player_2.position}.")
            if not self.rules_engine.can_accept_transfer(self.state, team_id_2, player_1.position):
                self.state.roster_for_team(team_id_1).player_ids.append(player_id_1)
                self.state.roster_for_team(team_id_2).player_ids.append(player_id_2)
                self.state.recalculate_roster_counts(team_id_1, self.player_catalog.require)
                self.state.recalculate_roster_counts(team_id_2, self.player_catalog.require)
                raise ValueError(f"Team {team_id_2} cannot receive a {player_1.position}.")
            self.state.roster_for_team(team_id_1).player_ids.append(player_id_2)
            self.state.roster_for_team(team_id_2).player_ids.append(player_id_1)
            self.state.recalculate_roster_counts(team_id_1, self.player_catalog.require)
            self.state.recalculate_roster_counts(team_id_2, self.player_catalog.require)

        self.state.place_player_visual(team_id_1, round_1, player_id_2)
        self.state.place_player_visual(team_id_2, round_2, player_id_1)

        swap = Swap(
            player_id_1=player_id_1,
            team_id_1=team_id_1,
            round_1=round_1,
            player_id_2=player_id_2,
            team_id_2=team_id_2,
            round_2=round_2,
            previous_override_team_id=self.state.override_team_id,
        )
        self.state.append_swap(swap)
        self.state.append_action(
            DraftAction(action_type="swap", history_index=len(self.state.swap_history) - 1)
        )
        return swap

    def set_override_team(self, team_id: int) -> None:
        """Override the next team on the clock.

        Selecting the natural snake team clears any active override.
        """
        if self.current_pick_index >= len(self.draft_order):
            self.state.override_team_id = team_id
            return
        snake_team_id = self.draft_order[self.current_pick_index]
        if team_id == snake_team_id:
            self.state.override_team_id = None
            return
        self.state.override_team_id = team_id

    def simulate_single_pick(
        self,
        manual_draft_teams: set[int],
        build_state_fn=None,
        get_action_mask_fn=None,
        policy_bot: BotGM | None = None,
    ) -> Player:
        """Simulate a single non-manual team pick."""
        team_id = self.team_on_clock
        if team_id is None:
            raise ValueError("The draft has already concluded. No more picks can be made.")
        self.state.override_team_id = None
        if team_id in manual_draft_teams:
            raise ValueError("It is a manual team's turn. Cannot simulate pick.")
        selected_player = self._select_simulated_pick(
            team_id=team_id,
            build_state_fn=build_state_fn,
            get_action_mask_fn=get_action_mask_fn,
            policy_bot=policy_bot,
        )
        if selected_player is None:
            raise ValueError(self._format_no_valid_pick_error(team_id))
        self.apply_pick(team_id=team_id, player_id=selected_player.player_id, is_manual_pick=False)
        return selected_player

    def simulate_remaining(
        self,
        manual_draft_teams: set[int],
        build_state_fn=None,
        get_action_mask_fn=None,
        policy_bot: BotGM | None = None,
    ) -> None:
        """Fill empty board cells round-by-round in snake direction.

        Walks each visual round in original snake order and drafts only into
        empty cells for non-manual teams with roster room. Skips cells when
        no legal pick exists instead of aborting the run.
        """
        self.state.override_team_id = None
        team_ids = sorted(set(self.draft_order) | set(self.team_rosters.keys()))
        num_teams = max(team_ids) if team_ids else 0
        num_rounds = self.state.total_roster_size_per_team

        for round_index in range(num_rounds):
            walk_order = range(1, num_teams + 1)
            if round_index % 2 == 1:
                walk_order = reversed(list(walk_order))
            for team_id in walk_order:
                if team_id in manual_draft_teams:
                    continue
                if self.state.cell_player_id(team_id, round_index) is not None:
                    continue
                if self.state.roster_for_team(team_id).size >= self.state.total_roster_size_per_team:
                    continue
                selected_player = self._select_simulated_pick(
                    team_id=team_id,
                    build_state_fn=build_state_fn,
                    get_action_mask_fn=get_action_mask_fn,
                    policy_bot=policy_bot,
                )
                if selected_player is None:
                    continue
                self.apply_pick(
                    team_id=team_id,
                    player_id=selected_player.player_id,
                    is_manual_pick=False,
                    visual_round=round_index,
                )

        self.state.current_pick_index = len(self.draft_order)
        self.state.current_pick_number = len(self.draft_order) + 1

    def resolve_roster_players(self, team_id: int) -> list[Player]:
        """Return resolved players for one team roster."""
        return self.player_catalog.resolve(self.state.roster_for_team(team_id).player_ids)

    def apply_pick(
        self,
        team_id: int,
        player_id: int,
        is_manual_pick: bool,
        previous_override_team_id: Optional[int] = None,
        visual_round: Optional[int] = None,
    ) -> None:
        """Apply a known legal pick for a specific team.

        Parameters
        ----------
        team_id : int
            Team receiving the pick.
        player_id : int
            Player to draft.
        is_manual_pick : bool
            Whether the pick was made manually.
        previous_override_team_id : int, optional
            Override team id to restore on undo.
        visual_round : int, optional
            Explicit visual-board round. Defaults to the first empty round.
        """
        player = self.player_catalog.require(player_id)
        if self.state.roster_for_team(team_id).size >= self.state.total_roster_size_per_team:
            raise ValueError(f"Team {team_id} roster is full.")
        if visual_round is not None:
            target_round = visual_round
            if not self._is_valid_round(team_id, target_round):
                raise ValueError(f"Invalid round index: {target_round}.")
            if self.state.cell_player_id(team_id, target_round) is not None:
                raise ValueError(f"Team {team_id} round {target_round} is already occupied.")
        else:
            target_round = self.state.first_empty_round(team_id)
            if target_round is None:
                raise ValueError(f"Team {team_id} has no empty visual board slots.")

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
        self.state.append_action(
            DraftAction(action_type="pick", history_index=len(self.state.draft_history) - 1)
        )
        self.state.add_player_to_roster(team_id, player)
        self.state.place_player_visual(team_id, target_round, player_id)
        self.state.advance_pick()
        self.state.override_team_id = None

    def apply_display_pick(self, team_id: int, player_id: int) -> None:
        """Place a non-skill pick on the board without counting roster slots.

        Parameters
        ----------
        team_id : int
            Team receiving the display-only pick.
        player_id : int
            Placeholder or kicker/DST player id.
        """
        self.player_catalog.require(player_id)
        target_round = self.state.first_empty_round(team_id)
        if target_round is None:
            self.state.expand_visual_board_by(1)
            target_round = self.state.first_empty_round(team_id)
        if target_round is None:
            raise ValueError(f"Team {team_id} has no empty visual board slots.")

        self.state.append_pick(
            Pick(
                pick_number=self.current_pick_number,
                team_id=team_id,
                player_id=player_id,
                is_manual_pick=False,
                previous_pick_index=self.current_pick_index,
                previous_override_team_id=self.state.override_team_id,
            )
        )
        self.state.append_action(
            DraftAction(action_type="pick", history_index=len(self.state.draft_history) - 1)
        )
        self.state.available_player_ids.discard(player_id)
        self.state.shelved_player_ids.discard(player_id)
        self.state.display_only_player_ids.add(player_id)
        self.state.place_player_visual(team_id, target_round, player_id)
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

    def _format_no_valid_pick_error(self, team_id: int) -> str:
        """Build a diagnostic message when no legal simulated pick exists."""
        roster = self.state.roster_for_team(team_id)
        available_positions = Counter(
            self.player_catalog.require(player_id).position
            for player_id in self.available_player_ids
            if player_id in self.player_catalog
        )
        can_draft = {
            position: self.can_draft_position(team_id, position, is_manual=False)
            for position in ("QB", "RB", "WR", "TE")
        }
        team_ids = sorted(set(self.draft_order) | set(self.team_rosters.keys()))
        team_summaries = []
        for other_team_id in team_ids:
            other = self.state.roster_for_team(other_team_id)
            team_summaries.append(
                f"t{other_team_id}:size={other.size} "
                f"Q{other.qb_count}R{other.rb_count}W{other.wr_count}"
                f"T{other.te_count}F{other.flex_count}"
            )
        return (
            f"Team {team_id} could not make a valid pick "
            f"(pick_index={self.current_pick_index}, "
            f"roster_size={roster.size}/{self.state.total_roster_size_per_team}, "
            f"counts=QB:{roster.qb_count} RB:{roster.rb_count} "
            f"WR:{roster.wr_count} TE:{roster.te_count} FLEX:{roster.flex_count}, "
            f"available={dict(available_positions)}, can_draft={can_draft}, "
            f"teams=[{', '.join(team_summaries)}])."
        )

    def _select_simulated_pick(
        self,
        team_id: int,
        build_state_fn=None,
        get_action_mask_fn=None,
        policy_bot: BotGM | None = None,
    ) -> Optional[Player]:
        """Return a simulated pick for one team via policy or configured bot.

        Parameters
        ----------
        team_id : int
            Drafting team id.
        build_state_fn : callable, optional
            Builds policy state for a team.
        get_action_mask_fn : callable, optional
            Builds a valid-action mask for a team.
        policy_bot : BotGM, optional
            Explicit policy bot that bypasses per-team strategies.

        Returns
        -------
        Player or None
            Selected player when a legal pick exists.
        """
        if policy_bot is not None:
            if build_state_fn is None or get_action_mask_fn is None:
                raise ValueError("Policy simulation requires state builder callbacks.")
            return policy_bot.execute_pick(
                team_id=team_id,
                available_player_ids=self.available_player_ids,
                player_catalog=self.player_catalog,
                team_roster=self.state.roster_for_team(team_id),
                roster_structure=self.state.roster_structure,
                bench_maxes=self.state.bench_maxes,
                can_draft_position_fn=self.can_draft_position,
                try_select_player_fn=self.try_select_player_for_team,
                build_state_fn=build_state_fn,
                get_action_mask_fn=get_action_mask_fn,
            )
        return self._select_bot_pick(
            team_id=team_id,
            build_state_fn=build_state_fn,
            get_action_mask_fn=get_action_mask_fn,
        )

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

    def _undo_latest_pick_action(self, action: DraftAction) -> None:
        """Undo the latest action when it points to the latest pick."""
        if action.history_index != len(self.state.draft_history) - 1:
            raise ValueError("Cannot undo pick out of chronological order.")
        last_pick = self.state.pop_pick()
        if last_pick is None:
            raise ValueError("No pick history to undo.")
        player = self.player_catalog.require(last_pick.player_id)
        cell = self.state.find_player_cell(last_pick.player_id)
        if cell is not None:
            self.state.clear_cell(cell[0], cell[1])
        self.state.remove_player_from_roster(last_pick.team_id, player)
        self.state.recalculate_roster_counts(last_pick.team_id, self.player_catalog.require)
        self.state.current_pick_index = last_pick.previous_pick_index
        self.state.current_pick_number = last_pick.pick_number
        self.state.override_team_id = last_pick.previous_override_team_id
        self.state.pop_action()

    def _undo_latest_transfer_action(self, action: DraftAction) -> None:
        """Undo the latest action when it points to the latest transfer."""
        if action.history_index != len(self.state.transfer_history) - 1:
            raise ValueError("Cannot undo transfer out of chronological order.")
        if not self.state.transfer_history:
            raise ValueError("No transfer history to undo.")
        transfer = self.state.transfer_history[-1]
        player = self.player_catalog.require(transfer.player_id)
        current_team_id = self.state.find_player_team_id(transfer.player_id)
        if current_team_id != transfer.to_team_id:
            raise ValueError("Cannot undo transfer because player ownership changed.")
        if transfer.from_team_id != transfer.to_team_id:
            self.state.move_player_between_rosters(
                transfer.to_team_id, transfer.from_team_id, player
            )
            self.state.recalculate_roster_counts(transfer.to_team_id, self.player_catalog.require)
            self.state.recalculate_roster_counts(transfer.from_team_id, self.player_catalog.require)
        self.state.clear_cell(transfer.to_team_id, transfer.to_round)
        self.state.place_player_visual(
            transfer.from_team_id, transfer.from_round, transfer.player_id
        )
        self.state.override_team_id = transfer.previous_override_team_id
        self.state.pop_transfer()
        self.state.pop_action()

    def _undo_latest_swap_action(self, action: DraftAction) -> None:
        """Undo the latest action when it points to the latest swap."""
        if action.history_index != len(self.state.swap_history) - 1:
            raise ValueError("Cannot undo swap out of chronological order.")
        if not self.state.swap_history:
            raise ValueError("No swap history to undo.")
        swap = self.state.swap_history[-1]
        player_1 = self.player_catalog.require(swap.player_id_1)
        player_2 = self.player_catalog.require(swap.player_id_2)
        if swap.team_id_1 != swap.team_id_2:
            self.state.remove_player_from_roster(
                swap.team_id_2, player_1, restore_availability=False
            )
            self.state.remove_player_from_roster(
                swap.team_id_1, player_2, restore_availability=False
            )
            self.state.roster_for_team(swap.team_id_1).player_ids.append(swap.player_id_1)
            self.state.roster_for_team(swap.team_id_2).player_ids.append(swap.player_id_2)
            self.state.recalculate_roster_counts(swap.team_id_1, self.player_catalog.require)
            self.state.recalculate_roster_counts(swap.team_id_2, self.player_catalog.require)
        self.state.place_player_visual(swap.team_id_1, swap.round_1, swap.player_id_1)
        self.state.place_player_visual(swap.team_id_2, swap.round_2, swap.player_id_2)
        self.state.override_team_id = swap.previous_override_team_id
        self.state.pop_swap()
        self.state.pop_action()

    def _is_valid_team_id(self, team_id: int) -> bool:
        """Return whether a team id belongs to the draft."""
        valid_team_ids = set(self.state.draft_order) | set(self.state.team_rosters.keys())
        return team_id in valid_team_ids

    def _is_valid_round(self, team_id: int, round_index: int) -> bool:
        """Return whether a round index exists on a team's visual board."""
        if round_index < 0:
            return False
        return round_index in self.state.visual_board.get(team_id, {})
