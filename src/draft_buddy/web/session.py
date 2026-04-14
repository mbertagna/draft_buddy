"""Session orchestration for the FastAPI presentation layer."""

from __future__ import annotations

import json
import math
import os
import random
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from draft_buddy.config import Config
from draft_buddy.core import BotGM, DraftState, FantasyRulesEngine, InferenceProvider, create_bot_gm
from draft_buddy.domain.entities import Player
from draft_buddy.data.player_loader import load_player_data
from draft_buddy.logic.draft_presentation_service import DraftPresentationService


class DraftSession:
    """Session-scoped draft coordinator built on core state.

    Parameters
    ----------
    config : Config
        Runtime application configuration.
    """

    def __init__(
        self, config: Config, inference_provider: Optional[InferenceProvider] = None
    ) -> None:
        self._config = config
        self._inference_provider = inference_provider
        self.manual_draft_teams = set(config.draft.MANUAL_DRAFT_TEAMS)
        self.all_players_data = load_player_data(
            config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG
        )
        self.player_map = {player.player_id: player for player in self.all_players_data}
        self.total_roster_size_per_team = sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE
        self._state = DraftState(
            all_player_ids={player.player_id for player in self.all_players_data},
            draft_order=self._generate_snake_draft_order(),
            roster_structure=config.draft.ROSTER_STRUCTURE,
            bench_maxes=config.draft.BENCH_MAXES,
            total_roster_size_per_team=self.total_roster_size_per_team,
            agent_team_id=config.draft.AGENT_START_POSITION,
        )
        self._rules = FantasyRulesEngine(
            roster_structure=config.draft.ROSTER_STRUCTURE,
            bench_maxes=config.draft.BENCH_MAXES,
            total_roster_size_per_team=self.total_roster_size_per_team,
        )
        self._action_to_position = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}
        self._bots = self._build_bot_map()
        self.weekly_projections = {
            player.player_id: {"pts": [player.projected_points] * 18, "pos": player.position}
            for player in self.all_players_data
        }

    @property
    def draft_order(self) -> List[int]:
        """Return draft order."""
        return self._state.get_draft_order()

    @property
    def current_pick_idx(self) -> int:
        """Return current pick index."""
        return self._state.get_current_pick_idx()

    @property
    def current_pick_number(self) -> int:
        """Return current pick number."""
        return self._state.get_current_pick_number()

    @property
    def teams_rosters(self) -> Dict[int, Dict[str, Any]]:
        """Return rosters."""
        return self._state.get_rosters()

    @property
    def team_manager_mapping(self) -> Dict[int, str]:
        """Return team manager mapping."""
        return self._config.draft.TEAM_MANAGER_MAPPING

    @property
    def roster_structure(self) -> Dict[str, int]:
        """Return roster structure."""
        return self._config.draft.ROSTER_STRUCTURE

    @property
    def bench_maxes(self) -> Dict[str, int]:
        """Return bench maximums."""
        return self._config.draft.BENCH_MAXES

    @property
    def num_teams(self) -> int:
        """Return number of teams."""
        return self._config.draft.NUM_TEAMS

    @property
    def agent_team_id(self) -> int:
        """Return perspective team id."""
        return self._state.get_agent_team_id()

    @property
    def available_players_ids(self) -> set[int]:
        """Return currently available player IDs."""
        return self._state.get_available_player_ids()

    @property
    def _overridden_team_id(self) -> Optional[int]:
        """Compatibility access for presentation helpers."""
        return self._state.get_overridden_team_id()

    @property
    def _draft_history(self) -> List[Dict[str, Any]]:
        """Compatibility access for export/history."""
        return self._state.get_draft_history()

    def get_positional_baselines(self) -> Dict[str, float]:
        """Return per-position replacement baselines for VORP."""
        baselines: Dict[str, float] = {}
        for position in ["QB", "RB", "WR", "TE"]:
            available = sorted(
                [
                    self.player_map[player_id]
                    for player_id in self.available_players_ids
                    if self.player_map[player_id].position == position
                ],
                key=lambda player: player.projected_points,
                reverse=True,
            )
            if not available:
                baselines[position] = 0.0
                continue
            needed_starters = 0
            required = self.roster_structure.get(position, 0)
            for team_id in range(1, self.num_teams + 1):
                roster_counts = self.teams_rosters[team_id]
                missing = max(0, required - roster_counts.get(position, 0))
                needed_starters += missing
            replacement_idx = min(max(0, needed_starters + 1), len(available) - 1)
            before = available[max(0, replacement_idx - 1)].projected_points
            current = available[replacement_idx].projected_points
            after = available[min(len(available) - 1, replacement_idx + 1)].projected_points
            baselines[position] = (before + current + after) / 3.0
        return baselines

    def get_ui_state(self) -> Dict[str, Any]:
        """Build UI state for API responses."""
        return DraftPresentationService.get_ui_state(self)

    def save_state(self, file_path: str) -> None:
        """Persist session state to JSON file."""
        serializable_rosters = defaultdict(
            lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "PLAYERS": []}
        )
        for team_id, roster_data in self.teams_rosters.items():
            serializable_rosters[team_id] = roster_data.copy()
            serializable_rosters[team_id]["PLAYERS"] = [player.to_dict() for player in roster_data["PLAYERS"]]
        payload = {
            "available_players_ids": list(self.available_players_ids),
            "teams_rosters": serializable_rosters,
            "draft_order": self.draft_order,
            "current_pick_idx": self.current_pick_idx,
            "current_pick_number": self.current_pick_number,
            "_draft_history": self._draft_history,
            "_overridden_team_id": self._overridden_team_id,
        }
        temp_file_path = f"{file_path}.tmp"
        with open(temp_file_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
        os.replace(temp_file_path, file_path)

    def load_state(self, file_path: str) -> None:
        """Load session state if file exists."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        self._state.load_from_serialized(payload, lambda player: Player(**player))

    def reset(self) -> None:
        """Reset session to a fresh draft."""
        self._state.reset(
            all_player_ids={player.player_id for player in self.all_players_data},
            draft_order=self._generate_snake_draft_order(),
            agent_team_id=self._config.draft.AGENT_START_POSITION,
        )

    def draft_player(self, player_id: int) -> None:
        """Draft a specific player for the team on the clock."""
        if self.current_pick_idx >= len(self.draft_order) and self._overridden_team_id is None:
            raise ValueError("The draft has already concluded. No more picks can be made.")
        current_team_id = (
            self._overridden_team_id
            if self._overridden_team_id is not None
            else self.draft_order[self.current_pick_idx]
        )
        drafted_player = self.player_map.get(player_id)
        if drafted_player is None or drafted_player.player_id not in self.available_players_ids:
            self._state.set_overridden_team_id(None)
            raise ValueError(f"Player with ID {player_id} is not available to be drafted.")
        if not self._rules.can_draft_manual(self._state, current_team_id, drafted_player.position, self.player_map):
            self._state.set_overridden_team_id(None)
            raise ValueError(f"Team {current_team_id} cannot draft a {drafted_player.position}.")

        self._state.append_draft_history(
            {
                "player_id": drafted_player.player_id,
                "team_id": current_team_id,
                "is_manual_pick": True,
                "previous_pick_idx": self.current_pick_idx,
                "previous_pick_number": self.current_pick_number,
                "previous_overridden_team_id": self._overridden_team_id,
                "was_override": self._overridden_team_id is not None,
            }
        )
        self._state.add_player_to_roster(current_team_id, drafted_player)
        self._state.advance_pick()
        self._state.set_overridden_team_id(None)

    def undo_last_pick(self) -> None:
        """Undo latest pick."""
        last_pick = self._state.pop_draft_history()
        if last_pick is None:
            raise ValueError("No picks to undo.")
        player = self.player_map[last_pick["player_id"]]
        self._state.remove_player_from_roster(last_pick["team_id"], player)
        self._state.set_current_pick_idx(last_pick["previous_pick_idx"])
        self._state.set_current_pick_number(last_pick["previous_pick_number"])
        self._state.set_overridden_team_id(last_pick.get("previous_overridden_team_id"))

    def set_current_team_picking(self, team_id: int) -> None:
        """Override next pick team."""
        if team_id not in range(1, self.num_teams + 1):
            raise ValueError(f"Invalid team ID: {team_id}. Must be between 1 and {self.num_teams}.")
        self._state.set_overridden_team_id(team_id)

    def simulate_single_pick(self) -> None:
        """Simulate one opponent pick."""
        if self.current_pick_idx >= len(self.draft_order):
            raise ValueError("The draft has already concluded. No more picks can be made.")
        team_id = self._overridden_team_id or self.draft_order[self.current_pick_idx]
        self._state.set_overridden_team_id(None)
        if team_id in self.manual_draft_teams:
            raise ValueError("It is a manual team's turn. Cannot simulate pick.")
        player = self._simulate_bot_pick(team_id)
        if player is None:
            raise ValueError(f"Team {team_id} could not make a valid pick.")
        self._state.append_draft_history(
            {
                "player_id": player.player_id,
                "team_id": team_id,
                "is_manual_pick": False,
                "previous_pick_idx": self.current_pick_idx,
                "previous_pick_number": self.current_pick_number,
                "previous_overridden_team_id": None,
                "was_override": False,
            }
        )
        self._state.add_player_to_roster(team_id, player)
        self._state.advance_pick()

    def simulate_scheduled_picks_remaining(self) -> None:
        """Simulate all remaining scheduled picks."""
        while self.current_pick_idx < len(self.draft_order):
            self.simulate_single_pick()

    def _simulate_bot_pick(self, team_id: int) -> Optional[Player]:
        """Select a bot pick with fallback behavior."""
        strategy = self._bots.get(team_id)
        if strategy is None:
            strategy = self._create_bot_strategy(
                team_id=team_id,
                strategy_config=self._config.opponent.DEFAULT_OPPONENT_STRATEGY,
            )
            self._bots[team_id] = strategy

        chosen = strategy.execute_pick(
            team_id=team_id,
            available_player_ids=self.available_players_ids,
            player_map=self.player_map,
            roster_counts=self.teams_rosters[team_id],
            roster_structure=self.roster_structure,
            bench_maxes=self.bench_maxes,
            can_draft_position_fn=self._can_draft_position,
            try_select_player_fn=self._try_select_player_for_team,
            build_state_fn=self._build_state_for_team,
            get_action_mask_fn=self._get_action_mask_for_team,
        )
        if chosen is not None:
            return chosen
        eligible = [
            self.player_map[player_id]
            for player_id in self.available_players_ids
            if self.player_map[player_id].position in {"QB", "RB", "WR", "TE"}
            and self._can_draft_position(team_id, self.player_map[player_id].position)
        ]
        return random.choice(eligible) if eligible else None

    def _can_draft_position(self, team_id: int, position: str, is_manual: bool = False) -> bool:
        """Evaluate whether team can draft a position."""
        if is_manual:
            return self._rules.can_draft_manual(self._state, team_id, position, self.player_map)
        return self._rules.can_draft_simulated(self._state, team_id, position, self.player_map)

    def _get_action_mask_for_team(self, team_id: int) -> np.ndarray:
        """Return valid-action mask for a team.

        Parameters
        ----------
        team_id : int
            Team perspective for action validation.

        Returns
        -------
        np.ndarray
            Boolean mask over ``QB``, ``RB``, ``WR``, ``TE`` actions.
        """
        mask = np.zeros(len(self._action_to_position), dtype=bool)
        for action, position in self._action_to_position.items():
            if self._can_draft_position(team_id, position, is_manual=False):
                mask[action] = True
        return mask

    def _build_state_for_team(self, team_id: int) -> np.ndarray:
        """Build normalized inference state for a team.

        Parameters
        ----------
        team_id : int
            Team perspective.

        Returns
        -------
        np.ndarray
            Inference feature vector.

        Raises
        ------
        ValueError
            If no inference provider is configured.
        """
        if self._inference_provider is None:
            raise ValueError("Inference provider is not configured.")
        return self._inference_provider.build_state_vector(
            team_id=team_id, draft_state=self._state, player_map=self.player_map
        )

    def _try_select_player_for_team(self, team_id: int, position_choice: str, available_ids: set) -> tuple[bool, Optional[Player]]:
        """Pick best available player by position for a team."""
        available_players_for_pos = [
            self.player_map[player_id]
            for player_id in available_ids
            if self.player_map[player_id].position == position_choice
        ]
        if not available_players_for_pos:
            return False, None
        if not self._can_draft_position(team_id, position_choice, is_manual=False):
            return False, None
        return True, max(available_players_for_pos, key=lambda player: player.projected_points)

    def _build_bot_map(self) -> Dict[int, BotGM]:
        """Build team-id to bot strategy map."""
        bots: Dict[int, BotGM] = {}
        for team_id in range(1, self.num_teams + 1):
            if team_id == self._config.draft.AGENT_START_POSITION:
                continue
            strategy_config = self._config.opponent.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self._config.opponent.DEFAULT_OPPONENT_STRATEGY
            )
            bots[team_id] = self._create_bot_strategy(team_id, strategy_config)
        return bots

    def _create_bot_strategy(self, team_id: int, strategy_config: Dict[str, Any]) -> BotGM:
        """Create a BotGM strategy from configuration.

        Parameters
        ----------
        team_id : int
            Team receiving strategy.
        strategy_config : Dict[str, Any]
            Strategy configuration for team.

        Returns
        -------
        BotGM
            Concrete strategy implementation.
        """
        if strategy_config.get("logic") == "AGENT_MODEL" and self._inference_provider is not None:
            model_bot = self._inference_provider.create_bot(
                team_id=team_id,
                strategy_config=strategy_config,
                action_to_position=self._action_to_position,
            )
            if model_bot is not None:
                return model_bot
        return create_bot_gm(strategy_config["logic"], strategy_config)

    def get_ai_suggestion(self) -> Dict[str, Any]:
        """Return AI suggestion for the team currently on the clock.

        Returns
        -------
        Dict[str, Any]
            Position probabilities or an ``error`` payload.
        """
        if self.current_pick_idx >= len(self.draft_order):
            return {"error": "Draft is over."}
        current_team_on_clock = self.draft_order[self.current_pick_idx]
        return self.get_ai_suggestion_for_team(current_team_on_clock)

    def get_ai_suggestion_for_team(
        self, team_id: int, ignore_player_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Return position probabilities for a team perspective.

        Parameters
        ----------
        team_id : int
            Team perspective for prediction.
        ignore_player_ids : Optional[List[int]], optional
            Player ids to temporarily treat as unavailable.

        Returns
        -------
        Dict[str, Any]
            Position probabilities or an ``error`` payload.
        """
        if not (1 <= team_id <= self.num_teams):
            return {"error": f"Invalid team id {team_id}."}
        if self._inference_provider is None:
            return {"error": "AI model not loaded."}

        original_available_ids: Optional[set[int]] = None
        try:
            if ignore_player_ids:
                ignore_set = {
                    player_id
                    for player_id in ignore_player_ids
                    if player_id in self._state.get_available_player_ids()
                }
                if ignore_set:
                    original_available_ids = set(self._state.get_available_player_ids())
                    self._state.replace_available_player_ids(original_available_ids - ignore_set)
            return self._inference_provider.predict_action_probabilities(
                team_id=team_id,
                draft_state=self._state,
                player_map=self.player_map,
                action_to_position=self._action_to_position,
                get_action_mask_fn=self._get_action_mask_for_team,
            )
        except Exception as error:
            return {"error": str(error)}
        finally:
            if original_available_ids is not None:
                self._state.replace_available_player_ids(original_available_ids)

    def get_ai_suggestions_all(self) -> Dict[str, Any]:
        """Return AI position probabilities for all teams.

        Returns
        -------
        Dict[str, Any]
            Team-indexed probability maps or an ``error`` payload.
        """
        if self._inference_provider is None:
            return {"error": "AI model not loaded."}
        suggestions: Dict[int, Dict[str, Any]] = {}
        for team_id in range(1, self.num_teams + 1):
            suggestions[team_id] = self.get_ai_suggestion_for_team(team_id)
        return suggestions

    def _generate_snake_draft_order(self) -> List[int]:
        """Generate snake order for all rounds."""
        draft_order: List[int] = []
        rounds = math.ceil(self.total_roster_size_per_team)
        if rounds * self.num_teams > len(self.all_players_data):
            rounds = max(1, math.floor(len(self.all_players_data) / self.num_teams))
        for round_number in range(rounds):
            team_ids = range(1, self.num_teams + 1)
            if round_number % 2 == 1:
                team_ids = reversed(list(team_ids))
            draft_order.extend(team_ids)
        return draft_order


class DraftSessionManager:
    """Thread-safe mapping from session IDs to draft sessions."""

    def __init__(
        self, config: Config, inference_provider: Optional[InferenceProvider] = None
    ) -> None:
        self._config = config
        self._inference_provider = inference_provider
        self._sessions: Dict[str, DraftSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> DraftSession:
        """Return existing session or create a new one."""
        with self._lock:
            if session_id not in self._sessions:
                session = DraftSession(self._config, inference_provider=self._inference_provider)
                session.load_state(self._config.paths.DRAFT_STATE_FILE)
                if not session.draft_order:
                    session.reset()
                    session.save_state(self._config.paths.DRAFT_STATE_FILE)
                self._sessions[session_id] = session
            return self._sessions[session_id]

    def create_new(self, session_id: str) -> DraftSession:
        """Create and persist a fresh draft session."""
        with self._lock:
            session = DraftSession(self._config, inference_provider=self._inference_provider)
            session.reset()
            session.save_state(self._config.paths.DRAFT_STATE_FILE)
            self._sessions[session_id] = session
            return session
