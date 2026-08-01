"""Session orchestration for the FastAPI presentation layer."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from draft_buddy.config import Config
from draft_buddy.core import (
    BotGM,
    DraftController,
    DraftState,
    FantasyRulesEngine,
    InferenceProvider,
    create_bot_gm,
)
from draft_buddy.core.draft_state_store import archive_draft_state
from draft_buddy.data import load_player_catalog


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
        self.player_catalog = load_player_catalog(
            config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG
        )
        self.total_roster_size_per_team = (
            sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE
        )
        self._action_to_position = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}
        self._state = DraftState(
            all_player_ids=set(self.player_catalog.player_ids),
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
        self._controller = DraftController(
            state=self._state,
            player_catalog=self.player_catalog,
            rules_engine=self._rules,
            action_to_position=self._action_to_position,
            bot_factory=self._create_bot_strategy,
        )
        self.weekly_projections = self.player_catalog.to_weekly_projections()
        self._pending_state_load_warning: Optional[str] = None

    @property
    def draft_order(self) -> List[int]:
        """Return draft order."""
        return self._state.draft_order

    @property
    def current_pick_index(self) -> int:
        """Return current pick index."""
        return self._state.current_pick_index

    @property
    def current_pick_number(self) -> int:
        """Return current pick number."""
        return self._state.current_pick_number

    @property
    def team_rosters(self):
        """Return typed team rosters."""
        return self._state.team_rosters

    @property
    def team_manager_mapping(self) -> Dict[int, str]:
        """Return cosmetic team display names keyed by team id."""
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
        return self._state.agent_team_id

    @property
    def total_bench_size(self) -> int:
        """Return configured bench size per team."""
        return self._config.draft.TOTAL_BENCH_SIZE

    @property
    def scoring_rules(self) -> Dict[str, Optional[float]]:
        """Return the active fantasy scoring rules for this league."""
        return self._config.get_scoring_rules()

    @property
    def snake_team_on_turn(self) -> Optional[int]:
        """Return the team scheduled to pick in snake order."""
        if self.current_pick_index >= len(self.draft_order):
            return None
        return self.draft_order[self.current_pick_index]

    @property
    def override_active(self) -> bool:
        """Return whether the clock differs from the natural snake pick."""
        override_id = self._state.override_team_id
        if override_id is None:
            return False
        if self.current_pick_index >= len(self.draft_order):
            return False
        return override_id != self.draft_order[self.current_pick_index]

    @property
    def available_player_ids(self) -> set[int]:
        """Return currently available player ids."""
        return self._state.available_player_ids

    @property
    def draft_history(self):
        """Return typed draft history."""
        return self._state.draft_history

    @property
    def transfer_history(self):
        """Return typed transfer history."""
        return self._state.transfer_history

    def get_positional_baselines(self) -> Dict[str, float]:
        """Return per-position replacement baselines for VORP."""
        return self._controller.get_positional_baselines()

    def get_ui_state(self) -> Dict[str, Any]:
        """Build UI state for API responses."""
        structured_rosters: Dict[int, Dict[str, Any]] = {}
        team_points_summary: Dict[int, Dict[str, float]] = {}
        team_is_full: Dict[int, bool] = {}

        from draft_buddy.core.roster_utils import calculate_roster_scores, categorize_roster_by_slots

        for team_id, team_roster in self.team_rosters.items():
            roster_players = self._controller.resolve_roster_players(team_id)
            starters, bench, _ = categorize_roster_by_slots(
                roster_players, self.roster_structure, self.bench_maxes
            )
            scores = calculate_roster_scores(
                roster_players, self.roster_structure, self.bench_maxes
            )
            team_points_summary[team_id] = {
                "starters_total": scores["starters_total_points"],
                "bench_total": scores["bench_total_points"],
            }
            structured_rosters[team_id] = {
                "starters": {
                    position: [player.to_dict() for player in players]
                    for position, players in starters.items()
                },
                "bench": [player.to_dict() for player in bench],
                "players_flat": [player.to_dict() for player in roster_players],
            }
            team_is_full[team_id] = team_roster.size >= self.total_roster_size_per_team

        visual_board = {
            team_id: {
                round_index: player_id
                for round_index, player_id in rounds.items()
            }
            for team_id, rounds in self._state.visual_board.items()
        }

        payload = {
            "draft_order": self.draft_order,
            "current_pick_index": self.current_pick_index,
            "current_pick_number": self.current_pick_number,
            "current_team_picking": self._controller.team_on_clock,
            "snake_team_on_turn": self.snake_team_on_turn,
            "override_active": self.override_active,
            "team_rosters": structured_rosters,
            "visual_board": visual_board,
            "roster_counts": {
                team_id: {
                    "QB": team_roster.qb_count,
                    "RB": team_roster.rb_count,
                    "WR": team_roster.wr_count,
                    "TE": team_roster.te_count,
                    "FLEX": team_roster.flex_count,
                }
                for team_id, team_roster in self.team_rosters.items()
            },
            "team_projected_points": {
                team_id: sum(
                    player.projected_points
                    for player in self._controller.resolve_roster_players(team_id)
                )
                for team_id in self.team_rosters
            },
            "manual_draft_teams": list(self.manual_draft_teams),
            "roster_structure": self.roster_structure,
            "team_is_full": team_is_full,
            "team_points_summary": team_points_summary,
            "num_teams": self.num_teams,
            "total_roster_size_per_team": self.total_roster_size_per_team,
            "team_bye_weeks": self._aggregate_bye_weeks(),
            "agent_start_position": self.agent_team_id,
            "team_display_names": {
                team_id: self.team_manager_mapping.get(team_id) or f"Team {team_id}"
                for team_id in range(1, self.num_teams + 1)
            },
            "pick_by_player_id": {
                pick.player_id: pick.pick_number for pick in self.draft_history
            },
        }
        warning = self.consume_state_load_warning()
        if warning:
            payload["state_load_warning"] = warning
        return payload

    def consume_state_load_warning(self) -> Optional[str]:
        """Return and clear a one-shot draft-state load warning.

        Returns
        -------
        str or None
            Warning text when the last load used recovery or failed over.
        """
        warning = self._pending_state_load_warning
        self._pending_state_load_warning = None
        if warning is not None:
            return warning
        return self._controller.consume_load_warning()

    def set_state_load_warning(self, warning: str) -> None:
        """Store a one-shot warning for the next UI state response.

        Parameters
        ----------
        warning : str
            Human-readable load or recovery message.
        """
        self._pending_state_load_warning = warning

    def save_state(self, file_path: Optional[str] = None) -> None:
        """Persist session state to the configured durable draft files.

        Parameters
        ----------
        file_path : str, optional
            Primary path override. Defaults to configured ``DRAFT_STATE_FILE``.
        """
        primary = file_path or self._config.paths.DRAFT_STATE_FILE
        self._controller.save_state(
            primary,
            prev_path=self._config.paths.DRAFT_STATE_PREV_FILE,
        )

    def load_state(self, file_path: Optional[str] = None) -> None:
        """Load session state with durable recovery paths.

        Parameters
        ----------
        file_path : str, optional
            Primary path override. Defaults to configured ``DRAFT_STATE_FILE``.
        """
        primary = file_path or self._config.paths.DRAFT_STATE_FILE
        self._controller.load_state(
            primary,
            prev_path=self._config.paths.DRAFT_STATE_PREV_FILE,
            saved_states_dir=self._config.paths.SAVED_STATES_DIR,
        )

    def reset(self) -> None:
        """Reset session to a fresh draft."""
        self._controller.reset(
            draft_order=self._generate_snake_draft_order(),
            agent_team_id=self._config.draft.AGENT_START_POSITION,
        )

    def draft_player(self, player_id: int) -> None:
        """Draft a specific player for the team on the clock."""
        self._controller.draft_player(player_id, is_manual_pick=True)

    def undo_last_pick(self) -> None:
        """Undo latest draft action."""
        self._controller.undo_last_pick()

    def transfer_player(
        self, player_id: int, to_team_id: int, to_round: Optional[int] = None
    ) -> None:
        """Move a drafted player to another team or visual slot."""
        self._controller.transfer_player(player_id, to_team_id, to_round=to_round)

    def swap_players(self, player_id_1: int, player_id_2: int) -> None:
        """Swap two drafted players' teams and visual slots."""
        self._controller.swap_players(player_id_1, player_id_2)

    def set_current_team_picking(self, team_id: int) -> None:
        """Override next pick team."""
        if team_id not in range(1, self.num_teams + 1):
            raise ValueError(f"Invalid team ID: {team_id}. Must be between 1 and {self.num_teams}.")
        self._controller.set_override_team(team_id)

    def _resolve_policy_simulation(self) -> tuple[Optional[BotGM], Optional[Callable], Optional[Callable]]:
        """Build policy bot and callbacks for simulated picks.

        Returns
        -------
        tuple[Optional[BotGM], Optional[Callable], Optional[Callable]]
            Policy bot, state builder, and action-mask callbacks.
        """
        if self._inference_provider is None:
            return None, None, None
        policy_bot = self._inference_provider.create_policy_sim_bot(self._action_to_position)
        if policy_bot is None:
            return None, None, None
        build_state_fn = lambda team_id: self._inference_provider.build_state_vector(
            team_id,
            self._state,
            self.player_catalog,
        )
        return policy_bot, build_state_fn, self._controller.get_action_mask_for_team

    def simulate_single_pick(self, use_policy: bool = False) -> None:
        """Simulate one opponent pick.

        Parameters
        ----------
        use_policy : bool, optional
            When ``True``, use the loaded policy network instead of configured bots.
        """
        policy_bot = None
        build_state_fn = None
        get_action_mask_fn = None
        if use_policy:
            policy_bot, build_state_fn, get_action_mask_fn = self._resolve_policy_simulation()
            if policy_bot is None:
                raise ValueError(
                    "Policy model not loaded. Switch sim mode to Bot or configure MODEL_PATH_TO_LOAD."
                )
        self._controller.simulate_single_pick(
            manual_draft_teams=self.manual_draft_teams,
            build_state_fn=build_state_fn,
            get_action_mask_fn=get_action_mask_fn,
            policy_bot=policy_bot,
        )

    def simulate_scheduled_picks_remaining(self, use_policy: bool = False) -> None:
        """Simulate all remaining scheduled picks.

        Parameters
        ----------
        use_policy : bool, optional
            When ``True``, use the loaded policy network instead of configured bots.
        """
        policy_bot = None
        build_state_fn = None
        get_action_mask_fn = None
        if use_policy:
            policy_bot, build_state_fn, get_action_mask_fn = self._resolve_policy_simulation()
            if policy_bot is None:
                raise ValueError(
                    "Policy model not loaded. Switch sim mode to Bot or configure MODEL_PATH_TO_LOAD."
                )
        self._controller.simulate_remaining(
            manual_draft_teams=self.manual_draft_teams,
            build_state_fn=build_state_fn,
            get_action_mask_fn=get_action_mask_fn,
            policy_bot=policy_bot,
        )

    def get_ai_suggestion(self) -> Dict[str, Any]:
        """Return AI suggestion for the team currently on the clock."""
        current_team_on_clock = self._controller.team_on_clock
        if current_team_on_clock is None:
            return {"error": "Draft is over."}
        return self.get_ai_suggestion_for_team(current_team_on_clock)

    def get_ai_suggestion_for_team(
        self, team_id: int, ignore_player_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Return position probabilities for a team perspective."""
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
                    if player_id in self._state.available_player_ids
                }
                if ignore_set:
                    original_available_ids = set(self._state.available_player_ids)
                    self._state.available_player_ids = original_available_ids - ignore_set
            return self._inference_provider.predict_action_probabilities(
                team_id=team_id,
                draft_state=self._state,
                player_catalog=self.player_catalog,
                action_to_position=self._action_to_position,
                get_action_mask_fn=self._controller.get_action_mask_for_team,
            )
        except Exception as error:
            return {"error": str(error)}
        finally:
            if original_available_ids is not None:
                self._state.available_player_ids = original_available_ids

    def get_ai_suggestions_all(self) -> Dict[str, Any]:
        """Return AI position probabilities for all teams."""
        if self._inference_provider is None:
            return {"error": "AI model not loaded."}
        return {
            team_id: self.get_ai_suggestion_for_team(team_id)
            for team_id in range(1, self.num_teams + 1)
        }

    def _create_bot_strategy(self, team_id: int) -> BotGM:
        """Create a BotGM strategy from configuration."""
        strategy_config = self._config.opponent.OPPONENT_TEAM_STRATEGIES.get(
            team_id, self._config.opponent.DEFAULT_OPPONENT_STRATEGY
        )
        if strategy_config.get("logic") == "AGENT_MODEL" and self._inference_provider is not None:
            model_bot = self._inference_provider.create_bot(
                team_id=team_id,
                strategy_config=strategy_config,
                action_to_position=self._action_to_position,
            )
            if model_bot is not None:
                return model_bot
        return create_bot_gm(strategy_config["logic"], strategy_config)

    def _generate_snake_draft_order(self) -> List[int]:
        """Generate snake order for all rounds."""
        draft_order: List[int] = []
        rounds = math.ceil(self.total_roster_size_per_team)
        if rounds * self.num_teams > len(self.player_catalog):
            rounds = max(1, math.floor(len(self.player_catalog) / self.num_teams))
        for round_number in range(rounds):
            team_ids = range(1, self.num_teams + 1)
            if round_number % 2 == 1:
                team_ids = reversed(list(team_ids))
            draft_order.extend(team_ids)
        return draft_order

    def _aggregate_bye_weeks(self) -> Dict[int, Dict[int, Dict[str, int]]]:
        """Aggregate bye week counts by team and position."""
        bye_data: Dict[int, Dict[int, Dict[str, int]]] = {}
        for team_id in self.team_rosters:
            roster_players = self._controller.resolve_roster_players(team_id)
            team_byes: Dict[int, Dict[str, int]] = {}
            weeks = {
                player.bye_week
                for player in roster_players
                if player.bye_week and not np.isnan(player.bye_week)
            }
            for week in weeks:
                pos_counts: Dict[str, int] = {}
                for player in roster_players:
                    if player.bye_week == week:
                        pos_counts[player.position] = pos_counts.get(player.position, 0) + 1
                team_byes[int(week)] = pos_counts
            bye_data[team_id] = team_byes
        return bye_data


class DraftSessionManager:
    """Thread-safe shared draft session for the active draft state file."""

    def __init__(
        self, config: Config, inference_provider: Optional[InferenceProvider] = None
    ) -> None:
        self._config = config
        self._inference_provider = inference_provider
        self._shared_session: Optional[DraftSession] = None
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> DraftSession:
        """Return the shared draft session, creating it on first use.

        Parameters
        ----------
        session_id : str
            Cookie session id retained for API compatibility.

        Returns
        -------
        DraftSession
            Shared in-memory draft session.
        """
        _ = session_id
        with self._lock:
            return self._get_or_create_unlocked()

    def create_new(self, session_id: str) -> DraftSession:
        """Archive the current draft, then create and persist a fresh session.

        Parameters
        ----------
        session_id : str
            Cookie session id retained for API compatibility.

        Returns
        -------
        DraftSession
            Shared session reset to a new draft.
        """
        _ = session_id
        with self._lock:
            archive_draft_state(
                self._config.paths.DRAFT_STATE_FILE,
                self._config.paths.SAVED_STATES_DIR,
            )
            session = DraftSession(self._config, inference_provider=self._inference_provider)
            session.reset()
            session.save_state()
            self._shared_session = session
            return session

    def run_locked(self, session_id: str, mutation: Callable[[DraftSession], None]) -> DraftSession:
        """Run a mutation against the shared session under the manager lock.

        Parameters
        ----------
        session_id : str
            Cookie session id retained for API compatibility.
        mutation : callable
            Callback that receives the shared session and mutates it.

        Returns
        -------
        DraftSession
            Shared session after mutation and durable save.
        """
        with self._lock:
            session = self._get_or_create_unlocked()
            mutation(session)
            session.save_state()
            return session

    def _get_or_create_unlocked(self) -> DraftSession:
        """Return or build the shared session. Caller must hold ``_lock``."""
        if self._shared_session is not None:
            return self._shared_session

        session = DraftSession(self._config, inference_provider=self._inference_provider)
        try:
            session.load_state()
        except ValueError as error:
            warning = session._controller.consume_load_warning() or str(error)
            print(f"WARNING: {warning}")
            session.set_state_load_warning(warning)
            session.reset()
            session.save_state()
        else:
            if session._controller.load_warning:
                session.save_state()
        if not session.draft_order:
            session.reset()
            session.save_state()
        self._shared_session = session
        return session
