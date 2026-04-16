"""Gym environment adapter for Draft Buddy RL training."""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces

from draft_buddy.config import Config
from draft_buddy.core import DraftController, DraftState, FantasyRulesEngine, create_bot_gm
from draft_buddy.data import load_player_catalog
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.rl.checkpoint_manager import CheckpointManager
from draft_buddy.rl.feature_extractor import FeatureExtractor
from draft_buddy.rl.policy_network import PolicyNetwork
from draft_buddy.rl.state_normalizer import StateNormalizer
from draft_buddy.simulator import generate_round_robin_schedule


class DraftGymEnv(gym.Env):
    """Custom Gym environment for a fantasy football draft."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: Config,
        training: bool = False,
        state_normalizer: Optional[StateNormalizer] = None,
        draft_state: Optional[DraftState] = None,
        rules_engine: Optional[FantasyRulesEngine] = None,
        player_catalog=None,
    ) -> None:
        """Initialize the environment with injected draft dependencies.

        Parameters
        ----------
        config : Config
            Runtime configuration object.
        training : bool, optional
            Whether the environment is used for training rollouts.
        state_normalizer : Optional[StateNormalizer], optional
            State normalizer dependency.
        draft_state : Optional[DraftState], optional
            Existing draft state to reuse.
        rules_engine : Optional[FantasyRulesEngine], optional
            Draft rules dependency.
        player_catalog : Optional[PlayerCatalog], optional
            Shared player catalog. When omitted, it is loaded from configured CSV data.
        """
        super().__init__()
        self.config = config
        self.training = training
        self.manual_draft_teams = set(config.draft.MANUAL_DRAFT_TEAMS)
        self.player_catalog = player_catalog or load_player_catalog(
            config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG
        )
        self.action_to_position = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}
        self.position_to_action = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        self.action_space = spaces.Discrete(len(self.action_to_position))
        self.observation_space_dim = len(config.training.ENABLED_STATE_FEATURES)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_space_dim,),
            dtype=np.float32,
        )
        self.total_roster_size_per_team = (
            sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE
        )
        self._state = draft_state or DraftState(
            all_player_ids=set(self.player_catalog.player_ids),
            draft_order=self._generate_snake_draft_order(
                self.config.draft.NUM_TEAMS, self.total_roster_size_per_team
            ),
            roster_structure=self.config.draft.ROSTER_STRUCTURE,
            bench_maxes=self.config.draft.BENCH_MAXES,
            total_roster_size_per_team=self.total_roster_size_per_team,
            agent_team_id=self.config.draft.AGENT_START_POSITION,
        )
        self._rules = rules_engine or FantasyRulesEngine(
            roster_structure=self.config.draft.ROSTER_STRUCTURE,
            bench_maxes=self.config.draft.BENCH_MAXES,
            total_roster_size_per_team=self.total_roster_size_per_team,
        )
        self.opponent_models: Dict[int, PolicyNetwork] = {}
        self._controller = DraftController(
            state=self._state,
            player_catalog=self.player_catalog,
            rules_engine=self._rules,
            action_to_position=self.action_to_position,
            bot_factory=self._create_bot_strategy,
        )
        self._state_normalizer = state_normalizer or StateNormalizer(self.config)
        self._feature_extractor = FeatureExtractor(self.config, self._state_normalizer)
        self.agent_model: Optional[PolicyNetwork] = None
        self._sorted_available_by_pos_cache: Dict[str, List[object]] = {}
        self._load_opponent_models()
        self._load_agent_model()
        self.weekly_projections = self.player_catalog.to_weekly_projections()
        self.matchups_df = self._load_matchups()

    @property
    def available_player_ids(self) -> set[int]:
        """Return currently available player ids."""
        return self._state.available_player_ids

    @property
    def team_rosters(self):
        """Return typed team rosters."""
        return self._state.team_rosters

    @property
    def draft_order(self) -> list[int]:
        """Return draft order."""
        return self._state.draft_order

    @property
    def current_pick_index(self) -> int:
        """Return zero-based pick index."""
        return self._state.current_pick_index

    @current_pick_index.setter
    def current_pick_index(self, value: int) -> None:
        self._state.current_pick_index = value

    @property
    def current_pick_number(self) -> int:
        """Return one-based pick number."""
        return self._state.current_pick_number

    @current_pick_number.setter
    def current_pick_number(self, value: int) -> None:
        self._state.current_pick_number = value

    @property
    def agent_team_id(self) -> int:
        """Return agent team id."""
        return self._state.agent_team_id

    @agent_team_id.setter
    def agent_team_id(self, value: int) -> None:
        self._state.agent_team_id = value

    @property
    def override_team_id(self) -> Optional[int]:
        """Return temporary override team id."""
        return self._state.override_team_id

    @override_team_id.setter
    def override_team_id(self, value: Optional[int]) -> None:
        self._state.override_team_id = value

    @property
    def num_teams(self) -> int:
        """Return number of teams."""
        return self.config.draft.NUM_TEAMS

    @property
    def team_manager_mapping(self) -> Dict[int, str]:
        """Return team manager mapping."""
        return self.config.draft.TEAM_MANAGER_MAPPING

    def resolve_roster_players(self, team_id: int) -> list[object]:
        """Resolve roster player ids to Player objects."""
        return self._controller.resolve_roster_players(team_id)

    def save_state(self, file_path: str) -> None:
        """Save current draft state."""
        self._controller.save_state(file_path)

    def load_state(self, file_path: str) -> None:
        """Load saved draft state when present."""
        self._controller.load_state(file_path)
        self._invalidate_sorted_available_cache()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment state."""
        super().reset(seed=seed)
        if self.config.draft.RANDOMIZE_AGENT_START_POSITION and self.training:
            agent_team_id = random.randint(1, self.config.draft.NUM_TEAMS)
        else:
            agent_team_id = self.config.draft.AGENT_START_POSITION
        self._controller.reset(
            draft_order=self._generate_snake_draft_order(
                self.config.draft.NUM_TEAMS, self.total_roster_size_per_team
            ),
            agent_team_id=agent_team_id,
        )
        self._invalidate_sorted_available_cache()

        if self.training:
            while (
                self.current_pick_index < len(self.draft_order)
                and self.draft_order[self.current_pick_index] != self.agent_team_id
            ):
                self._controller.simulate_single_pick(
                    manual_draft_teams=set(),
                    build_state_fn=self._build_state_for_team,
                    get_action_mask_fn=self._get_action_mask_for_team,
                )
                self._invalidate_sorted_available_cache()

        perspective_team_id = (
            self.draft_order[self.current_pick_index]
            if self.current_pick_index < len(self.draft_order)
            else self.agent_team_id
        )
        observation = self._feature_extractor.extract(
            self._state, self.player_catalog, perspective_team_id
        )
        info = self._get_info()
        if (
            self.current_pick_index >= len(self.draft_order)
            and not self.resolve_roster_players(self.agent_team_id)
        ):
            info["episode_ended_before_agent_first_pick"] = True
        info["action_mask"] = self.get_action_mask()
        return observation, info

    def step(self, action: int):
        """Advance the environment by one agent action."""
        current_team_id = self.draft_order[self.current_pick_index]
        if current_team_id != self.agent_team_id:
            raise AssertionError(
                f"Expected agent team {self.agent_team_id}, got {current_team_id}."
            )

        selected_position = self.action_to_position[action]
        reward = 0.0
        done = False
        info: Dict[str, object] = {}

        pre_starter_points = 0.0
        if (
            self.config.reward.ENABLE_PICK_SHAPING_REWARD
            and self.config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD
        ):
            from draft_buddy.core.roster_utils import calculate_roster_scores

            current_scores = calculate_roster_scores(
                self.resolve_roster_players(self.agent_team_id),
                self.config.draft.ROSTER_STRUCTURE,
                self.config.draft.BENCH_MAXES,
            )
            pre_starter_points = current_scores["starters_total_points"]

        is_valid_pick, drafted_player = self._controller.try_select_player_for_team(
            self.agent_team_id, selected_position, self.available_player_ids
        )
        if not is_valid_pick or drafted_player is None:
            if self.config.reward.ENABLE_INVALID_ACTION_PENALTIES:
                penalty_key = f"roster_full_{selected_position}"
                if not self._get_best_available_player_by_pos(selected_position):
                    penalty_key = "no_players_available"
                reward += self.config.reward.INVALID_ACTION_PENALTIES.get(
                    penalty_key,
                    self.config.reward.INVALID_ACTION_PENALTIES["default_invalid"],
                )
            done = True
            info["invalid_action"] = True
            info["reason"] = f"Invalid pick: {selected_position}"
        else:
            self._controller.apply_pick(
                team_id=self.agent_team_id,
                player_id=drafted_player.player_id,
                is_manual_pick=False,
            )
            self._invalidate_sorted_available_cache()
            info["drafted_player"] = drafted_player.name
            info["drafted_position"] = drafted_player.position
            info["drafted_points"] = drafted_player.projected_points
            info["drafted_adp"] = drafted_player.adp

            from draft_buddy.rl.reward_calculator import RewardCalculator

            step_reward, step_info = RewardCalculator.calculate_step_reward(
                self.config, self, drafted_player, pre_starter_points
            )
            reward += step_reward
            info.update(step_info)

        if not done:
            while (
                self.current_pick_index < len(self.draft_order)
                and self.draft_order[self.current_pick_index] != self.agent_team_id
            ):
                self._controller.simulate_single_pick(
                    manual_draft_teams=set(),
                    build_state_fn=self._build_state_for_team,
                    get_action_mask_fn=self._get_action_mask_for_team,
                )
                self._invalidate_sorted_available_cache()

        if self.team_rosters[self.agent_team_id].size >= self.total_roster_size_per_team:
            done = True
            info["draft_complete"] = True
        elif self.current_pick_index >= len(self.draft_order):
            done = True
            info["draft_ended_prematurely"] = True

        if done:
            from draft_buddy.rl.reward_calculator import RewardCalculator

            final_reward, final_info = RewardCalculator.calculate_final_reward(
                self.config, self, self.matchups_df
            )
            reward += final_reward
            info.update(final_info)
            info["final_reward_total"] = reward

        perspective_team_id = (
            self.draft_order[self.current_pick_index]
            if self.current_pick_index < len(self.draft_order)
            else self.agent_team_id
        )
        observation = self._feature_extractor.extract(
            self._state, self.player_catalog, perspective_team_id
        )
        info["action_mask"] = self.get_action_mask()
        return observation, reward, done, False, info

    def draft_player(self, player_id: int) -> None:
        """Manually draft a player for the team on the clock."""
        self._controller.draft_player(player_id, is_manual_pick=True)
        self._invalidate_sorted_available_cache()

    def undo_last_pick(self) -> None:
        """Undo the latest pick."""
        self._controller.undo_last_pick()
        self._invalidate_sorted_available_cache()

    def set_current_team_picking(self, team_id: int) -> None:
        """Override the next drafting team."""
        if team_id not in range(1, self.num_teams + 1):
            raise ValueError(f"Invalid team ID: {team_id}. Must be between 1 and {self.num_teams}.")
        self._controller.set_override_team(team_id)

    def simulate_single_pick(self) -> None:
        """Simulate a single non-manual team pick."""
        self._controller.simulate_single_pick(
            manual_draft_teams=self.manual_draft_teams,
            build_state_fn=self._build_state_for_team,
            get_action_mask_fn=self._get_action_mask_for_team,
        )
        self._invalidate_sorted_available_cache()

    def get_action_mask(self) -> np.ndarray:
        """Return valid actions for the agent team."""
        return self._controller.get_action_mask_for_team(self.agent_team_id)

    def get_ai_suggestion(self):
        """Return AI suggestion for the team on the clock."""
        team_on_clock = self._controller.team_on_clock
        if team_on_clock is None:
            return {"error": "Draft is over."}
        return self.get_ai_suggestion_for_team(team_on_clock)

    def get_ai_suggestion_for_team(self, team_id: int, ignore_player_ids: Optional[List[int]] = None):
        """Return AI suggestion for a specific team."""
        if not (1 <= team_id <= self.num_teams):
            return {"error": f"Invalid team id {team_id}"}
        if not self.agent_model:
            return {"error": "AI model not loaded."}

        original_agent_team_id = self.agent_team_id
        original_available_ids = None
        self.agent_team_id = team_id
        try:
            if ignore_player_ids:
                ignore_set = {
                    player_id for player_id in ignore_player_ids if player_id in self.available_player_ids
                }
                if ignore_set:
                    original_available_ids = set(self.available_player_ids)
                    self._state.available_player_ids = original_available_ids - ignore_set
                    self._invalidate_sorted_available_cache()
            state = self._get_state_for_team(team_id)
            action_mask = self.get_action_mask()
        finally:
            self.agent_team_id = original_agent_team_id
            if original_available_ids is not None:
                self._state.available_player_ids = original_available_ids
                self._invalidate_sorted_available_cache()

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs_tensor = self.agent_model.get_action_probabilities(
                state_tensor, action_mask=action_mask
            )
        action_probs = action_probs_tensor.squeeze().tolist()
        return {self.action_to_position[i]: prob for i, prob in enumerate(action_probs)}

    def get_ai_suggestions_all(self):
        """Return AI action probabilities for all teams."""
        if not self.agent_model:
            return {"error": "AI model not loaded."}
        return {
            team_id: self.get_ai_suggestion_for_team(team_id)
            for team_id in range(1, self.num_teams + 1)
        }

    def get_draft_summary(self):
        """Return simple draft summary data."""
        summary = {
            "total_picks": self.current_pick_number - 1,
            "picks_by_position": {"QB": 0, "RB": 0, "WR": 0, "TE": 0},
        }
        for pick in self._state.draft_history:
            player = self.player_catalog.get(pick.player_id)
            if player:
                summary["picks_by_position"][player.position] += 1
        return summary

    def _load_agent_model(self) -> None:
        """Load the primary agent model for AI suggestions."""
        model_path = self.config.training.MODEL_PATH_TO_LOAD
        if not model_path or not os.path.exists(model_path):
            self.agent_model = None
            return
        input_dim = len(self.config.training.ENABLED_STATE_FEATURES)
        output_dim = self.action_space.n
        policy_network = PolicyNetwork(input_dim, output_dim, self.config.training.HIDDEN_DIM)
        checkpoint_manager = CheckpointManager(policy_network, value_network=None, optimizer=None)
        try:
            checkpoint_manager.load_checkpoint(model_path, self.config, is_training=False)
            self.agent_model = policy_network
        except Exception:
            self.agent_model = None

    def _load_opponent_models(self) -> None:
        """Load trained policy models for configured opponent teams."""
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            if team_id == self.config.draft.AGENT_START_POSITION:
                continue
            opponent_strategy = self.config.opponent.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self.config.opponent.DEFAULT_OPPONENT_STRATEGY
            )
            if opponent_strategy.get("logic") != "AGENT_MODEL":
                continue
            model_path_key = opponent_strategy.get("model_path_key")
            model_path = self.config.opponent.OPPONENT_MODEL_PATHS.get(model_path_key, "")
            if not model_path or not os.path.exists(model_path):
                continue
            input_dim = len(self.config.training.ENABLED_STATE_FEATURES)
            output_dim = self.action_space.n
            policy_network = PolicyNetwork(input_dim, output_dim, self.config.training.HIDDEN_DIM)
            checkpoint_manager = CheckpointManager(policy_network, value_network=None, optimizer=None)
            try:
                checkpoint_manager.load_checkpoint(model_path, self.config, is_training=False)
                self.opponent_models[team_id] = policy_network
            except Exception:
                continue

    def _create_bot_strategy(self, team_id: int):
        """Create a configured opponent bot."""
        strategy_config = self.config.opponent.OPPONENT_TEAM_STRATEGIES.get(
            team_id, self.config.opponent.DEFAULT_OPPONENT_STRATEGY
        )
        if strategy_config.get("logic") == "AGENT_MODEL" and team_id in self.opponent_models:
            return AgentModelBotGM(self.opponent_models[team_id], self.action_to_position)
        return create_bot_gm(
            logic=strategy_config["logic"],
            config=strategy_config,
            opponent_models=self.opponent_models,
            team_id=team_id,
            action_to_position=self.action_to_position,
        )

    def _load_matchups(self) -> pd.DataFrame:
        """Load configured matchup schedule or generate one when requested."""
        if self.config.reward.USE_RANDOM_MATCHUPS:
            manager_names = [
                self.team_manager_mapping.get(team_id)
                for team_id in range(1, self.config.draft.NUM_TEAMS + 1)
                if self.team_manager_mapping.get(team_id)
            ]
            num_weeks = int(self.config.reward.NUM_REGULAR_SEASON_WEEKS)
            return generate_round_robin_schedule(manager_names, num_weeks)
        default_matchups_filename = "red_league_matchups_2025.csv"
        size_specific_filename = (
            f"red_league_matchups_2025_{self.config.draft.NUM_TEAMS}_team.csv"
        )
        candidates = [
            os.path.join(self.config.paths.DATA_DIR, size_specific_filename),
            os.path.join(self.config.paths.DATA_DIR, default_matchups_filename),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return pd.read_csv(candidate)
        return pd.DataFrame()

    def _generate_snake_draft_order(self, num_teams: int, total_picks_per_team: int) -> list[int]:
        """Generate the full snake draft order."""
        draft_order: list[int] = []
        num_rounds = math.ceil(total_picks_per_team)
        if num_rounds * num_teams > len(self.player_catalog):
            num_rounds = max(1, math.floor(len(self.player_catalog) / num_teams))
        for round_number in range(num_rounds):
            if round_number % 2 == 0:
                draft_order.extend(range(1, num_teams + 1))
            else:
                draft_order.extend(range(num_teams, 0, -1))
        return draft_order

    def _get_info(self) -> Dict[str, object]:
        """Return lightweight info about the current draft state."""
        return {
            "current_pick_number": self.current_pick_number,
            "current_team_picking": self._controller.team_on_clock,
            "agent_roster_size": self.team_rosters[self.agent_team_id].size,
            "available_players_count": len(self.available_player_ids),
            "manual_draft_teams": list(self.manual_draft_teams),
        }

    def _get_state_for_team(self, team_id: int) -> np.ndarray:
        """Return normalized state from one team's perspective."""
        return self._feature_extractor.extract(self._state, self.player_catalog, team_id)

    def _get_state(self) -> np.ndarray:
        """Return normalized state for the current team on the clock."""
        perspective_team_id = (
            self.draft_order[self.current_pick_index]
            if self.current_pick_index < len(self.draft_order)
            else self.agent_team_id
        )
        return self._get_state_for_team(perspective_team_id)

    def _compute_global_state_features(self) -> Dict[str, float]:
        """Return global features for the current draft state."""
        return self._feature_extractor.compute_global_state_features(
            self._state, self.player_catalog
        )

    def _build_state_for_team(self, team_id: int) -> np.ndarray:
        """Return normalized state for one team."""
        return self._get_state_for_team(team_id)

    def _build_state_for_team_from_global(
        self, team_id: int, global_features: Dict[str, float]
    ) -> np.ndarray:
        """Return normalized state using precomputed global features."""
        state_map = self._feature_extractor.build_state_map_for_team(
            draft_state=self._state,
            player_catalog=self.player_catalog,
            team_id=team_id,
            global_features=global_features,
        )
        return self._state_normalizer.normalize(state_map)

    def _get_action_mask_for_team(self, team_id: int) -> np.ndarray:
        """Return valid-action mask for one team."""
        return self._controller.get_action_mask_for_team(team_id)

    def _get_sorted_available_for_position(self, position: str) -> List[object]:
        """Return sorted available players for one position."""
        if not self._sorted_available_by_pos_cache:
            self._build_sorted_available_cache()
        return self._sorted_available_by_pos_cache.get(position, [])

    def _build_sorted_available_cache(self) -> None:
        """Build sorted available-player cache by position."""
        cache = defaultdict(list)
        for player_id in self.available_player_ids:
            player = self.player_catalog.require(player_id)
            cache[player.position].append(player)
        for position in self.action_to_position.values():
            cache[position].sort(key=lambda player: player.projected_points, reverse=True)
        self._sorted_available_by_pos_cache = dict(cache)

    def _invalidate_sorted_available_cache(self) -> None:
        """Clear cached sorted-availability data."""
        self._sorted_available_by_pos_cache = {}

    def _get_dynamic_baseline_for_position(self, position: str, available_for_pos: List[object]) -> float:
        """Calculate smoothed replacement baseline for a sorted positional list."""
        if not available_for_pos:
            return 0.0
        num_starters_needed = 0
        required_starters = self.config.draft.ROSTER_STRUCTURE.get(position, 0)
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            roster = self.team_rosters[team_id]
            num_starters_needed += max(0, required_starters - roster.position_count(position))
        replacement_rank = num_starters_needed + 2
        last_index = len(available_for_pos) - 1
        replacement_index = min(replacement_rank - 1, last_index)
        score_at = available_for_pos[replacement_index].projected_points
        score_before = available_for_pos[max(0, replacement_index - 1)].projected_points
        score_after = available_for_pos[min(last_index, replacement_index + 1)].projected_points
        return (score_before + score_at + score_after) / 3.0

    def _calculate_vorp(self, position: str) -> float:
        """Calculate VORP for the best available player at a position."""
        available_for_pos = self._get_sorted_available_for_position(position)
        if not available_for_pos:
            return 0.0
        baseline = self._get_dynamic_baseline_for_position(position, available_for_pos)
        return available_for_pos[0].projected_points - baseline

    def get_positional_baselines(self) -> Dict[str, float]:
        """Return dynamic baselines by position."""
        self._build_sorted_available_cache()
        return {
            position: self._get_dynamic_baseline_for_position(
                position, self._get_sorted_available_for_position(position)
            )
            for position in self.action_to_position.values()
        }

    def _get_kth_best_available_player_by_pos(self, position: str, k: int):
        """Return the k-th best available player at a position."""
        players = self._get_sorted_available_for_position(position)
        if len(players) < k:
            return None
        return players[k - 1]

    def _get_best_available_player_by_pos(self, position: str):
        """Return the best available player at a position."""
        return self._get_kth_best_available_player_by_pos(position, 1)

    def _try_select_player_for_team(
        self, team_id: int, position_choice: str, available_player_ids: set[int]
    ) -> Tuple[bool, Optional[object]]:
        """Return the best available player for a team and position."""
        return self._controller.try_select_player_for_team(
            team_id, position_choice, available_player_ids
        )

    def _simulate_competing_pick(
        self, team_id: int, global_features: Optional[Dict[str, float]] = None
    ):
        """Return the simulated pick for a competing team."""
        if global_features is None:
            build_state_fn = self._build_state_for_team
        else:
            build_state_fn = lambda target_team_id: self._build_state_for_team_from_global(
                target_team_id, global_features
            )
        return self._controller._select_bot_pick(
            team_id=team_id,
            build_state_fn=build_state_fn,
            get_action_mask_fn=self._get_action_mask_for_team,
        )

    def _calculate_imminent_threat(self, position: str) -> int:
        """Return imminent threat count for the agent team."""
        return self._feature_extractor.calculate_imminent_threat(
            self._state, self.team_rosters, self.agent_team_id, position
        )

    def _get_agent_bye_week_vector(self) -> np.ndarray:
        """Return agent bye-week histogram."""
        return self._feature_extractor.get_agent_bye_week_vector(
            self._state, self.team_rosters, self.player_catalog
        )

    def _get_bye_week_conflict_count(self, position: str) -> int:
        """Return bye-week conflict count for the best available player."""
        return self._get_bye_week_conflict_count_for_team(self.agent_team_id, position)

    def _get_bye_week_conflict_count_for_team(self, team_id: int, position: str) -> int:
        """Return bye-week conflict count for one team and position."""
        best_player = self._get_best_available_player_by_pos(position)
        if not best_player or not best_player.bye_week or np.isnan(best_player.bye_week):
            return 0
        roster_players = self.resolve_roster_players(team_id)
        return sum(1 for player in roster_players if player.bye_week == best_player.bye_week)

    def _get_stack_count_for_team(self, team_id: int) -> int:
        """Return stack count for a team."""
        from draft_buddy.core.stacking import calculate_stack_count

        return calculate_stack_count(self.resolve_roster_players(team_id))

    def _get_current_stack_count(self) -> int:
        """Return current stack count for the agent."""
        return self._get_stack_count_for_team(self.agent_team_id)

    def _get_stack_target_available_flag_for_team(self, team_id: int) -> int:
        """Return whether a stacking target is available for a team."""
        roster_players = self.resolve_roster_players(team_id)
        qb_teams = {player.team for player in roster_players if player.position == "QB" and player.team}
        if not qb_teams:
            return 0
        for player_id in self.available_player_ids:
            player = self.player_catalog.require(player_id)
            if player.position in {"WR", "TE"} and player.team in qb_teams:
                return 1
        return 0

    def _get_stack_target_available_flag(self) -> int:
        """Return whether the agent has an available stack target."""
        return self._get_stack_target_available_flag_for_team(self.agent_team_id)
