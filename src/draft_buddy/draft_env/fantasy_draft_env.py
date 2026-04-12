import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import math
import random
import torch # Import torch for loading models
from typing import Optional, Dict, List, Tuple
import os

from draft_buddy.config import Config
from draft_buddy.utils.data_utils import load_player_data, Player, calculate_stack_count
from draft_buddy.draft_env.draft_rules import FantasyDraftRules
from draft_buddy.draft_env.draft_state import DraftState
from draft_buddy.logic.opponent_strategies import create_opponent_strategy, OpponentStrategy
from draft_buddy.models.checkpoint_manager import CheckpointManager
from draft_buddy.models.policy_network import PolicyNetwork
import json
from draft_buddy.utils.season_simulation_fast import simulate_season_fast, generate_round_robin_schedule
import pandas as pd

class FantasyFootballDraftEnv(gym.Env):
    """
    Custom OpenAI Gym Environment for a Fantasy Football Draft.
    The agent learns to pick optimal positions to maximize projected team points.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config: Config, training: bool = False):
        super(FantasyFootballDraftEnv, self).__init__()
        self.config = config
        self.training = training
        self.manual_draft_teams = set(config.draft.MANUAL_DRAFT_TEAMS) # Teams controlled by the user

        # --- Load Players Data ---
        self.all_players_data = load_player_data(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)
        self.player_map = {p.player_id: p for p in self.all_players_data}

        # --- Environment Dimensions ---
        # Action Space: 4 discrete actions (select QB, RB, WR, or TE)
        # Note: FLEX is not an action, but a roster slot that RB/WR/TE can fill.
        self.action_to_position = {0: 'QB', 1: 'RB', 2: 'WR', 3: 'TE'}
        self.position_to_action = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
        self.action_space = spaces.Discrete(len(self.action_to_position))

        # State Space: Dynamically determined by ENABLED_STATE_FEATURES
        self.state_features_map = {
            # Player availability info
            "best_available_qb_points": lambda: self._get_kth_best_available_player_by_pos('QB', 1).projected_points if self._get_kth_best_available_player_by_pos('QB', 1) else 0,
            "best_available_rb_points": lambda: self._get_kth_best_available_player_by_pos('RB', 1).projected_points if self._get_kth_best_available_player_by_pos('RB', 1) else 0,
            "best_available_wr_points": lambda: self._get_kth_best_available_player_by_pos('WR', 1).projected_points if self._get_kth_best_available_player_by_pos('WR', 1) else 0,
            "best_available_te_points": lambda: self._get_kth_best_available_player_by_pos('TE', 1).projected_points if self._get_kth_best_available_player_by_pos('TE', 1) else 0,
            "best_available_qb_vorp": lambda: self._calculate_vorp('QB'),
            "best_available_rb_vorp": lambda: self._calculate_vorp('RB'),
            "best_available_wr_vorp": lambda: self._calculate_vorp('WR'),
            "best_available_te_vorp": lambda: self._calculate_vorp('TE'),
            "qb_available_flag": lambda: 1 if any(p.position == 'QB' and p.player_id in self.available_players_ids for p in self.all_players_data) else 0,
            "rb_available_flag": lambda: 1 if any(p.position == 'RB' and p.player_id in self.available_players_ids for p in self.all_players_data) else 0,
            "wr_available_flag": lambda: 1 if any(p.position == 'WR' and p.player_id in self.available_players_ids for p in self.all_players_data) else 0,
            "te_available_flag": lambda: 1 if any(p.position == 'TE' and p.player_id in self.available_players_ids for p in self.all_players_data) else 0,
            # Current Roster Counts (for agent's team)
            "current_roster_qb_count": lambda: self.teams_rosters[self.agent_team_id]['QB'],
            "current_roster_rb_count": lambda: self.teams_rosters[self.agent_team_id]['RB'],
            "current_roster_wr_count": lambda: self.teams_rosters[self.agent_team_id]['WR'],
            "current_roster_te_count": lambda: self.teams_rosters[self.agent_team_id]['TE'],
            # Available Roster Slots (for agent's team)
            "available_roster_slots_qb": lambda: self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'] - self.teams_rosters[self.agent_team_id]['QB'],
            "available_roster_slots_rb": lambda: self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'] - self.teams_rosters[self.agent_team_id]['RB'],
            "available_roster_slots_wr": lambda: self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'] - self.teams_rosters[self.agent_team_id]['WR'],
            "available_roster_slots_te": lambda: self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'] - self.teams_rosters[self.agent_team_id]['TE'],
            "available_roster_slots_flex": lambda: self.config.draft.ROSTER_STRUCTURE['FLEX'] - self.teams_rosters[self.agent_team_id]['FLEX'], # Simplified flex calculation
            # Draft Progression Context
            "current_pick_number": lambda: self.current_pick_number,
            "agent_start_position": lambda: self.agent_team_id,
            # New features
            "second_best_available_qb_points": lambda: self._get_kth_best_available_player_by_pos('QB', 2).projected_points if self._get_kth_best_available_player_by_pos('QB', 2) else 0,
            "second_best_available_rb_points": lambda: self._get_kth_best_available_player_by_pos('RB', 2).projected_points if self._get_kth_best_available_player_by_pos('RB', 2) else 0,
            "second_best_available_wr_points": lambda: self._get_kth_best_available_player_by_pos('WR', 2).projected_points if self._get_kth_best_available_player_by_pos('WR', 2) else 0,
            "second_best_available_te_points": lambda: self._get_kth_best_available_player_by_pos('TE', 2).projected_points if self._get_kth_best_available_player_by_pos('TE', 2) else 0,
            "next_pick_opponent_qb_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'QB'),
            "next_pick_opponent_rb_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'RB'),
            "next_pick_opponent_wr_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'WR'),
            "next_pick_opponent_te_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'TE'),
            "best_qb_bye_week_conflict": lambda: self._get_bye_week_conflict_count('QB'),
            "best_rb_bye_week_conflict": lambda: self._get_bye_week_conflict_count('RB'),
            "best_wr_bye_week_conflict": lambda: self._get_bye_week_conflict_count('WR'),
            "best_te_bye_week_conflict": lambda: self._get_bye_week_conflict_count('TE'),

            # --- TIER 2 FEATURES ---

            # 1. Positional Scarcity (Drop-off Score)
            "qb_scarcity": lambda: self._calculate_scarcity('QB'),
            "rb_scarcity": lambda: self._calculate_scarcity('RB'),
            "wr_scarcity": lambda: self._calculate_scarcity('WR'),
            "te_scarcity": lambda: self._calculate_scarcity('TE'),

            # 2. Top-k Literal Data (Hybrid Component)
            "top_3_qb_points_1": lambda: self._get_kth_best_available_player_by_pos('QB', 1).projected_points if self._get_kth_best_available_player_by_pos('QB', 1) else 0.0,
            "top_3_qb_points_2": lambda: self._get_kth_best_available_player_by_pos('QB', 2).projected_points if self._get_kth_best_available_player_by_pos('QB', 2) else 0.0,
            "top_3_qb_points_3": lambda: self._get_kth_best_available_player_by_pos('QB', 3).projected_points if self._get_kth_best_available_player_by_pos('QB', 3) else 0.0,
            "top_3_rb_points_1": lambda: self._get_kth_best_available_player_by_pos('RB', 1).projected_points if self._get_kth_best_available_player_by_pos('RB', 1) else 0.0,
            "top_3_rb_points_2": lambda: self._get_kth_best_available_player_by_pos('RB', 2).projected_points if self._get_kth_best_available_player_by_pos('RB', 2) else 0.0,
            "top_3_rb_points_3": lambda: self._get_kth_best_available_player_by_pos('RB', 3).projected_points if self._get_kth_best_available_player_by_pos('RB', 3) else 0.0,
            "top_3_wr_points_1": lambda: self._get_kth_best_available_player_by_pos('WR', 1).projected_points if self._get_kth_best_available_player_by_pos('WR', 1) else 0.0,
            "top_3_wr_points_2": lambda: self._get_kth_best_available_player_by_pos('WR', 2).projected_points if self._get_kth_best_available_player_by_pos('WR', 2) else 0.0,
            "top_3_wr_points_3": lambda: self._get_kth_best_available_player_by_pos('WR', 3).projected_points if self._get_kth_best_available_player_by_pos('WR', 3) else 0.0,
            "top_3_te_points_1": lambda: self._get_kth_best_available_player_by_pos('TE', 1).projected_points if self._get_kth_best_available_player_by_pos('TE', 1) else 0.0,
            "top_3_te_points_2": lambda: self._get_kth_best_available_player_by_pos('TE', 2).projected_points if self._get_kth_best_available_player_by_pos('TE', 2) else 0.0,
            "top_3_te_points_3": lambda: self._get_kth_best_available_player_by_pos('TE', 3).projected_points if self._get_kth_best_available_player_by_pos('TE', 3) else 0.0,

            # 3. Opponent Threat Analysis (Imminent Threat)
            "qb_imminent_threat": lambda: self._calculate_imminent_threat('QB'),
            "rb_imminent_threat": lambda: self._calculate_imminent_threat('RB'),
            "wr_imminent_threat": lambda: self._calculate_imminent_threat('WR'),
            "te_imminent_threat": lambda: self._calculate_imminent_threat('TE'),

            # 4. Bye Week Management (Full Roster Vector)
            "bye_week_4_count": lambda: self._get_agent_bye_week_vector()[0],
            "bye_week_5_count": lambda: self._get_agent_bye_week_vector()[1],
            "bye_week_6_count": lambda: self._get_agent_bye_week_vector()[2],
            "bye_week_7_count": lambda: self._get_agent_bye_week_vector()[3],
            "bye_week_8_count": lambda: self._get_agent_bye_week_vector()[4],
            "bye_week_9_count": lambda: self._get_agent_bye_week_vector()[5],
            "bye_week_10_count": lambda: self._get_agent_bye_week_vector()[6],
            "bye_week_11_count": lambda: self._get_agent_bye_week_vector()[7],
            "bye_week_12_count": lambda: self._get_agent_bye_week_vector()[8],
            "bye_week_13_count": lambda: self._get_agent_bye_week_vector()[9],
            "bye_week_14_count": lambda: self._get_agent_bye_week_vector()[10],

            # 5. Positional Stacking Features
            "current_stack_count": lambda: self._get_current_stack_count(),
            "stack_target_available_flag": lambda: self._get_stack_target_available_flag(),
        }
        self.observation_space_dim = len(config.training.ENABLED_STATE_FEATURES)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)

        total_roster_size_per_team = sum(self.config.draft.ROSTER_STRUCTURE.values()) + self.config.draft.TOTAL_BENCH_SIZE
        self.total_roster_size_per_team = total_roster_size_per_team

        all_player_ids = {p.player_id for p in self.all_players_data}
        draft_order = self._generate_snake_draft_order(self.config.draft.NUM_TEAMS, total_roster_size_per_team)
        self._state = DraftState(
            all_player_ids=all_player_ids,
            draft_order=draft_order,
            roster_structure=self.config.draft.ROSTER_STRUCTURE,
            bench_maxes=self.config.draft.BENCH_MAXES,
            total_roster_size_per_team=total_roster_size_per_team,
            agent_team_id=self.config.draft.AGENT_START_POSITION,
        )
        self._rules = FantasyDraftRules(
            roster_structure=self.config.draft.ROSTER_STRUCTURE,
            bench_maxes=self.config.draft.BENCH_MAXES,
            total_roster_size_per_team=total_roster_size_per_team,
        )

        self.opponent_models: Dict[int, PolicyNetwork] = {}
        self._opponent_strategies: Dict[int, OpponentStrategy] = {}
        self._load_opponent_models()

        self.agent_model: Optional[PolicyNetwork] = None
        self._load_agent_model()

        if getattr(self.config, "USE_RANDOM_MATCHUPS", False):
            manager_names = [self.config.draft.TEAM_MANAGER_MAPPING.get(tid) for tid in range(1, self.config.draft.NUM_TEAMS + 1)]
            manager_names = [m for m in manager_names if m]
            num_weeks = int(getattr(self.config, "NUM_REGULAR_SEASON_WEEKS", 14))
            self.matchups_df = generate_round_robin_schedule(manager_names, num_weeks)
        else:
            default_matchups_filename = "red_league_matchups_2025.csv"
            size_specific_filename = f"red_league_matchups_2025_{self.config.draft.NUM_TEAMS}_team.csv"
            candidate_paths = [
                os.path.join(self.config.paths.DATA_DIR, size_specific_filename),
                os.path.join(self.config.paths.DATA_DIR, default_matchups_filename),
            ]
            matchups_path = None
            for path in candidate_paths:
                if os.path.exists(path):
                    matchups_path = path
                    break
            if matchups_path is None:
                matchups_path = os.path.join(self.config.paths.DATA_DIR, default_matchups_filename)
            try:
                self.matchups_df = pd.read_csv(matchups_path)
            except FileNotFoundError:
                self.matchups_df = pd.DataFrame()
                if self.config.reward.ENABLE_SEASON_SIM_REWARD:
                    print(f"Warning: ENABLE_SEASON_SIM_REWARD is True, but matchups file not found at {matchups_path}. Season sim rewards will be disabled.")

        self.weekly_projections = self._create_weekly_projections()
        self._sorted_available_by_pos_cache: Dict[str, List[Player]] = {}
        from draft_buddy.utils.state_normalizer import StateNormalizer
        self._state_normalizer = StateNormalizer(self.config)

    @property
    def available_players_ids(self):
        """Delegates to DraftState."""
        return self._state.get_available_player_ids()

    @property
    def teams_rosters(self):
        """Delegates to DraftState."""
        return self._state.get_rosters()

    @property
    def draft_order(self):
        """Delegates to DraftState."""
        return self._state.get_draft_order()

    @property
    def current_pick_idx(self):
        """Delegates to DraftState."""
        return self._state.get_current_pick_idx()

    @current_pick_idx.setter
    def current_pick_idx(self, value):
        self._state.set_current_pick_idx(value)

    @property
    def current_pick_number(self):
        """Delegates to DraftState."""
        return self._state.get_current_pick_number()

    @current_pick_number.setter
    def current_pick_number(self, value):
        self._state.set_current_pick_number(value)

    @property
    def agent_team_id(self):
        """Delegates to DraftState."""
        return self._state.get_agent_team_id()

    @agent_team_id.setter
    def agent_team_id(self, value):
        self._state.set_agent_team_id(value)

    @property
    def _overridden_team_id(self):
        """Delegates to DraftState."""
        return self._state.get_overridden_team_id()

    @_overridden_team_id.setter
    def _overridden_team_id(self, value):
        self._state.set_overridden_team_id(value)

    @property
    def _draft_history(self):
        """Delegates to DraftState."""
        return self._state.get_draft_history()

    @property
    def roster_structure(self) -> Dict[str, int]:
        """Returns the league roster structure."""
        return self.config.draft.ROSTER_STRUCTURE

    @property
    def bench_maxes(self) -> Dict[str, int]:
        """Returns the league bench maximums."""
        return self.config.draft.BENCH_MAXES

    @property
    def num_teams(self) -> int:
        """Returns the number of teams in the league."""
        return self.config.draft.NUM_TEAMS

    @property
    def team_manager_mapping(self) -> Dict[int, str]:
        """Returns the mapping of team IDs to manager names."""
        return self.config.draft.TEAM_MANAGER_MAPPING

    def _get_bye_week_conflict_count(self, position: str) -> int:
        """
        Calculates how many players on the agent's roster share a bye week with the best available player of a given position.
        """
        best_player = self._get_best_available_player_by_pos(position)
        if not best_player or not best_player.bye_week or np.isnan(best_player.bye_week):
            return 0

        agent_roster = self.teams_rosters[self.agent_team_id]['PLAYERS']
        conflict_count = sum(
            1 for p in agent_roster if p.bye_week == best_player.bye_week
        )
        return conflict_count

    def _create_weekly_projections(self):
        """Build week-to-week point projections for each player."""
        weekly_projections = {}
        for player in self.all_players_data:
            weekly_projections[player.player_id] = {
                'pts': [player.projected_points] * 18,
                'pos': player.position
            }
        return weekly_projections

    def save_state(self, file_path: str):
        """Saves the current draft state to a file using an atomic write."""
        # Create a serializable version of teams_rosters
        serializable_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []})
        for team_id, roster_data in self.teams_rosters.items():
            serializable_rosters[team_id] = roster_data.copy()
            serializable_rosters[team_id]['PLAYERS'] = [p.to_dict() for p in roster_data['PLAYERS']]

        state = {
            'available_players_ids': list(self.available_players_ids),
            'teams_rosters': serializable_rosters,
            'draft_order': self.draft_order,
            'current_pick_idx': self.current_pick_idx,
            'current_pick_number': self.current_pick_number,
            '_draft_history': self._draft_history,
            '_overridden_team_id': self._overridden_team_id,
        }
        
        temp_file_path = file_path + ".tmp"
        try:
            with open(temp_file_path, 'w') as f:
                json.dump(state, f, indent=4)
            os.replace(temp_file_path, file_path)
        except Exception as e:
            print(f"Error saving draft state: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    def load_state(self, file_path: str):
        """Loads the draft state from a file with error handling."""
        if not os.path.exists(file_path):
            print("No saved draft state found.")
            return
        try:
            with open(file_path, "r") as f:
                state = json.load(f)
            self._state.load_from_serialized(state, lambda p: Player(**p))
            self._invalidate_sorted_available_cache()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load draft state from {file_path} due to error: {e}. Starting fresh.")
            self.reset()


    def _load_agent_model(self):
        """Loads the primary agent model for AI suggestions with config validation."""
        model_path = self.config.training.MODEL_PATH_TO_LOAD

        if not model_path or not os.path.exists(model_path):
            print(f"Warning: Agent model for suggestions not found at {model_path}. AI suggestions will be disabled.")
            return

        try:
            input_dim = len(self.config.training.ENABLED_STATE_FEATURES)
            output_dim = self.action_space.n
            policy_network = PolicyNetwork(input_dim, output_dim, self.config.training.HIDDEN_DIM)
            checkpoint_manager = CheckpointManager(
                policy_network, value_network=None, optimizer=None
            )
            checkpoint_manager.load_checkpoint(model_path, self.config, is_training=False)
            self.agent_model = policy_network
            print(f"Successfully loaded agent model for suggestions from {model_path}")
        except ValueError as e:
            print(f"CRITICAL WARNING: Agent model compatibility error for '{model_path}'.")
            print(f"  Reason: {e}")
            print("  AI suggestions will be disabled.")
            self.agent_model = None
        except FileNotFoundError:
            print(f"Warning: Agent model not found at {model_path}. AI suggestions will be disabled.")
            self.agent_model = None
        except Exception as e:
            print(f"Error loading agent model for suggestions from {model_path}: {e}")
            self.agent_model = None


    def _load_opponent_models(self):
        """
        Loads trained PolicyNetwork models for opponents with config validation.
        Only loads models for teams explicitly marked with 'AGENT_MODEL' logic.
        """
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            if team_id == self.config.draft.AGENT_START_POSITION:
                continue

            opponent_strategy = self.config.opponent.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self.config.opponent.DEFAULT_OPPONENT_STRATEGY
            )

            if opponent_strategy['logic'] == 'AGENT_MODEL':
                model_path_key = opponent_strategy.get('model_path_key')
                if not model_path_key or model_path_key not in self.config.opponent.OPPONENT_MODEL_PATHS:
                    print(f"Warning: 'model_path_key' missing or invalid for Team {team_id} configured as 'AGENT_MODEL'. Falling back to default strategy.")
                    self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.opponent.DEFAULT_OPPONENT_STRATEGY.copy()
                    continue

                model_path = self.config.opponent.OPPONENT_MODEL_PATHS[model_path_key]
                if not os.path.exists(model_path):
                    print(f"Warning: Opponent model not found at {model_path} for Team {team_id}. Falling back to default strategy.")
                    self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.opponent.DEFAULT_OPPONENT_STRATEGY.copy()
                    continue

                try:
                    input_dim = len(self.config.training.ENABLED_STATE_FEATURES)
                    output_dim = self.action_space.n
                    policy_network = PolicyNetwork(input_dim, output_dim, self.config.training.HIDDEN_DIM)
                    checkpoint_manager = CheckpointManager(
                        policy_network, value_network=None, optimizer=None
                    )
                    checkpoint_manager.load_checkpoint(model_path, self.config, is_training=False)
                    self.opponent_models[team_id] = policy_network
                    print(f"Loaded agent model for opponent Team {team_id} from {model_path}")
                except ValueError as e:
                    print(f"CRITICAL WARNING: Opponent model compatibility error for Team {team_id} at '{model_path}'.")
                    print(f"  Reason: {e}")
                    print("  Falling back to default strategy.")
                    self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.opponent.DEFAULT_OPPONENT_STRATEGY.copy()
                except FileNotFoundError:
                    print(f"Warning: Opponent model not found at {model_path} for Team {team_id}. Falling back to default strategy.")
                    self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.opponent.DEFAULT_OPPONENT_STRATEGY.copy()
                except Exception as e:
                    print(f"Error loading opponent model for Team {team_id} from {model_path}: {e}. Falling back to default strategy.")
                    self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.opponent.DEFAULT_OPPONENT_STRATEGY.copy()

        # Optional: Randomize non-agent opponent strategies per episode
        if self.config.opponent.RANDOMIZE_OPPONENT_STRATEGIES and (not self.config.opponent.RANDOMIZE_ONLY_DURING_TRAINING or self.training):
            self._randomize_opponent_strategies()

        # Build strategy instances for each opponent team
        self._build_opponent_strategies()

    def _randomize_opponent_strategies(self) -> None:
        """Randomizes opponent strategies from templates for diversity in training/interaction."""
        import random as _rnd
        templates = getattr(self.config, 'OPPONENT_STRATEGY_TEMPLATES', [])
        if not templates:
            return
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            if team_id == self.config.draft.AGENT_START_POSITION:
                continue
            current = self.config.opponent.OPPONENT_TEAM_STRATEGIES.get(team_id, self.config.opponent.DEFAULT_OPPONENT_STRATEGY)
            if current.get('logic') == 'AGENT_MODEL' and not self.config.opponent.RANDOMIZE_INCLUDE_AGENT_MODELS:
                continue
            tpl = _rnd.choice(templates)
            # Sample parameters
            low, high = tpl['randomness_factor_range']
            sampled_randomness = _rnd.uniform(low, high)
            sampled_subopt = _rnd.choice(tpl['suboptimal_strategy_choices'])
            sampled_priority = _rnd.choice(tpl['positional_priority_choices'])
            # Build new strategy
            randomized = {
                'logic': tpl['logic'],
                'randomness_factor': sampled_randomness,
                'suboptimal_strategy': sampled_subopt,
                'positional_priority': sampled_priority,
            }
            self.config.opponent.OPPONENT_TEAM_STRATEGIES[team_id] = randomized

    def _build_opponent_strategies(self) -> None:
        """Builds OpponentStrategy instances for each opponent team."""
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            if team_id == self.config.draft.AGENT_START_POSITION:
                continue
            strategy_config = self.config.opponent.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self.config.opponent.DEFAULT_OPPONENT_STRATEGY
            )
            self._opponent_strategies[team_id] = create_opponent_strategy(
                logic=strategy_config["logic"],
                config=strategy_config,
                opponent_models=self.opponent_models,
                team_id=team_id,
                action_to_position=self.action_to_position,
            )

    def _get_dynamic_baseline_for_position(self, position: str, available_for_pos: List[Player]) -> float:
        """Calculates the smoothed baseline score for a list of pre-sorted players."""
        # This assumes available_for_pos is already sorted by projected_points descending.
        if not available_for_pos:
            return 0.0

        num_starters_needed = 0
        if position in self.config.draft.ROSTER_STRUCTURE:
            required_starters = self.config.draft.ROSTER_STRUCTURE[position]
            for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
                roster_counts = self.teams_rosters[team_id]
                num_on_roster = roster_counts.get(position, 0)
                if num_on_roster < required_starters:
                    num_starters_needed += (required_starters - num_on_roster)

        replacement_rank = num_starters_needed + 2
        num_available = len(available_for_pos)
        last_idx = num_available - 1
        replacement_idx = min(replacement_rank - 1, last_idx)

        score_at = available_for_pos[replacement_idx].projected_points
        score_before = available_for_pos[max(0, replacement_idx - 1)].projected_points
        score_after = available_for_pos[min(last_idx, replacement_idx + 1)].projected_points

        return (score_before + score_at + score_after) / 3.0

    def _calculate_vorp(self, position: str) -> float:
        """
        Calculates the Value Over Replacement Player (VORP) for the best available
        player at a given position.

        This method is designed to be a self-contained, robust, and clear
        implementation that handles all steps from data filtering to the final
        VORP calculation.

        Args:
            position: The position ('QB', 'RB', 'WR', 'TE') to calculate VORP for.

        Returns:
            The calculated VORP score as a float.
        """
        # --- 1. Performance-intensive step: Filter and sort available players ---
        # This is the most expensive part of the VORP calculation. For higher
        # performance, the result of this operation could be cached across the four
        # VORP calculations within a single environment step.
        # Use cached, pre-sorted available players list
        available_for_pos = self._get_sorted_available_for_position(position)

        # --- 2. Handle edge case: No players available ---
        if not available_for_pos:
            return 0.0

        # --- 3. Calculate the smoothed baseline score ---
        baseline_score = self._get_dynamic_baseline_for_position(position, available_for_pos)

        # --- 4. Calculate final VORP ---
        # VORP is the score of the best available player minus the baseline.
        best_player_score = available_for_pos[0].projected_points
        vorp = best_player_score - baseline_score

        return vorp

    def _build_sorted_available_cache(self) -> None:
        """Builds and stores per-position sorted available players cache for this step."""
        players_by_pos: Dict[str, List[Player]] = defaultdict(list)
        for p_id in self.available_players_ids:
            player = self.player_map[p_id]
            players_by_pos[player.position].append(player)

        sorted_cache: Dict[str, List[Player]] = {}
        for position in self.action_to_position.values():
            sorted_cache[position] = sorted(
                players_by_pos[position],
                key=lambda p: p.projected_points,
                reverse=True
            )

        self._sorted_available_by_pos_cache = sorted_cache

    def _get_sorted_available_for_position(self, position: str) -> List[Player]:
        """Returns cached sorted available players for a position, building cache if missing."""
        if not self._sorted_available_by_pos_cache:
            self._build_sorted_available_cache()
        return self._sorted_available_by_pos_cache.get(position, [])

    def _invalidate_sorted_available_cache(self) -> None:
        """Invalidates the per-position sorted available players cache."""
        self._sorted_available_by_pos_cache = {}

    def get_positional_baselines(self) -> Dict[str, float]:
        """
        Calculates and returns the dynamic baseline score for each primary position.

        This method is optimized to filter and sort the available player pool only
        once per position group, making it efficient for frontend use where VORP
        for all players is needed.

        Returns:
            A dictionary mapping each position ('QB', 'RB', 'WR', 'TE') to its
            calculated baseline score.
        """
        baselines: Dict[str, float] = {}
        # Ensure cache exists
        self._build_sorted_available_cache()
        for position in self.action_to_position.values():
            sorted_players = self._get_sorted_available_for_position(position)
            baselines[position] = self._get_dynamic_baseline_for_position(position, sorted_players)
        return baselines

    def _get_kth_best_available_player_by_pos(self, position: str, k: int) -> Optional[Player]:
        """
        Returns the k-th best available player with the highest projected points for a given position.
        Returns None if fewer than k players are available for that position.
        """
        # Use cached sorted list for efficiency
        sorted_players = self._get_sorted_available_for_position(position)
        if len(sorted_players) < k:
            return None
        return sorted_players[k-1] # k-1 because of 0-based indexing

    def _get_next_opponent_team_id(self) -> int:
        """
        Determines the ID of the team that will draft immediately after the agent.
        Returns the agent's ID if it's the last pick in the round (or draft is effectively over).
        """
        # If the draft order is exhausted, or it's the last pick, there's no "next opponent"
        if self.current_pick_idx + 1 >= len(self.draft_order):
            return self.agent_team_id # Or some other neutral ID, agent's is fine for default counts
        
        # Find the next pick in the sequence that is not the agent's pick
        next_pick_index = self.current_pick_idx + 1
        while next_pick_index < len(self.draft_order):
            next_team_id = self.draft_order[next_pick_index]
            if next_team_id != self.agent_team_id:
                return next_team_id
            next_pick_index += 1
        
        # If no opponent is found after the agent's current turn in the remaining draft order
        return self.agent_team_id # Fallback to agent's team id for count (implies no immediate opponent)

    def _get_opponent_roster_count(self, opponent_team_id: int, position: str) -> int:
        """
        Returns the current count of a specific position on the specified opponent's roster.
        Returns 0 if the opponent_team_id is invalid or not found.
        """
        if opponent_team_id in self.teams_rosters:
            return self.teams_rosters[opponent_team_id].get(position, 0)
        return 0 # No roster information for this opponent_team_id

    def _get_next_opponent_team_id_for(self, perspective_team_id: int) -> int:
        """
        Determines the ID of the team that will draft immediately after the given team perspective.
        Mirrors _get_next_opponent_team_id but parameterized by team id.
        """
        if self.current_pick_idx + 1 >= len(self.draft_order):
            return perspective_team_id

        next_pick_index = self.current_pick_idx + 1
        while next_pick_index < len(self.draft_order):
            next_team_id = self.draft_order[next_pick_index]
            if next_team_id != perspective_team_id:
                return next_team_id
            next_pick_index += 1
        return perspective_team_id

    def _get_bye_week_conflict_count_for_team(self, team_id: int, position: str) -> int:
        """Bye week conflict count for best available player vs specified team's roster."""
        best_player = self._get_best_available_player_by_pos(position)
        if not best_player or not best_player.bye_week or np.isnan(best_player.bye_week):
            return 0
        agent_roster = self.teams_rosters[team_id]['PLAYERS']
        return sum(1 for p in agent_roster if p.bye_week == best_player.bye_week)

    def _get_bye_week_vector_for_team(self, team_id: int) -> np.ndarray:
        """Returns a bye week vector (weeks 4-14) for the specified team."""
        bye_week_vector = np.zeros(11, dtype=np.float32)
        team_roster = self.teams_rosters[team_id]['PLAYERS']
        for player in team_roster:
            bye_week = player.bye_week
            if bye_week and not np.isnan(bye_week) and 4 <= bye_week <= 14:
                idx = int(bye_week) - 4
                bye_week_vector[idx] += 1
        return bye_week_vector

    def _calculate_imminent_threat_for_team(self, team_id: int, position: str) -> int:
        """
        Parameterized version of _calculate_imminent_threat that evaluates from a given team's perspective.
        """
        threat_count = 0
        try:
            next_pick_idx = self.draft_order.index(team_id, self.current_pick_idx + 1)
        except ValueError:
            next_pick_idx = len(self.draft_order)

        intervening_picks = self.draft_order[self.current_pick_idx + 1: next_pick_idx]
        opponent_teams_in_window = set(intervening_picks)

        needed_starters = self.config.draft.ROSTER_STRUCTURE.get(position, 0)
        if needed_starters == 0:
            return 0

        for opp_id in opponent_teams_in_window:
            roster_counts = self.teams_rosters[opp_id]
            if roster_counts.get(position, 0) < needed_starters:
                threat_count += 1
        return threat_count

    def _compute_global_state_features(self) -> Dict[str, float]:
        """
        Computes and returns state feature values that are global (independent of team perspective)
        for the features listed in config.training.ENABLED_STATE_FEATURES.
        Relies on the per-step sorted cache.
        """
        self._build_sorted_available_cache()

        def get_k(pos: str, k: int) -> float:
            p = self._get_kth_best_available_player_by_pos(pos, k)
            return p.projected_points if p else 0.0

        globals_map: Dict[str, float] = {}

        enabled = set(self.config.training.ENABLED_STATE_FEATURES)

        # Best available points
        if 'best_available_qb_points' in enabled: globals_map['best_available_qb_points'] = get_k('QB', 1)
        if 'best_available_rb_points' in enabled: globals_map['best_available_rb_points'] = get_k('RB', 1)
        if 'best_available_wr_points' in enabled: globals_map['best_available_wr_points'] = get_k('WR', 1)
        if 'best_available_te_points' in enabled: globals_map['best_available_te_points'] = get_k('TE', 1)

        # VORP
        if 'best_available_qb_vorp' in enabled: globals_map['best_available_qb_vorp'] = self._calculate_vorp('QB')
        if 'best_available_rb_vorp' in enabled: globals_map['best_available_rb_vorp'] = self._calculate_vorp('RB')
        if 'best_available_wr_vorp' in enabled: globals_map['best_available_wr_vorp'] = self._calculate_vorp('WR')
        if 'best_available_te_vorp' in enabled: globals_map['best_available_te_vorp'] = self._calculate_vorp('TE')

        # Availability flags (use cache)
        if 'qb_available_flag' in enabled:
            globals_map['qb_available_flag'] = 1.0 if self._get_sorted_available_for_position('QB') else 0.0
        if 'rb_available_flag' in enabled:
            globals_map['rb_available_flag'] = 1.0 if self._get_sorted_available_for_position('RB') else 0.0
        if 'wr_available_flag' in enabled:
            globals_map['wr_available_flag'] = 1.0 if self._get_sorted_available_for_position('WR') else 0.0
        if 'te_available_flag' in enabled:
            globals_map['te_available_flag'] = 1.0 if self._get_sorted_available_for_position('TE') else 0.0

        # Current pick number
        if 'current_pick_number' in enabled:
            globals_map['current_pick_number'] = float(self.current_pick_number)

        # Second best available points
        if 'second_best_available_qb_points' in enabled: globals_map['second_best_available_qb_points'] = get_k('QB', 2)
        if 'second_best_available_rb_points' in enabled: globals_map['second_best_available_rb_points'] = get_k('RB', 2)
        if 'second_best_available_wr_points' in enabled: globals_map['second_best_available_wr_points'] = get_k('WR', 2)
        if 'second_best_available_te_points' in enabled: globals_map['second_best_available_te_points'] = get_k('TE', 2)

        # Scarcity
        if 'qb_scarcity' in enabled: globals_map['qb_scarcity'] = self._calculate_scarcity('QB')
        if 'rb_scarcity' in enabled: globals_map['rb_scarcity'] = self._calculate_scarcity('RB')
        if 'wr_scarcity' in enabled: globals_map['wr_scarcity'] = self._calculate_scarcity('WR')
        if 'te_scarcity' in enabled: globals_map['te_scarcity'] = self._calculate_scarcity('TE')

        # Top-3 points
        for pos in ['QB', 'RB', 'WR', 'TE']:
            for k in [1, 2, 3]:
                key = f"top_3_{pos.lower()}_points_{k}"
                if key in enabled:
                    globals_map[key] = get_k(pos, k)

        return globals_map

    def _build_state_for_team_from_global(self, team_id: int, global_features: Dict[str, float]) -> np.ndarray:
        """
        Builds a full state vector for a given team_id by combining provided global features
        with efficiently computed team-specific features. Applies configured normalization.
        """
        state_values_map: Dict[str, float] = dict(global_features) if global_features else {}
        enabled = self.config.training.ENABLED_STATE_FEATURES

        # Team-specific simple counts
        roster_counts = self.teams_rosters[team_id]
        if 'current_roster_qb_count' in enabled: state_values_map['current_roster_qb_count'] = float(roster_counts['QB'])
        if 'current_roster_rb_count' in enabled: state_values_map['current_roster_rb_count'] = float(roster_counts['RB'])
        if 'current_roster_wr_count' in enabled: state_values_map['current_roster_wr_count'] = float(roster_counts['WR'])
        if 'current_roster_te_count' in enabled: state_values_map['current_roster_te_count'] = float(roster_counts['TE'])

        # Available roster slots
        if 'available_roster_slots_qb' in enabled:
            state_values_map['available_roster_slots_qb'] = float(self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'] - roster_counts['QB'])
        if 'available_roster_slots_rb' in enabled:
            state_values_map['available_roster_slots_rb'] = float(self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'] - roster_counts['RB'])
        if 'available_roster_slots_wr' in enabled:
            state_values_map['available_roster_slots_wr'] = float(self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'] - roster_counts['WR'])
        if 'available_roster_slots_te' in enabled:
            state_values_map['available_roster_slots_te'] = float(self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'] - roster_counts['TE'])
        if 'available_roster_slots_flex' in enabled:
            state_values_map['available_roster_slots_flex'] = float(self.config.draft.ROSTER_STRUCTURE['FLEX'] - roster_counts['FLEX'])

        # Agent start position from team's perspective
        if 'agent_start_position' in enabled:
            state_values_map['agent_start_position'] = float(team_id)

        # Next pick opponent counts (perspective-specific)
        next_opp_id = self._get_next_opponent_team_id_for(team_id)
        if 'next_pick_opponent_qb_count' in enabled:
            state_values_map['next_pick_opponent_qb_count'] = float(self._get_opponent_roster_count(next_opp_id, 'QB'))
        if 'next_pick_opponent_rb_count' in enabled:
            state_values_map['next_pick_opponent_rb_count'] = float(self._get_opponent_roster_count(next_opp_id, 'RB'))
        if 'next_pick_opponent_wr_count' in enabled:
            state_values_map['next_pick_opponent_wr_count'] = float(self._get_opponent_roster_count(next_opp_id, 'WR'))
        if 'next_pick_opponent_te_count' in enabled:
            state_values_map['next_pick_opponent_te_count'] = float(self._get_opponent_roster_count(next_opp_id, 'TE'))

        # Bye week conflicts for best player at pos
        if 'best_qb_bye_week_conflict' in enabled:
            state_values_map['best_qb_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'QB'))
        if 'best_rb_bye_week_conflict' in enabled:
            state_values_map['best_rb_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'RB'))
        if 'best_wr_bye_week_conflict' in enabled:
            state_values_map['best_wr_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'WR'))
        if 'best_te_bye_week_conflict' in enabled:
            state_values_map['best_te_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'TE'))

        # Imminent threat (perspective-specific)
        if 'qb_imminent_threat' in enabled:
            state_values_map['qb_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'QB'))
        if 'rb_imminent_threat' in enabled:
            state_values_map['rb_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'RB'))
        if 'wr_imminent_threat' in enabled:
            state_values_map['wr_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'WR'))
        if 'te_imminent_threat' in enabled:
            state_values_map['te_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'TE'))

        # Bye week vector
        if any(k.startswith('bye_week_') for k in enabled):
            bye_vec = self._get_bye_week_vector_for_team(team_id)
            mapping = {
                'bye_week_4_count': 0,
                'bye_week_5_count': 1,
                'bye_week_6_count': 2,
                'bye_week_7_count': 3,
                'bye_week_8_count': 4,
                'bye_week_9_count': 5,
                'bye_week_10_count': 6,
                'bye_week_11_count': 7,
                'bye_week_12_count': 8,
                'bye_week_13_count': 9,
                'bye_week_14_count': 10,
            }
            for key, idx in mapping.items():
                if key in enabled:
                    state_values_map[key] = float(bye_vec[idx])

        # Assemble in correct order
        raw_state = []
        for feature_name in enabled:
            raw_state.append(state_values_map.get(feature_name, 0.0))
        state_array = np.array(raw_state, dtype=np.float32)

        # Normalize
        if self.config.training.STATE_NORMALIZATION_METHOD == 'min_max':
            state_array = self._normalize_min_max(state_array)
        elif self.config.training.STATE_NORMALIZATION_METHOD == 'z_score':
            state_array = self._normalize_z_score(state_array)

        return state_array

    def _calculate_scarcity(self, position: str, k: int = 5) -> float:
        """
        Calculates the point difference between the best available player and the k-th
        best available player at a position to measure talent drop-off.
        """
        best_player = self._get_kth_best_available_player_by_pos(position, 1)
        if not best_player:
            return 0.0

        kth_player = self._get_kth_best_available_player_by_pos(position, k)

        if kth_player:
            # If the k-th player exists, calculate the drop-off directly.
            return best_player.projected_points - kth_player.projected_points
        else:
            # Edge case: Fewer than k players are available.
            # Use cached sorted list to determine the last one.
            eligible_players = self._get_sorted_available_for_position(position)
            if len(eligible_players) > 1:
                last_player = eligible_players[-1]
                return best_player.projected_points - last_player.projected_points
            else:
                return 0.0

    def _calculate_imminent_threat(self, position: str) -> int:
        """
        Calculates how many teams picking between the agent's current and next pick
        still need to fill a starting spot at the given position.
        """
        threat_count = 0
        try:
            # Find the index of the agent's next pick
            next_agent_pick_idx = self.draft_order.index(self.agent_team_id, self.current_pick_idx + 1)
        except ValueError:
            # Agent has no more picks, so the window is the rest of the draft.
            next_agent_pick_idx = len(self.draft_order)

        # Identify the team IDs picking between now and the agent's next turn
        intervening_picks = self.draft_order[self.current_pick_idx + 1 : next_agent_pick_idx]
        opponent_teams_in_window = set(intervening_picks)

        # Count how many of those opponents need a starter for the position
        needed_starters = self.config.draft.ROSTER_STRUCTURE.get(position, 0)
        if needed_starters == 0:
            return 0

        for team_id in opponent_teams_in_window:
            roster_counts = self.teams_rosters[team_id]
            if roster_counts.get(position, 0) < needed_starters:
                threat_count += 1

        return threat_count

    def _get_agent_bye_week_vector(self) -> np.ndarray:
        """
        Returns a fixed-length numpy array representing the count of players
        on the agent's roster for each bye week (Weeks 4-14).
        """
        # Vector for Weeks 4 through 14 (11 weeks total)
        bye_week_vector = np.zeros(11, dtype=np.float32)
        agent_roster = self.teams_rosters[self.agent_team_id]['PLAYERS']

        for player in agent_roster:
            bye_week = player.bye_week
            if bye_week and not np.isnan(bye_week) and 4 <= bye_week <= 14:
                # Subtract 4 to map week 4 to index 0, week 5 to index 1, etc.
                vector_index = int(bye_week) - 4
                bye_week_vector[vector_index] += 1

        return bye_week_vector

    def _get_stack_count_for_team(self, team_id: int) -> int:
        """
        Return QB–WR/TE stack count for a team's current roster.

        Parameters
        ----------
        team_id : int
            Team whose roster is evaluated.

        Returns
        -------
        int
            Number of valid stacks on that roster.
        """
        roster = self.teams_rosters[team_id]['PLAYERS']
        return calculate_stack_count(roster)

    def _get_current_stack_count(self) -> int:
        """
        Returns the count of existing QB-WR/TE stacks currently on the agent's roster.

        Returns
        -------
        int
            Number of valid stacks on the current roster
        """
        return self._get_stack_count_for_team(self.agent_team_id)

    def _get_stack_target_available_flag_for_team(self, team_id: int) -> int:
        """
        Whether a stacking target exists in the pool for a given team's roster.

        Parameters
        ----------
        team_id : int
            Team whose rostered QBs define stack eligibility.

        Returns
        -------
        int
            1 if a valid stacking target is available, 0 otherwise.
        """
        agent_roster = self.teams_rosters[team_id]['PLAYERS']

        qb_teams = set()
        for player in agent_roster:
            if player.position == 'QB' and player.team:
                qb_teams.add(player.team)

        if not qb_teams:
            return 0

        for player_id in self.available_players_ids:
            player = self.player_map[player_id]
            if (
                player.position in ['WR', 'TE']
                and player.team
                and player.team in qb_teams
            ):
                return 1

        return 0

    def _get_stack_target_available_flag(self) -> int:
        """
        Returns a binary flag indicating whether a valid stacking target
        (a WR or TE from the same NFL team as a QB currently on the agent's roster)
        is available in the remaining draft pool.

        Returns
        -------
        int
            1 if a valid stacking target is available, 0 otherwise
        """
        return self._get_stack_target_available_flag_for_team(self.agent_team_id)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        all_player_ids = {p.player_id for p in self.all_players_data}
        draft_order = self._generate_snake_draft_order(self.config.draft.NUM_TEAMS, self.total_roster_size_per_team)
        if getattr(self.config, "RANDOMIZE_AGENT_START_POSITION", False) and self.training:
            import random as _rnd
            agent_team_id = _rnd.randint(1, self.config.draft.NUM_TEAMS)
        else:
            agent_team_id = self.config.draft.AGENT_START_POSITION

        self._state.reset(all_player_ids, draft_order, agent_team_id)
        self._invalidate_sorted_available_cache()

        if self.training:
            while self._state.get_current_pick_idx() < len(self._state.get_draft_order()) and \
                  self._state.get_draft_order()[self._state.get_current_pick_idx()] != self._state.get_agent_team_id():

                current_sim_team_id = self._state.get_draft_order()[self._state.get_current_pick_idx()]
                global_features = self._compute_global_state_features()
                sim_drafted_player = self._simulate_competing_pick(current_sim_team_id, global_features)

                if sim_drafted_player:
                    self._state.add_player_to_roster(current_sim_team_id, sim_drafted_player)
                    self._invalidate_sorted_available_cache()
                    self._state.append_draft_history({
                        "player_id": sim_drafted_player.player_id,
                        "team_id": current_sim_team_id,
                        "is_manual_pick": False,
                        "previous_pick_idx": self._state.get_current_pick_idx(),
                        "previous_pick_number": self._state.get_current_pick_number(),
                        "previous_overridden_team_id": None,
                        "was_override": False,
                    })
                else:
                    if not self._state.get_available_player_ids():
                        print(f"[{self._state.get_current_pick_number()}] No players left in pool. Ending draft early during reset advance.")
                        break
                    print(f"[{self._state.get_current_pick_number()}] Warning: Competing team {current_sim_team_id} could not make a valid pick.")

                self._state.advance_pick()

        observation = self._get_state()
        info = self._get_info() 
        
        # If the draft ended before the agent even got its first pick (e.g., very late draft position in a short draft)
        if self.current_pick_idx >= len(self.draft_order) and len(self.teams_rosters[self.agent_team_id]['PLAYERS']) == 0:
            info['episode_ended_before_agent_first_pick'] = True

        # Add the action mask to info
        info['action_mask'] = self.get_action_mask()

        # Invalidate per-step cache at the very end to ensure freshness for next call
        self._invalidate_sorted_available_cache()
        return observation, info

    def step(self, action: int):
        current_team_id = self.draft_order[self.current_pick_idx]
        assert current_team_id == self.agent_team_id, f"Assertion failed: It's not the agent's turn! Expected {self.agent_team_id}, Got {current_team_id} at pick {self.current_pick_number}"

        selected_position = self.action_to_position[action]
        reward = 0
        done = False
        info = {}

        # --- 1. Agent's Turn ---
        # Compute starter points BEFORE pick (for shaping)
        pre_starter_points = 0.0
        if self.config.reward.ENABLE_PICK_SHAPING_REWARD and self.config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
            from draft_buddy.utils.roster_utils import calculate_roster_scores
            curr_scores = calculate_roster_scores(
                self.teams_rosters[self.agent_team_id]['PLAYERS'],
                self.config.draft.ROSTER_STRUCTURE,
                self.config.draft.BENCH_MAXES
            )
            pre_starter_points = curr_scores['starters_total_points']

        is_valid_pick, drafted_player = self._try_select_player_for_team(
            self.agent_team_id, selected_position, self.available_players_ids
        )

        if not is_valid_pick:
            # If invalid action, episode terminates.
            # Apply penalty only if ENABLE_INVALID_ACTION_PENALTIES is True.
            if self.config.reward.ENABLE_INVALID_ACTION_PENALTIES:
                penalty_key = f'roster_full_{selected_position}'
                if not self._get_best_available_player_by_pos(selected_position):
                    penalty_key = 'no_players_available'
                reward += self.config.reward.INVALID_ACTION_PENALTIES.get(penalty_key, self.config.reward.INVALID_ACTION_PENALTIES['default_invalid'])
            
            done = True # Invalid action still ends the episode for safety/clarity
            info['invalid_action'] = True
            info['reason'] = f"Invalid pick: {selected_position} - {'Penalty applied' if self.config.reward.ENABLE_INVALID_ACTION_PENALTIES else 'No penalty'}"
        else:
            self._state.add_player_to_roster(self.agent_team_id, drafted_player)
            info["drafted_player"] = drafted_player.name
            info['drafted_position'] = drafted_player.position
            info["drafted_points"] = drafted_player.projected_points
            info["drafted_adp"] = drafted_player.adp

            self._invalidate_sorted_available_cache()

            # Delegate Step Reward Calculation
            from draft_buddy.utils.reward_calculator import RewardCalculator
            step_reward, step_info = RewardCalculator.calculate_step_reward(
                self.config, self, drafted_player, pre_starter_points
            )
            reward += step_reward
            info.update(step_info)

        self._state.advance_pick()

        # --- 2. Simulate Competing Teams' Turns ---
        if not done: # Only simulate if agent's pick was valid
            while self.current_pick_idx < len(self.draft_order) and \
                  self.draft_order[self.current_pick_idx] != self.agent_team_id:

                current_sim_team_id = self.draft_order[self.current_pick_idx]
                # Build global features for this opponent pick using current availability
                global_features = self._compute_global_state_features()
                sim_drafted_player = self._simulate_competing_pick(current_sim_team_id, global_features)

                if sim_drafted_player:
                    self._state.add_player_to_roster(current_sim_team_id, sim_drafted_player)
                    self._invalidate_sorted_available_cache()
                else:
                    print(f"[{self.current_pick_number}] Warning: Competing team {current_sim_team_id} could not make a pick. Ending round early.")
                    done = True 
                    info['draft_ended_prematurely_sim_team'] = True
                    break

                self._state.advance_pick()

        # --- 3. Check for Done Condition ---
        if len(self.teams_rosters[self.agent_team_id]['PLAYERS']) >= self.total_roster_size_per_team:
            done = True 
            info['draft_complete'] = True 
        elif self.current_pick_idx >= len(self.draft_order):
            done = True 
            info['draft_ended_prematurely'] = True 

        # --- 4. Final Reward Calculation if Done ---
        if done:
            from draft_buddy.utils.reward_calculator import RewardCalculator
            final_reward, final_info = RewardCalculator.calculate_final_reward(
                self.config, self, self.matchups_df
            )
            reward += final_reward
            info.update(final_info)
            info['final_reward_total'] = reward # The total reward given to the agent

        # --- 5. Get Next State and Action Mask ---
        observation = self._get_state()
        info['action_mask'] = self.get_action_mask()

        # Invalidate per-step cache at the very end to ensure freshness for next call
        self._invalidate_sorted_available_cache()
        return observation, reward, done, False, info


    # -------------------- Season Reward Helpers --------------------
    def _categorize_roster_by_slots(self, team_roster: List[Player], roster_structure: Dict, bench_maxes: Dict) -> Tuple[Dict[str, List[Player]], List[Player], List[Player]]:
        """
        Categorizes players into starters, bench, and flex players based on roster structure
        and projected points, ensuring no player is double-counted.
        """
        from draft_buddy.utils.roster_utils import categorize_roster_by_slots
        return categorize_roster_by_slots(team_roster, roster_structure, bench_maxes)

    def _generate_snake_draft_order(self, num_teams, total_picks_per_team):
        """Generates a snake draft order for all picks."""
        draft_order = []
        num_rounds = math.ceil(total_picks_per_team) # Each team gets this many picks
        
        # Ensure sufficient players for the draft if possible.
        if num_rounds * num_teams > len(self.all_players_data):
            num_rounds = math.floor(len(self.all_players_data) / num_teams)
            print(f"Warning: Not enough players for full draft. Reducing to {num_rounds} rounds ({num_rounds * num_teams} total picks).")
            # Ensure at least one round if possible
            if num_rounds == 0 and len(self.all_players_data) > 0:
                num_rounds = 1
                print("Adjusted to at least 1 round as players are available.")
            elif num_rounds == 0:
                 print("No players available to draft. Draft will be empty.")
                 return [] # Return empty draft order if no players

        for round_num in range(num_rounds):
            if round_num % 2 == 0:
                # Odd rounds (1, 3, 5...) - ascending order
                for team_id in range(1, num_teams + 1):
                    draft_order.append(team_id)
            else:
                # Even rounds (2, 4, 6...) - descending order
                for team_id in range(num_teams, 0, -1):
                    draft_order.append(team_id)
        return draft_order

    def _get_info(self):
        """Returns a dictionary with useful information about the current state."""
        team_on_clock = self._overridden_team_id if self._overridden_team_id is not None else \
                        (self.draft_order[self.current_pick_idx] if self.current_pick_idx < len(self.draft_order) else None)
        return {
            'current_pick_number': self.current_pick_number,
            'current_team_picking': team_on_clock,
            'agent_roster_size': len(self.teams_rosters[self.config.draft.AGENT_START_POSITION]['PLAYERS']),
            'available_players_count': len(self.available_players_ids),
            'manual_draft_teams': list(self.manual_draft_teams)
        }

    def _get_state_for_team(self, team_id: int) -> np.ndarray:
        """
        Normalized state vector from the perspective of a specific team (roster, threats, etc.).

        Parameters
        ----------
        team_id : int
            Team whose features populate the observation.

        Returns
        -------
        np.ndarray
            Normalized state vector.
        """
        state_values_map = self._compute_global_state_features()
        full_state_map = self._build_state_map_for_team(team_id, state_values_map)
        return self._state_normalizer.normalize(full_state_map)

    def _get_state(self) -> np.ndarray:
        """
        Constructs and returns the normalized state vector for the current team on clock.

        Returns
        -------
        np.ndarray
            Normalized state vector.
        """
        if self.current_pick_idx < len(self.draft_order):
            perspective_team_id = self.draft_order[self.current_pick_idx]
        else:
            perspective_team_id = self.agent_team_id
        return self._get_state_for_team(perspective_team_id)

    def _build_state_map_for_team(self, team_id: int, global_features: Dict[str, float]) -> Dict[str, float]:
        """
        Combines global features with team-specific features into a single map.
        """
        state_values_map = global_features.copy()
        enabled = set(self.config.training.ENABLED_STATE_FEATURES)

        # Team-specific simple counts
        roster_counts = self.teams_rosters[team_id]
        if 'current_roster_qb_count' in enabled: state_values_map['current_roster_qb_count'] = float(roster_counts['QB'])
        if 'current_roster_rb_count' in enabled: state_values_map['current_roster_rb_count'] = float(roster_counts['RB'])
        if 'current_roster_wr_count' in enabled: state_values_map['current_roster_wr_count'] = float(roster_counts['WR'])
        if 'current_roster_te_count' in enabled: state_values_map['current_roster_te_count'] = float(roster_counts['TE'])

        # Available roster slots
        if 'available_roster_slots_qb' in enabled:
            state_values_map['available_roster_slots_qb'] = float(self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'] - roster_counts['QB'])
        if 'available_roster_slots_rb' in enabled:
            state_values_map['available_roster_slots_rb'] = float(self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'] - roster_counts['RB'])
        if 'available_roster_slots_wr' in enabled:
            state_values_map['available_roster_slots_wr'] = float(self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'] - roster_counts['WR'])
        if 'available_roster_slots_te' in enabled:
            state_values_map['available_roster_slots_te'] = float(self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'] - roster_counts['TE'])
        if 'available_roster_slots_flex' in enabled:
            state_values_map['available_roster_slots_flex'] = float(self.config.draft.ROSTER_STRUCTURE['FLEX'] - roster_counts['FLEX'])

        if 'current_pick_number' in enabled: state_values_map['current_pick_number'] = float(self.current_pick_number)
        if 'agent_start_position' in enabled: state_values_map['agent_start_position'] = float(team_id)

        # Imminent threats from perspectives of next teams
        next_opp_id = self._get_next_opponent_team_id_for(team_id)
        if 'next_pick_opponent_qb_count' in enabled: state_values_map['next_pick_opponent_qb_count'] = float(self._get_opponent_roster_count(next_opp_id, 'QB'))
        if 'next_pick_opponent_rb_count' in enabled: state_values_map['next_pick_opponent_rb_count'] = float(self._get_opponent_roster_count(next_opp_id, 'RB'))
        if 'next_pick_opponent_wr_count' in enabled: state_values_map['next_pick_opponent_wr_count'] = float(self._get_opponent_roster_count(next_opp_id, 'WR'))
        if 'next_pick_opponent_te_count' in enabled: state_values_map['next_pick_opponent_te_count'] = float(self._get_opponent_roster_count(next_opp_id, 'TE'))

        # Team specific conflicts
        if 'best_qb_bye_week_conflict' in enabled: state_values_map['best_qb_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'QB'))
        if 'best_rb_bye_week_conflict' in enabled: state_values_map['best_rb_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'RB'))
        if 'best_wr_bye_week_conflict' in enabled: state_values_map['best_wr_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'WR'))
        if 'best_te_bye_week_conflict' in enabled: state_values_map['best_te_bye_week_conflict'] = float(self._get_bye_week_conflict_count_for_team(team_id, 'TE'))

        # TIER 2 features
        if 'qb_imminent_threat' in enabled: state_values_map['qb_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'QB'))
        if 'rb_imminent_threat' in enabled: state_values_map['rb_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'RB'))
        if 'wr_imminent_threat' in enabled: state_values_map['wr_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'WR'))
        if 'te_imminent_threat' in enabled: state_values_map['te_imminent_threat'] = float(self._calculate_imminent_threat_for_team(team_id, 'TE'))

        # Bye week vector for team
        team_bye_vec = self._get_bye_week_vector_for_team(team_id)
        for i in range(4, 15):
            fname = f"bye_week_{i}_count"
            if fname in enabled: state_values_map[fname] = float(team_bye_vec[i-4])

        # Stacking features (perspective of team_id)
        if 'current_stack_count' in enabled:
            state_values_map['current_stack_count'] = float(self._get_stack_count_for_team(team_id))
        if 'stack_target_available_flag' in enabled:
            state_values_map['stack_target_available_flag'] = float(
                self._get_stack_target_available_flag_for_team(team_id)
            )

        return state_values_map

    # Original _get_best_available_player_by_pos is now generalized by _get_kth_best_available_player_by_pos
    def _get_best_available_player_by_pos(self, position: str) -> Optional[Player]:
        """Returns the available player with the highest projected points for a given position."""
        return self._get_kth_best_available_player_by_pos(position, 1)

    def _can_team_draft_position_manual(self, team_id: int, position: str) -> bool:
        """Validates if a manual user can draft a position. Delegates to DraftRules."""
        return self._rules.can_draft_manual(self._state, team_id, position, self.player_map)

    def _can_team_draft_position_simulated(self, team_id: int, position: str) -> bool:
        """Validates if an automated agent can draft a position. Delegates to DraftRules."""
        return self._rules.can_draft_simulated(self._state, team_id, position, self.player_map)

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array representing the valid actions for the agent in the current state.
        A value of True means the action is valid, False means it's invalid.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        for action, position_choice in self.action_to_position.items():
            if self._can_team_draft_position_simulated(self.agent_team_id, position_choice):
                mask[action] = True
        return mask

    def _try_select_player_for_team(self, team_id: int, position_choice: str, available_players_ids: set) -> Tuple[bool, Optional[Player]]:
        """
        Attempts to find and 'draft' the best player for a given position and team.
        Returns (True, player) if successful, (False, None) otherwise.
        Handles roster limits and player availability.
        """
        # This function is called after action mask might have already validated.
        # However, it still needs to perform its own checks in case it's called internally
        # or if action masking is disabled.
        
        # 1. Check if any players are available for the chosen position
        available_players_for_pos = [
            self.player_map[pid] for pid in available_players_ids
            if self.player_map[pid].position == position_choice
        ]
        if not available_players_for_pos:
            return False, None # No players of this position available at all

        # 2. Check roster limits for the chosen position for this specific team
        if not self._can_team_draft_position_simulated(team_id, position_choice):
            return False, None

        # Find the best available player for the chosen position (highest projected points)
        best_player = None
        max_points = -1
        for p in available_players_for_pos:
            if p.projected_points > max_points:
                max_points = p.projected_points
                best_player = p

        if best_player:
            return True, best_player
        else:
            return False, None  # Should not happen if available_players_for_pos is not empty and rules allow the pick.

    def _simulate_competing_pick(self, team_id: int, global_features: Optional[Dict[str, float]] = None) -> Optional[Player]:
        """Delegates to the opponent's strategy to select a player."""
        strategy = self._opponent_strategies.get(team_id)
        if not strategy:
            strategy = create_opponent_strategy(
                logic=self.config.opponent.DEFAULT_OPPONENT_STRATEGY["logic"],
                config=self.config.opponent.DEFAULT_OPPONENT_STRATEGY,
                opponent_models=self.opponent_models,
                team_id=team_id,
                action_to_position=self.action_to_position,
            )
            self._opponent_strategies[team_id] = strategy

        def can_draft(team_id: int, position: str, is_manual: bool = False) -> bool:
            if is_manual:
                return self._can_team_draft_position_manual(team_id, position)
            return self._can_team_draft_position_simulated(team_id, position)

        def try_select(team_id: int, position: str, available_ids: set):
            return self._try_select_player_for_team(team_id, position, available_ids)

        def build_state(tid: int) -> np.ndarray:
            if global_features is None:
                gf = self._compute_global_state_features()
            else:
                gf = global_features
            return self._build_state_for_team_from_global(tid, gf)

        def get_mask(tid: int) -> np.ndarray:
            orig = self.agent_team_id
            self.agent_team_id = tid
            mask = self.get_action_mask()
            self.agent_team_id = orig
            return mask

        chosen = strategy.execute_pick(
            team_id=team_id,
            available_player_ids=self.available_players_ids,
            player_map=self.player_map,
            roster_counts=self.teams_rosters[team_id],
            roster_structure=self.config.draft.ROSTER_STRUCTURE,
            bench_maxes=self.config.draft.BENCH_MAXES,
            can_draft_position_fn=can_draft,
            try_select_player_fn=try_select,
            build_state_fn=build_state,
            get_action_mask_fn=get_mask,
        )

        if chosen is None:
            eligible = [
                self.player_map[pid] for pid in self.available_players_ids
                if self.player_map.get(pid) and self.player_map[pid].position in {"QB", "RB", "WR", "TE"}
                and self._can_team_draft_position_simulated(team_id, self.player_map[pid].position)
            ]
            if eligible:
                chosen = random.choice(eligible)
        return chosen

    def draft_player(self, player_id: int):
        """
        Manually drafts a specific player for the current team.
        This method does NOT advance the draft or simulate opponent picks.
        """
        # Allow manual picks beyond the scheduled draft as long as an override team is set
        if self.current_pick_idx >= len(self.draft_order) and self._overridden_team_id is None:
            raise ValueError("The draft has already concluded. No more picks can be made.")

        # If a team was overridden, use that ID, otherwise use the team from the draft order.
        if self._overridden_team_id is not None:
            current_team_id = self._overridden_team_id
        else:
            current_team_id = self.draft_order[self.current_pick_idx]
        
        # Find the player object
        drafted_player = self.player_map.get(player_id)
        if not drafted_player or drafted_player.player_id not in self.available_players_ids:
            self._state.set_overridden_team_id(None)
            raise ValueError(f"Player with ID {player_id} is not available to be drafted.")

        if not self._can_team_draft_position_manual(current_team_id, drafted_player.position):
            self._state.set_overridden_team_id(None)
            raise ValueError(
                f"Team {current_team_id} cannot draft a {drafted_player.position}. "
                "The roster is full for that position or the bench is at capacity."
            )

        # Store the state before making the pick for the undo history
        was_override = (self._overridden_team_id is not None)
        
        self._state.append_draft_history({
            'player_id': drafted_player.player_id,
            'team_id': current_team_id,
            'is_manual_pick': True,
            'previous_pick_idx': self.current_pick_idx,
            'previous_pick_number': self.current_pick_number,
            'previous_overridden_team_id': self._overridden_team_id,
            "was_override": was_override,
        })

        self._state.add_player_to_roster(current_team_id, drafted_player)
        self._invalidate_sorted_available_cache()

        self._state.advance_pick()
        self._state.set_overridden_team_id(None)
        # Invalidate cache after manual draft
        self._invalidate_sorted_available_cache()


    def undo_last_pick(self):
        """Reverts the last pick made in the draft."""
        last_pick_info = self._state.pop_draft_history()
        if not last_pick_info:
            raise ValueError("No picks to undo.")

        player_id = last_pick_info["player_id"]
        team_id = last_pick_info["team_id"]
        player = self.player_map[player_id]

        self._state.remove_player_from_roster(team_id, player)
        self._state.set_current_pick_idx(last_pick_info["previous_pick_idx"])
        self._state.set_current_pick_number(last_pick_info["previous_pick_number"])
        self._state.set_overridden_team_id(last_pick_info.get("previous_overridden_team_id"))

        self._invalidate_sorted_available_cache()

        team_on_clock_display = "N/A"
        if self._state.get_current_pick_idx() < len(self._state.get_draft_order()):
            team_on_clock_display = self._state.get_draft_order()[self._state.get_current_pick_idx()]

        print(f"Undo successful. Current pick: {self._state.get_current_pick_number()}, Team on clock: {team_on_clock_display}")

    def set_current_team_picking(self, team_id: int):
        """
        Sets the current team on the clock to the specified team_id for the *next* pick only.
        This is primarily for manual override from the frontend.
        """
        if team_id not in range(1, self.config.draft.NUM_TEAMS + 1):
            raise ValueError(f"Invalid team ID: {team_id}. Must be between 1 and {self.config.draft.NUM_TEAMS}.")
        
        # For UI flexibility, allow setting an override even if no scheduled picks remain.
        # Manual drafting can continue post-schedule to fill rosters.

        self._state.set_overridden_team_id(team_id)
        print(f"Next pick will be overridden for Team {team_id}.")

    def simulate_single_pick(self):
        """
        Simulates a single pick by the current team on the clock.
        This is used for 'Manual Step' mode.
        """
        if self.current_pick_idx >= len(self.draft_order):
            raise ValueError("The draft has already concluded. No more picks can be made.")

        current_sim_team_id = self._state.get_overridden_team_id() or self._state.get_draft_order()[self._state.get_current_pick_idx()]
        self._state.set_overridden_team_id(None)

        if current_sim_team_id in self.manual_draft_teams:
            raise ValueError("It is a manual team's turn. Cannot simulate pick.")

        # Use precomputed global features for lightweight opponent state
        global_features = self._compute_global_state_features()
        sim_drafted_player = self._simulate_competing_pick(current_sim_team_id, global_features)

        if sim_drafted_player:
            self._state.add_player_to_roster(current_sim_team_id, sim_drafted_player)
            self._invalidate_sorted_available_cache()
            self._state.append_draft_history({
                "player_id": sim_drafted_player.player_id,
                "team_id": current_sim_team_id,
                "is_manual_pick": False,
                "previous_pick_idx": self._state.get_current_pick_idx(),
                "previous_pick_number": self._state.get_current_pick_number(),
                "previous_overridden_team_id": None,
                "was_override": False,
            })
        else:
            # If the team cannot make a valid pick under constraints, only skip if team is full.
            roster_data = self._state.get_rosters()[current_sim_team_id]
            if len(roster_data["PLAYERS"]) >= self.total_roster_size_per_team:
                print(f"Competing team {current_sim_team_id} cannot pick and roster is full. Skipping.")
                self._state.advance_pick()
                return

            if not self._state.get_available_player_ids():
                print("No players left to force-pick. Skipping.")
                self._state.advance_pick()
                return
            
            all_available_players = [self.player_map[pid] for pid in self._state.get_available_player_ids()]
            # Prefer finite ADP; if equal/unavailable, fall back to higher projected points
            def force_key(p):
                return (np.isinf(p.adp), p.adp if np.isfinite(p.adp) else float('inf'), -p.projected_points)
            forced_player = min(all_available_players, key=force_key)

            self._state.add_player_to_roster(current_sim_team_id, forced_player)
            self._invalidate_sorted_available_cache()
            self._state.append_draft_history({
                "player_id": forced_player.player_id,
                "team_id": current_sim_team_id,
                "is_manual_pick": False,
                "previous_pick_idx": self._state.get_current_pick_idx(),
                "previous_pick_number": self._state.get_current_pick_number(),
                "previous_overridden_team_id": None,
                "was_override": False,
            })

        self._state.advance_pick()
        print(f"Simulated pick for Team {current_sim_team_id}. Current pick: {self._state.get_current_pick_number()}")

    def get_ai_suggestion(self):
        """Gets the AI's suggested action probabilities for the current state."""
        if self.current_pick_idx >= len(self.draft_order):
            return {"error": "Draft is over."}

        if not self.agent_model:
            return {"error": "AI model not loaded."}

        # The team on the clock is the one we want a suggestion for.
        current_team_on_clock = self.draft_order[self.current_pick_idx]

        # Temporarily set the environment's perspective to the team on the clock
        original_agent_team_id = self.agent_team_id
        self.agent_team_id = current_team_on_clock
        
        state = self._get_state()
        action_mask = self.get_action_mask()

        # Restore the original agent team ID perspective
        self.agent_team_id = original_agent_team_id

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs_tensor = self.agent_model.get_action_probabilities(state_tensor, action_mask=action_mask)
            action_probs = action_probs_tensor.squeeze().tolist() # Convert tensor to list

        # Create a dictionary mapping position to probability
        suggestion_probs = {
            self.action_to_position[i]: prob 
            for i, prob in enumerate(action_probs)
        }

        return suggestion_probs

    def get_ai_suggestion_for_team(self, team_id: int, ignore_player_ids: Optional[List[int]] = None):
        """Gets AI suggested action probabilities for a specific team's perspective.

        If ignore_player_ids is provided, those players will be treated as unavailable
        for the purpose of computing this suggestion only (no state mutation).
        """
        if not (1 <= team_id <= self.config.draft.NUM_TEAMS):
            return {"error": f"Invalid team id {team_id}"}
        if not self.agent_model:
            return {"error": "AI model not loaded."}

        original_agent_team_id = self.agent_team_id
        original_available_ids = None
        self.agent_team_id = team_id
        try:
            if ignore_player_ids:
                ignore_set = set(pid for pid in ignore_player_ids if pid in self._state.get_available_player_ids())
                if ignore_set:
                    original_available_ids = set(self._state.get_available_player_ids())
                    self._state.replace_available_player_ids(original_available_ids - ignore_set)
                    self._invalidate_sorted_available_cache()

            state = self._get_state_for_team(team_id)
            action_mask = self.get_action_mask()
        finally:
            self.agent_team_id = original_agent_team_id
            if original_available_ids is not None:
                self._state.replace_available_player_ids(original_available_ids)
                self._invalidate_sorted_available_cache()

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs_tensor = self.agent_model.get_action_probabilities(state_tensor, action_mask=action_mask)
            action_probs = action_probs_tensor.squeeze().tolist()

        return {self.action_to_position[i]: prob for i, prob in enumerate(action_probs)}

    def get_ai_suggestions_all(self):
        """Returns AI action probabilities for all teams 1..NUM_TEAMS."""
        if not self.agent_model:
            return {"error": "AI model not loaded."}
        suggestions: Dict[int, Dict[str, float]] = {}
        for team_id in range(1, self.config.draft.NUM_TEAMS + 1):
            suggestions[team_id] = self.get_ai_suggestion_for_team(team_id)
        return suggestions

    def get_draft_summary(self):
        """Returns a summary of the draft."""
        summary = {
            'total_picks': self.current_pick_number - 1,
            'picks_by_position': {
                'QB': 0,
                'RB': 0,
                'WR': 0,
                'TE': 0
            }
        }

        for pick in self._draft_history:
            player = self.player_map.get(pick['player_id'])
            if player:
                summary['picks_by_position'][player.position] += 1

        return summary

    def render(self):
        """
        Renders the current state of the environment.
        For fantasy football, this could print current draft board, agent's roster, etc.
        """
        current_team_picking_id = self.draft_order[self.current_pick_idx] if self.current_pick_idx < len(self.draft_order) else 'N/A'
        print(f"\n--- Current Pick: {self.current_pick_number} (Team {current_team_picking_id}) ---")
        
        # Display agent's roster
        agent_roster_data = self.teams_rosters[self.agent_team_id]
        print(f"Agent's Roster (Team {self.agent_team_id}):")
        roster_summary = []
        for pos_type, count_needed in self.config.draft.ROSTER_STRUCTURE.items():
            if pos_type == 'FLEX':
                roster_summary.append(f"{pos_type}: {agent_roster_data['FLEX']}/{count_needed}")
            else:
                bench_max = self.config.draft.BENCH_MAXES.get(pos_type, 0)
                current_total = agent_roster_data[pos_type]
                roster_summary.append(f"{pos_type}: {current_total}/{count_needed + bench_max}")
        print("  " + " | ".join(roster_summary))

        # Display top available players by position (optional)
        print("\nTop Available Players (by Projected Points):")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            best_player = self._get_best_available_player_by_pos(pos)
            if best_player:
                print(f"  {pos}: {best_player.name} ({best_player.projected_points:.1f} pts, ADP {best_player.adp:.1f})")
            else:
                print(f"  {pos}: None available")

    def close(self):
        """
        Clean up resources.
        """
        pass

if __name__ == '__main__':
    # Example of how to use the environment
    config = Config()
    env = FantasyFootballDraftEnv(config, training=True)
    
    # Reset the environment to get the initial state and info
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        
        # Get the action mask from the info dictionary
        action_mask = info['action_mask']
        
        # Choose a random valid action
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            print("No valid actions available. Ending episode.")
            break
        
        action = np.random.choice(valid_actions)
        
        print(f"\n>>> Agent (Team {env.agent_team_id}) is picking position: {env.action_to_position[action]}")
        
        # Take a step in the environment
        obs, reward, done, _, info = env.step(action)
        
        total_reward += reward
        
        print(f"Reward for this step: {reward:.2f}")
        if 'drafted_player' in info:
            print(f"Agent drafted: {info['drafted_player']} ({info['drafted_position']}) - {info['drafted_points']:.1f} pts")
        
        if done:
            print("\n--- Draft Finished ---")
            print(f"Total reward for the episode: {total_reward:.2f}")
            if 'final_score_agent' in info:
                print(f"Agent's final weighted score: {info['final_score_agent']:.2f}")
            if 'competitive_mode' in info:
                print(f"Competitive mode: {info['competitive_mode']}")
            if 'target_opponent_score' in info:
                print(f"Target opponent score: {info['target_opponent_score']:.2f}")
            if info.get('opponent_std_dev_applied'):
                print(f"Opponent score std dev penalty applied: {info['opponent_std_dev_penalty']:.2f}")
            
            # Print final rosters for all teams
            for team_id in range(1, config.draft.NUM_TEAMS + 1):
                roster_data = env.teams_rosters[team_id]
                team_score = sum(p.projected_points for p in roster_data['PLAYERS'])
                print(f"\nTeam {team_id} Roster (Total Raw Points: {team_score:.2f}):")
                for player in sorted(roster_data['PLAYERS'], key=lambda p: p.projected_points, reverse=True):
                    print(f"  - {player.name} ({player.position}, {player.projected_points:.1f} pts)")

    env.close()