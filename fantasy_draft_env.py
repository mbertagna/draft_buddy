import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import math
import random
import torch # Import torch for loading models
from typing import Optional, Dict, List, Tuple
import os

from config import Config
from data_utils import load_player_data, Player
from policy_network import PolicyNetwork # Import PolicyNetwork
import json

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
        self.manual_draft_teams = set(config.MANUAL_DRAFT_TEAMS) # Teams controlled by the user

        # --- Load Players Data ---
        self.all_players_data = load_player_data(config.PLAYER_DATA_CSV, config.MOCK_ADP_CONFIG)
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
            "available_roster_slots_qb": lambda: self.config.ROSTER_STRUCTURE['QB'] + self.config.BENCH_MAXES['QB'] - self.teams_rosters[self.agent_team_id]['QB'],
            "available_roster_slots_rb": lambda: self.config.ROSTER_STRUCTURE['RB'] + self.config.BENCH_MAXES['RB'] - self.teams_rosters[self.agent_team_id]['RB'],
            "available_roster_slots_wr": lambda: self.config.ROSTER_STRUCTURE['WR'] + self.config.BENCH_MAXES['WR'] - self.teams_rosters[self.agent_team_id]['WR'],
            "available_roster_slots_te": lambda: self.config.ROSTER_STRUCTURE['TE'] + self.config.BENCH_MAXES['TE'] - self.teams_rosters[self.agent_team_id]['TE'],
            "available_roster_slots_flex": lambda: self.config.ROSTER_STRUCTURE['FLEX'] - self.teams_rosters[self.agent_team_id]['FLEX'], # Simplified flex calculation
            # Draft Progression Context
            "current_pick_number": lambda: self.current_pick_number,
            "agent_start_position": lambda: self.config.AGENT_START_POSITION,
            # New features
            "second_best_available_qb_points": lambda: self._get_kth_best_available_player_by_pos('QB', 2).projected_points if self._get_kth_best_available_player_by_pos('QB', 2) else 0,
            "second_best_available_rb_points": lambda: self._get_kth_best_available_player_by_pos('RB', 2).projected_points if self._get_kth_best_available_player_by_pos('RB', 2) else 0,
            "second_best_available_wr_points": lambda: self._get_kth_best_available_player_by_pos('WR', 2).projected_points if self._get_kth_best_available_player_by_pos('WR', 2) else 0,
            "second_best_available_te_points": lambda: self._get_kth_best_available_player_by_pos('TE', 2).projected_points if self._get_kth_best_available_player_by_pos('TE', 2) else 0,
            "next_pick_opponent_qb_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'QB'),
            "next_pick_opponent_rb_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'RB'),
            "next_pick_opponent_wr_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'WR'),
            "next_pick_opponent_te_count": lambda: self._get_opponent_roster_count(self._get_next_opponent_team_id(), 'TE'),
        }
        self.observation_space_dim = len(config.ENABLED_STATE_FEATURES)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)

        # --- Internal Game State (will be reset for each episode) ---
        self.available_players_ids = set() # Set of player_ids currently available
        self.teams_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []}) # Tracks counts and actual players for each team
        self.draft_order = [] # List of team_id in draft order for current episode
        self.current_pick_idx = 0 # Index in self.draft_order
        self.current_pick_number = 0 # Global pick number (1-indexed)
        self.agent_team_id = self.config.AGENT_START_POSITION # Agent's unique ID (matches its start position)
        self._draft_history = [] # Stores (player_id, team_id, previous_state_info) for undo functionality
        self._overridden_team_id = None # Stores team ID for a single-pick override
        self._previous_pick_state = [] # Stores (current_pick_idx, current_pick_number, overridden_team_id) before each pick

        # Pre-calculate total roster size for done check
        self.total_roster_size_per_team = sum(self.config.ROSTER_STRUCTURE.values()) + sum(self.config.BENCH_MAXES.values())

        # New: Load opponent models
        self.opponent_models: Dict[int, PolicyNetwork] = {}
        self._load_opponent_models()
        
        # New: Load agent model for suggestions
        self.agent_model: Optional[PolicyNetwork] = None
        self._load_agent_model()

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
            with open(file_path, 'r') as f:
                state = json.load(f)
            self.available_players_ids = set(state.get('available_players_ids', []))
            
            # Reconstruct Player objects from dictionaries, ensuring team IDs are integers
            reconstructed_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []})
            if 'teams_rosters' in state:
                for team_id_str, roster_data in state['teams_rosters'].items():
                    try:
                        int_team_id = int(team_id_str)
                        reconstructed_rosters[int_team_id] = roster_data.copy()
                        reconstructed_rosters[int_team_id]['PLAYERS'] = [Player(**p) for p in roster_data.get('PLAYERS', [])]
                    except (ValueError, TypeError):
                        print(f"Warning: Could not parse team_id '{team_id_str}'. Skipping this team.")
                        continue
            self.teams_rosters = reconstructed_rosters

            self.draft_order = state.get('draft_order', [])
            self.current_pick_idx = state.get('current_pick_idx', 0)
            self.current_pick_number = state.get('current_pick_number', 1)
            self._draft_history = state.get('_draft_history', [])
            self._overridden_team_id = state.get('_overridden_team_id', None)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load draft state from {file_path} due to error: {e}. Starting fresh.")
            # If loading fails, reset the environment to a clean state
            self.reset()


    def _load_agent_model(self):
        """Loads the primary agent model for AI suggestions."""
        input_dim = len(self.config.ENABLED_STATE_FEATURES)
        output_dim = self.action_space.n
        model_path = self.config.MODEL_PATH_TO_LOAD

        if not model_path or not os.path.exists(model_path):
            print(f"Warning: Agent model for suggestions not found at {model_path}. AI suggestions will be disabled.")
            return

        try:
            self.agent_model = PolicyNetwork(input_dim, output_dim, self.config.HIDDEN_DIM)
            self.agent_model.load_state_dict(torch.load(model_path))
            self.agent_model.eval()
            print(f"Successfully loaded agent model for suggestions from {model_path}")
        except Exception as e:
            print(f"Error loading agent model for suggestions from {model_path}: {e}")
            self.agent_model = None


    def _load_opponent_models(self):
        """
        Loads trained PolicyNetwork models for opponents specified in config.
        Only loads models for teams explicitly marked with 'AGENT_MODEL' logic.
        """
        input_dim = len(self.config.ENABLED_STATE_FEATURES)
        output_dim = self.action_space.n

        for team_id in range(1, self.config.NUM_TEAMS + 1):
            if team_id == self.config.AGENT_START_POSITION:
                continue # Skip loading a model for the agent's own team

            opponent_strategy = self.config.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self.config.DEFAULT_OPPONENT_STRATEGY
            )

            if opponent_strategy['logic'] == 'AGENT_MODEL':
                model_path_key = opponent_strategy.get('model_path_key')
                if model_path_key and model_path_key in self.config.OPPONENT_MODEL_PATHS:
                    model_path = self.config.OPPONENT_MODEL_PATHS[model_path_key]
                    try:
                        model = PolicyNetwork(input_dim, output_dim, self.config.HIDDEN_DIM)
                        model.load_state_dict(torch.load(model_path))
                        model.eval() # Set to evaluation mode
                        self.opponent_models[team_id] = model
                        print(f"Loaded agent model for opponent Team {team_id} from {model_path}")
                    except FileNotFoundError:
                        print(f"Warning: Opponent model not found at {model_path} for Team {team_id}. Falling back to DEFAULT_OPPONENT_STRATEGY.")
                        # If model not found, revert this opponent to default heuristic
                        self.config.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.DEFAULT_OPPONENT_STRATEGY.copy()
                    except Exception as e:
                        print(f"Error loading opponent model for Team {team_id} from {model_path}: {e}. Falling back to DEFAULT_OPPONENT_STRATEGY.")
                        self.config.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.DEFAULT_OPPONENT_STRATEGY.copy()
                else:
                    print(f"Warning: 'model_path_key' missing or invalid for Team {team_id} configured as 'AGENT_MODEL'. Falling back to DEFAULT_OPPONENT_STRATEGY.")
                    self.config.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.DEFAULT_OPPONENT_STRATEGY.copy()

    def _get_kth_best_available_player_by_pos(self, position: str, k: int) -> Optional[Player]:
        """
        Returns the k-th best available player with the highest projected points for a given position.
        Returns None if fewer than k players are available for that position.
        """
        eligible_players = []
        for player_id in self.available_players_ids:
            player = self.player_map[player_id]
            if player.position == position:
                eligible_players.append(player)
        
        if len(eligible_players) < k:
            return None
            
        # Sort by projected points descending
        eligible_players.sort(key=lambda p: p.projected_points, reverse=True)
        return eligible_players[k-1] # k-1 because of 0-based indexing

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


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Reset all internal game state variables
        self.available_players_ids = {p.player_id for p in self.all_players_data}
        self.teams_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []})
        self.current_pick_idx = 0
        self.current_pick_number = 1 # Start at pick 1
        self._draft_history = [] # Clear draft history on reset
        self._overridden_team_id = None # Clear any override on reset

        # Generate draft order (snake draft)
        self.draft_order = self._generate_snake_draft_order(self.config.NUM_TEAMS, self.total_roster_size_per_team)

        # If in training mode, simulate opponent picks until it's the agent's turn
        if self.training:
            while self.current_pick_idx < len(self.draft_order) and \
                  self.draft_order[self.current_pick_idx] != self.agent_team_id:

                current_sim_team_id = self.draft_order[self.current_pick_idx]
                sim_drafted_player = self._simulate_competing_pick(current_sim_team_id)

                if sim_drafted_player:
                    self.teams_rosters[current_sim_team_id]['PLAYERS'].append(sim_drafted_player)
                    self.available_players_ids.remove(sim_drafted_player.player_id)
                    self._update_roster_counts(current_sim_team_id, sim_drafted_player)
                    # Record the simulated pick in history, marking it as not manual
                    self._draft_history.append({
                        'player_id': sim_drafted_player.player_id,
                        'team_id': current_sim_team_id,
                        'is_manual_pick': False,
                        'previous_pick_idx': self.current_pick_idx, # Store state before this sim pick
                        'previous_pick_number': self.current_pick_number,
                        'previous_overridden_team_id': None, # Sim picks don't have overrides
                        'was_override': False
                    })
                else:
                    if not self.available_players_ids:
                        print(f"[{self.current_pick_number}] No players left in pool. Ending draft early during reset advance.")
                        break 
                    print(f"[{self.current_pick_number}] Warning: Competing team {current_sim_team_id} could not make a valid pick.")


                self.current_pick_idx += 1
                self.current_pick_number += 1

        observation = self._get_state()
        info = self._get_info() 
        
        # If the draft ended before the agent even got its first pick (e.g., very late draft position in a short draft)
        if self.current_pick_idx >= len(self.draft_order) and len(self.teams_rosters[self.agent_team_id]['PLAYERS']) == 0:
            info['episode_ended_before_agent_first_pick'] = True

        # Add the action mask to info
        info['action_mask'] = self.get_action_mask()

        return observation, info

    def step(self, action: int):
        current_team_id = self.draft_order[self.current_pick_idx]
        assert current_team_id == self.agent_team_id, f"Assertion failed: It's not the agent's turn! Expected {self.agent_team_id}, Got {current_team_id} at pick {self.current_pick_number}"

        selected_position = self.action_to_position[action]
        reward = 0
        done = False
        info = {}

        # --- 1. Agent's Turn ---
        is_valid_pick, drafted_player = self._try_select_player_for_team(
            self.agent_team_id, selected_position, self.available_players_ids
        )

        if not is_valid_pick:
            # If invalid action, episode terminates.
            # Apply penalty only if ENABLE_INVALID_ACTION_PENALTIES is True.
            if self.config.ENABLE_INVALID_ACTION_PENALTIES:
                penalty_key = f'roster_full_{selected_position}'
                if not self._get_best_available_player_by_pos(selected_position):
                    penalty_key = 'no_players_available'
                reward += self.config.INVALID_ACTION_PENALTIES.get(penalty_key, self.config.INVALID_ACTION_PENALTIES['default_invalid'])
            
            done = True # Invalid action still ends the episode for safety/clarity
            info['invalid_action'] = True
            info['reason'] = f"Invalid pick: {selected_position} - {'Penalty applied' if self.config.ENABLE_INVALID_ACTION_PENALTIES else 'No penalty'}"
        else:
            # Valid pick
            self.teams_rosters[self.agent_team_id]['PLAYERS'].append(drafted_player)
            self.available_players_ids.remove(drafted_player.player_id)
            info['drafted_player'] = drafted_player.name
            info['drafted_position'] = drafted_player.position
            info['drafted_points'] = drafted_player.projected_points
            info['drafted_adp'] = drafted_player.adp

            # Update position counts
            self._update_roster_counts(self.agent_team_id, drafted_player)

            # Apply intermediate reward based on configuration
            if self.config.ENABLE_INTERMEDIATE_REWARD:
                if self.config.INTERMEDIATE_REWARD_MODE == 'STATIC':
                    reward += self.config.INTERMEDIATE_REWARD_VALUE
                elif self.config.INTERMEDIATE_REWARD_MODE == 'PROPORTIONAL':
                    reward += drafted_player.projected_points * self.config.PROPORTIONAL_REWARD_SCALING_FACTOR
                else:
                    print(f"Warning: Unknown INTERMEDIATE_REWARD_MODE: {self.config.INTERMEDIATE_REWARD_MODE}. No intermediate reward applied.")


        self.current_pick_idx += 1
        self.current_pick_number += 1

        # --- 2. Simulate Competing Teams' Turns ---
        if not done: # Only simulate if agent's pick was valid
            while self.current_pick_idx < len(self.draft_order) and \
                  self.draft_order[self.current_pick_idx] != self.agent_team_id:

                current_sim_team_id = self.draft_order[self.current_pick_idx]
                sim_drafted_player = self._simulate_competing_pick(current_sim_team_id)

                if sim_drafted_player:
                    self.teams_rosters[current_sim_team_id]['PLAYERS'].append(sim_drafted_player)
                    self.available_players_ids.remove(sim_drafted_player.player_id)
                    self._update_roster_counts(current_sim_team_id, sim_drafted_player)
                else:
                    print(f"[{self.current_pick_number}] Warning: Competing team {current_sim_team_id} could not make a pick. Ending round early.")
                    done = True 
                    info['draft_ended_prematurely_sim_team'] = True
                    break

                self.current_pick_idx += 1
                self.current_pick_number += 1

        # --- 3. Check for Done Condition ---
        if len(self.teams_rosters[self.agent_team_id]['PLAYERS']) >= self.total_roster_size_per_team:
            done = True 
            info['draft_complete'] = True 
        elif self.current_pick_idx >= len(self.draft_order):
            done = True 
            info['draft_ended_prematurely'] = True 

        # --- 4. Final Reward Calculation if Done ---
        if done:
            agent_final_weighted_score = 0
            if self.config.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
                starters_dict, all_non_starters_list, _ = self._categorize_roster_by_slots(
                    self.teams_rosters[self.agent_team_id]['PLAYERS'],
                    self.config.ROSTER_STRUCTURE,
                    self.config.BENCH_MAXES
                )
                starter_points = sum(p.projected_points for pos_list in starters_dict.values() for p in pos_list)
                bench_points = sum(p.projected_points for p in all_non_starters_list)
                
                agent_final_weighted_score = (starter_points * self.config.STARTER_POINTS_WEIGHT) + \
                                             (bench_points * self.config.BENCH_POINTS_WEIGHT)
                info['raw_starter_points'] = starter_points
                info['raw_bench_points'] = bench_points
            else:
                agent_final_weighted_score = sum(p.projected_points for p in self.teams_rosters[self.agent_team_id]['PLAYERS'])
            
            info['final_score_agent'] = agent_final_weighted_score # Store for info

            # Apply bonus for full roster
            if 'draft_complete' in info and self.config.BONUS_FOR_FULL_ROSTER > 0:
                reward += self.config.BONUS_FOR_FULL_ROSTER

            # --- Competitive Reward Logic ---
            if self.config.ENABLE_COMPETITIVE_REWARD:
                opponent_scores = []
                for team_id, roster_data in self.teams_rosters.items():
                    if team_id != self.agent_team_id:
                        if self.config.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
                            opp_starters_dict, opp_all_non_starters_list, _ = self._categorize_roster_by_slots(
                                roster_data['PLAYERS'],
                                self.config.ROSTER_STRUCTURE,
                                self.config.BENCH_MAXES
                            )
                            opp_starter_points = sum(p.projected_points for pos_list in opp_starters_dict.values() for p in pos_list)
                            opp_bench_points = sum(p.projected_points for p in opp_all_non_starters_list)
                            opponent_score = (opp_starter_points * self.config.STARTER_POINTS_WEIGHT) + \
                                             (opp_bench_points * self.config.BENCH_POINTS_WEIGHT)
                        else:
                            opponent_score = sum(p.projected_points for p in roster_data['PLAYERS'])
                        opponent_scores.append(opponent_score)
                
                competitive_reward_component = 0 # Initialize competitive component
                if not opponent_scores: # Handle case with no opponents (e.g., NUM_TEAMS = 1)
                    info['competitive_mode'] = 'None (No Opponents)'
                    # No competitive reward if there are no opponents
                else:
                    opponent_scores_np = np.array(opponent_scores)

                    if self.config.COMPETITIVE_REWARD_MODE == 'MAX_OPPONENT_DIFFERENCE':
                        max_opponent_score = np.max(opponent_scores_np)
                        competitive_reward_component = agent_final_weighted_score - max_opponent_score
                        info['competitive_mode'] = 'Max Opponent Difference'
                        info['target_opponent_score'] = max_opponent_score # Log the specific score targeted
                    elif self.config.COMPETITIVE_REWARD_MODE == 'AVG_OPPONENT_DIFFERENCE':
                        avg_opponent_score = np.mean(opponent_scores_np)
                        competitive_reward_component = agent_final_weighted_score - avg_opponent_score
                        info['competitive_mode'] = 'Average Opponent Difference'
                        info['target_opponent_score'] = avg_opponent_score # Log the specific score targeted
                    elif self.config.COMPETITIVE_REWARD_MODE == 'NONE': # Explicitly handle NONE for competitive reward
                        competitive_reward_component = 0
                        info['competitive_mode'] = 'None'
                    else: # Fallback for unknown competitive mode
                        print(f"Warning: Unknown COMPETITIVE_REWARD_MODE: {self.config.COMPETITIVE_REWARD_MODE}. No competitive reward applied.")
                        competitive_reward_component = 0
                        info['competitive_mode'] = 'Unknown/Fallback'

                    # Apply STD Deviation Penalty (if enabled and applicable)
                    if self.config.ENABLE_OPPONENT_STD_DEV_PENALTY and len(opponent_scores_np) > 1:
                        opponent_std_dev = np.std(opponent_scores_np)
                        std_dev_penalty = opponent_std_dev * self.config.OPPONENT_STD_DEV_PENALTY_WEIGHT
                        
                        competitive_reward_component -= std_dev_penalty # Subtract penalty
                        info['opponent_std_dev_applied'] = True
                        info['opponent_std_dev'] = opponent_std_dev
                        info['std_dev_penalty_amount'] = std_dev_penalty
                    else:
                        info['opponent_std_dev_applied'] = False

                reward += competitive_reward_component
            else:
                # If competitive reward is NOT enabled, the base reward is simply the agent's score
                reward += agent_final_weighted_score
                info['competitive_mode'] = 'Disabled'
                info['opponent_std_dev_applied'] = False
                
            info['final_reward_total'] = reward # The total reward given to the agent

        # --- 5. Get Next State and Action Mask ---
        observation = self._get_state()
        info['action_mask'] = self.get_action_mask()

        return observation, reward, done, False, info

    def _categorize_roster_by_slots(self, team_roster: List[Player], roster_structure: Dict, bench_maxes: Dict) -> Tuple[Dict[str, List[Player]], List[Player], List[Player]]:
        """
        Categorizes players into starters, bench, and explicit flex players based on roster structure
        and projected points. This is used for reward calculation.
        Returns: (starters_dict, all_non_starters_list, flex_players_list)
        """
        starters = defaultdict(list) # {'QB': [player1], 'RB': [player2, player3], ...}
        temp_flex_candidates = [] # Players that could potentially fill a FLEX spot
        other_bench_players = [] # Players that go straight to the bench

        # Sort players by projected points (descending) for optimal placement
        sorted_players = sorted(team_roster, key=lambda p: p.projected_points, reverse=True)

        temp_pos_counts = defaultdict(int) # Tracks players in specific position slots (QB, RB, WR, TE)
        
        # First pass: Fill all direct starter spots
        for player in sorted_players:
            pos = player.position
            if temp_pos_counts[pos] < roster_structure.get(pos, 0):
                starters[pos].append(player)
                temp_pos_counts[pos] += 1
            elif pos in ['RB', 'WR', 'TE']: # These can be flex or bench
                temp_flex_candidates.append(player)
            else: # QB, TE (if not filling starter and not flex eligible) or too many of any pos
                other_bench_players.append(player)

        # Second pass: Fill FLEX spots from the best available flex_candidates
        flex_players_list = []
        current_flex_fill = 0
        
        # Sort flex candidates by projected points to pick the best ones for FLEX
        temp_flex_candidates.sort(key=lambda p: p.projected_points, reverse=True)

        for player in temp_flex_candidates:
            if current_flex_fill < roster_structure.get('FLEX', 0):
                flex_players_list.append(player)
                current_flex_fill += 1
            else:
                other_bench_players.append(player) # If not used for flex, move to general bench

        # Finally, remaining direct bench players from first pass go to all_non_starters
        # And any remaining flex candidates that didn't fit into a flex spot.
        all_non_starters_list = other_bench_players + [p for p in temp_flex_candidates if p not in flex_players_list]

        # Ensure all players are accounted for (for debugging)
        # assert len(team_roster) == sum(len(lst) for lst in starters.values()) + len(all_non_starters_list)

        return starters, all_non_starters_list, flex_players_list


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
            'agent_roster_size': len(self.teams_rosters[self.config.AGENT_START_POSITION]['PLAYERS']),
            'available_players_count': len(self.available_players_ids),
            'manual_draft_teams': list(self.manual_draft_teams)
        }

    def _get_state(self) -> np.ndarray:
        """
        Constructs the state vector based on enabled features and normalizes it.
        """
        state_values = []
        for feature_name in self.config.ENABLED_STATE_FEATURES:
            if feature_name in self.state_features_map:
                value = self.state_features_map[feature_name]()
                state_values.append(value)
            else:
                # This should ideally not happen if ENABLED_STATE_FEATURES is subset of ALL_STATE_FEATURES
                # and all mappings are correctly defined.
                print(f"Warning: Unknown state feature '{feature_name}' requested. Appending 0.")
                state_values.append(0)

        state_array = np.array(state_values, dtype=np.float32)

        # Normalize the state array
        if self.config.STATE_NORMALIZATION_METHOD == 'min_max':
            state_array = self._normalize_min_max(state_array)
        elif self.config.STATE_NORMALIZATION_METHOD == 'z_score':
            state_array = self._normalize_z_score(state_array)
        # 'none' means no normalization

        return state_array

    def _normalize_min_max(self, state_array: np.ndarray) -> np.ndarray:
        # These max values should be estimated based on your player data's expected ranges.
        # It's good practice to run some data analysis to set these accurately.
        max_values = {
            "best_available_qb_points": 400.0, "best_available_rb_points": 350.0,
            "best_available_wr_points": 350.0, "best_available_te_points": 300.0,
            "current_roster_qb_count": self.config.ROSTER_STRUCTURE['QB'] + self.config.BENCH_MAXES['QB'],
            "current_roster_rb_count": self.config.ROSTER_STRUCTURE['RB'] + self.config.BENCH_MAXES['RB'],
            "current_roster_wr_count": self.config.ROSTER_STRUCTURE['WR'] + self.config.BENCH_MAXES['WR'],
            "current_roster_te_count": self.config.ROSTER_STRUCTURE['TE'] + self.config.BENCH_MAXES['TE'],
            "available_roster_slots_qb": self.config.ROSTER_STRUCTURE['QB'] + self.config.BENCH_MAXES['QB'],
            "available_roster_slots_rb": self.config.ROSTER_STRUCTURE['RB'] + self.config.BENCH_MAXES['RB'],
            "available_roster_slots_wr": self.config.ROSTER_STRUCTURE['WR'] + self.config.BENCH_MAXES['WR'],
            "available_roster_slots_te": self.config.ROSTER_STRUCTURE['TE'] + self.config.BENCH_MAXES['TE'],
            "available_roster_slots_flex": self.config.ROSTER_STRUCTURE['FLEX'],
            "qb_available_flag": 1.0, "rb_available_flag": 1.0,
            "wr_available_flag": 1.0, "te_available_flag": 1.0,
            "current_pick_number": len(self.draft_order) if self.draft_order else 1.0, # Max possible pick number
            "agent_start_position": float(self.config.NUM_TEAMS), # Max team number
            # New features max values
            "second_best_available_qb_points": 400.0, # Same max as best for now
            "second_best_available_rb_points": 350.0, # Same max as best for now
            "second_best_available_wr_points": 350.0, # Same max as best for now
            "second_best_available_te_points": 300.0, # Same max as best for now
            "next_pick_opponent_qb_count": self.config.ROSTER_STRUCTURE['QB'] + self.config.BENCH_MAXES['QB'],
            "next_pick_opponent_rb_count": self.config.ROSTER_STRUCTURE['RB'] + self.config.BENCH_MAXES['RB'],
            "next_pick_opponent_wr_count": self.config.ROSTER_STRUCTURE['WR'] + self.config.BENCH_MAXES['WR'],
            "next_pick_opponent_te_count": self.config.ROSTER_STRUCTURE['TE'] + self.config.BENCH_MAXES['TE'],
        }
        min_values = {k: 0.0 for k in max_values}
        
        normalized_state = []
        for i, feature_name in enumerate(self.config.ENABLED_STATE_FEATURES):
            val = state_array[i]
            max_val = max_values.get(feature_name)
            min_val = min_values.get(feature_name)

            if max_val is None or min_val is None:
                # Fallback for features not explicitly in max_values, treat as boolean/flag if reasonable
                if 'flag' in feature_name:
                    max_val = 1.0
                    min_val = 0.0
                else: # Generic fallback for unknown numerics
                    print(f"Warning: No explicit min/max for '{feature_name}'. Using 0-1 range if value is 0 or 1, else using value as is.")
                    if val == 0.0 or val == 1.0: # Assume it's a flag if its value is 0 or 1 and no specific range
                        max_val = 1.0
                        min_val = 0.0
                    else: # If it's a number outside 0/1 and no range, return as is (no norm)
                        normalized_state.append(val)
                        continue # Skip to next feature

            if max_val == min_val:
                normalized_state.append(0.0);
            else:
                normalized_state.append((val - min_val) / (max_val - min_val))
        return np.array(normalized_state, dtype=np.float32)

    def _normalize_z_score(self, state_array: np.ndarray) -> np.ndarray:
        # Placeholder for z-score. For proper implementation, you'd collect
        # mean and std deviation over many episodes.
        mean = np.zeros_like(state_array)
        std = np.ones_like(state_array)
        # Avoid division by zero
        std[std == 0] = 1.0 # If std is 0, set to 1 to avoid error. Will result in 0 after subtraction.
        return (state_array - mean) / std

    # Original _get_best_available_player_by_pos is now generalized by _get_kth_best_available_player_by_pos
    def _get_best_available_player_by_pos(self, position: str) -> Optional[Player]:
        """Returns the available player with the highest projected points for a given position."""
        return self._get_kth_best_available_player_by_pos(position, 1)

    def _can_team_draft_position(self, team_id: int, position: str) -> bool:
        """
        Checks if a team can draft a player of a given position, considering roster limits
        and general availability of players of that position in the pool.
        """
        # First, check if there are any players of this position available in the pool
        any_player_of_pos_available = any(
            self.player_map[pid].position == position for pid in self.available_players_ids
        )
        if not any_player_of_pos_available:
            return False

        # Now, check if the specific team has room for this position
        roster_counts = self.teams_rosters[team_id]
        current_pos_count = roster_counts[position]
        current_flex_count = roster_counts['FLEX']

        max_pos_count = self.config.ROSTER_STRUCTURE.get(position, 0) + self.config.BENCH_MAXES.get(position, 0)
        max_flex_count = self.config.ROSTER_STRUCTURE.get('FLEX', 0)

        if current_pos_count < max_pos_count:
            return True # Room in starting or bench roster for this position
        elif position in ['RB', 'WR', 'TE'] and current_flex_count < max_flex_count:
            return True # Room in FLEX spot for RB/WR/TE
        
        return False # No room for this position

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array representing the valid actions for the agent in the current state.
        A value of True means the action is valid, False means it's invalid.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        for action, position_choice in self.action_to_position.items():
            if self._can_team_draft_position(self.agent_team_id, position_choice):
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
        if not self._can_team_draft_position(team_id, position_choice):
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
            return False, None # Should not happen if available_players_for_pos is not empty and _can_team_draft_position is True

    def _update_roster_counts(self, team_id: int, player: Player):
        """Updates team's roster counts, including handling FLEX logic."""
        
        # Check if the player fills a direct position spot first
        current_pos_count = self.teams_rosters[team_id][player.position]
        required_fixed_spots = self.config.ROSTER_STRUCTURE.get(player.position, 0)
        max_pos_spots = required_fixed_spots + self.config.BENCH_MAXES.get(player.position, 0)

        if current_pos_count < max_pos_spots:
            # Player fills a direct position spot (starter or bench)
            self.teams_rosters[team_id][player.position] += 1
        elif player.position in ['RB', 'WR', 'TE'] and self.teams_rosters[team_id]['FLEX'] < self.config.ROSTER_STRUCTURE['FLEX']:
            # Player fills a FLEX spot
            self.teams_rosters[team_id]['FLEX'] += 1
        else:
            # This case indicates an error in _try_select_player_for_team or environment logic
            # as a player should only be "drafted" if there's a spot.
            print(f"Error: Player {player.name} ({player.position}) drafted for team {team_id} but no valid spot found to update counts!")
            # This shouldn't be reached if _try_select_player_for_team works correctly.

    def _simulate_competing_pick(self, team_id: int) -> Optional[Player]:
        roster_counts = self.teams_rosters[team_id]
        
        # Get opponent's specific strategy, or fallback to default
        opponent_strategy = self.config.OPPONENT_TEAM_STRATEGIES.get(
            team_id, self.config.DEFAULT_OPPONENT_STRATEGY
        )
        logic = opponent_strategy['logic']

        # Define positions that our environment and roster structure can handle.
        handled_positions = set(self.config.ROSTER_STRUCTURE.keys()) - {'FLEX'}
        flex_eligible_positions = {'RB', 'WR', 'TE'}
        all_draftable_positions = handled_positions.union(flex_eligible_positions)

        # Get all players that are available AND whose positions are handled by our logic
        available_players_filtered = [self.player_map[pid] 
                                      for pid in self.available_players_ids 
                                      if self.player_map[pid].position in all_draftable_positions]

        # Filter these further to only include players that the current team can actually draft
        # (i.e., they have a roster spot for that position or FLEX)
        eligible_players_for_pick = []
        for player in available_players_filtered:
            if self._can_team_draft_position(team_id, player.position):
                eligible_players_for_pick.append(player)
        
        if not eligible_players_for_pick:
            return None # No eligible player found for this team

        # --- Determine the pick based on opponent's strategy ---
        chosen_player = None

        if logic == 'AGENT_MODEL':
            # Use a trained agent model for this opponent
            if team_id in self.opponent_models:
                opponent_model = self.opponent_models[team_id]
                # Get the state from the perspective of this opponent team
                # Temporarily set agent_team_id to this opponent's team_id to get its state features
                original_agent_team_id = self.agent_team_id
                self.agent_team_id = team_id
                opponent_state = self._get_state()
                # Get the action mask for this opponent team
                opponent_action_mask = self.get_action_mask()
                # Restore agent_team_id
                self.agent_team_id = original_agent_team_id

                state_tensor = torch.from_numpy(opponent_state).float().unsqueeze(0)
                with torch.no_grad():
                    # Opponent models should also respect action masking
                    action_probs = opponent_model.get_action_probabilities(state_tensor, action_mask=opponent_action_mask)
                    action_chosen = torch.argmax(action_probs).item() # Greedy pick for opponent agents

                selected_position_by_opponent = self.action_to_position[action_chosen]
                
                # Attempt to pick the player chosen by the opponent agent
                is_valid_pick_by_opponent, drafted_player_obj = self._try_select_player_for_team(
                    team_id, selected_position_by_opponent, self.available_players_ids
                )
                if is_valid_pick_by_opponent:
                    chosen_player = drafted_player_obj
                else:
                    # If the opponent model made an invalid pick (which should be rare with masking)
                    # or there's an unforeseen edge case, fall back to a simple strategy.
                    print(f"Warning: Opponent Team {team_id} (AGENT_MODEL) tried an invalid pick ({selected_position_by_opponent}). Falling back to ADP.")
                    sorted_by_adp = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                    chosen_player = sorted_by_adp[0] if sorted_by_adp else None
            else:
                print(f"Warning: AGENT_MODEL requested for Team {team_id} but no model loaded. Falling back to DEFAULT_OPPONENT_STRATEGY.")
                # Update strategy to prevent repeated warnings in this episode
                self.config.OPPONENT_TEAM_STRATEGIES[team_id] = self.config.DEFAULT_OPPONENT_STRATEGY.copy()
                # Fallback to default heuristic for this turn
                logic = self.config.DEFAULT_OPPONENT_STRATEGY['logic']
                randomness_factor = self.config.DEFAULT_OPPONENT_STRATEGY['randomness_factor']
                suboptimal_strategy = self.config.DEFAULT_OPPONENT_STRATEGY['suboptimal_strategy']
                positional_priority = self.config.DEFAULT_OPPONENT_STRATEGIES['positional_priority'] # Used for HEURISTIC logic
                # Now proceed with the non-AGENT_MODEL logic as a fallback below

        elif logic == 'RANDOM':
            # 'RANDOM' logic directly picks a random eligible player
            chosen_player = random.choice(eligible_players_for_pick)
        elif logic == 'ADP' or logic == 'HEURISTIC':
            # Determine the "best" pick based on the configured logic
            best_pick_by_logic = None
            if logic == 'ADP':
                sorted_by_logic = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                best_pick_by_logic = sorted_by_logic[0] if sorted_by_logic else None
            elif logic == 'HEURISTIC':
                # Heuristic: Prioritize by position need and then projected points, using custom priority
                positional_priority = opponent_strategy['positional_priority'] # Ensure we use specific opponent's priority
                
                def get_best_for_pos_heuristic(pos_list: List[str]):
                    for pos in pos_list:
                        if roster_counts[pos] < self.config.ROSTER_STRUCTURE.get(pos, 0): 
                            eligible = [p for p in eligible_players_for_pick if p.position == pos]
                            if eligible: return sorted(eligible, key=lambda p: p.projected_points, reverse=True)[0]
                    return None

                best_pick_by_logic = get_best_for_pos_heuristic(positional_priority)
                
                if not best_pick_by_logic: 
                    for pos in positional_priority: 
                        max_pos_count_total = self.config.ROSTER_STRUCTURE.get(pos, 0) + self.config.BENCH_MAXES.get(pos, 0)
                        if roster_counts[pos] < max_pos_count_total:
                            eligible = [p for p in eligible_players_for_pick if p.position == pos]
                            if eligible:
                                best_pick_by_logic = sorted(eligible, key=lambda p: p.projected_points, reverse=True)[0]
                                if best_pick_by_logic: break 
                
                if not best_pick_by_logic:
                    if roster_counts['FLEX'] < self.config.ROSTER_STRUCTURE['FLEX']:
                         eligible_flex_players = [p for p in eligible_players_for_pick if p.position in ['RB', 'WR', 'TE']]
                         if eligible_flex_players:
                             best_pick_by_logic = sorted(eligible_flex_players, key=lambda p: p.projected_points, reverse=True)[0]
            
            if not best_pick_by_logic:
                return None 

            randomness_factor = opponent_strategy['randomness_factor'] # Ensure we use specific opponent's randomness
            suboptimal_strategy = opponent_strategy['suboptimal_strategy'] # Ensure we use specific opponent's suboptimal strategy

            # Apply randomness
            if random.random() < randomness_factor:
                if suboptimal_strategy == 'RANDOM_ELIGIBLE':
                    chosen_player = random.choice(eligible_players_for_pick)
                elif suboptimal_strategy == 'NEXT_BEST_ADP':
                    temp_list = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                    if len(temp_list) > 1:
                        chosen_player = temp_list[1]
                    else:
                        chosen_player = temp_list[0]
                elif suboptimal_strategy == 'NEXT_BEST_HEURISTIC':
                    if len(eligible_players_for_pick) > 1:
                        other_eligible = [p for p in eligible_players_for_pick if p != best_pick_by_logic]
                        if other_eligible:
                            chosen_player = random.choice(other_eligible)
                        else:
                            chosen_player = best_pick_by_logic
                    else:
                        chosen_player = best_pick_by_logic
                else:
                    print(f"Warning: Unknown suboptimal pick strategy: {suboptimal_strategy}. Falling back to best pick.")
                    chosen_player = best_pick_by_logic
            else:
                chosen_player = best_pick_by_logic
        else: # For safety, if logic is somehow neither defined nor AGENT_MODEL, fallback to ADP
            print(f"Warning: Unknown competing team logic: {logic}. Falling back to ADP.")
            sorted_by_adp = sorted(eligible_players_for_pick, key=lambda p: p.adp)
            chosen_player = sorted_by_adp[0] if sorted_by_adp else None

        return chosen_player

    def draft_player(self, player_id: int):
        """
        Manually drafts a specific player for the current team.
        This method does NOT advance the draft or simulate opponent picks.
        """
        if self.current_pick_idx >= len(self.draft_order):
            raise ValueError("The draft has already concluded. No more picks can be made.")

        # If a team was overridden, use that ID, otherwise use the team from the draft order.
        current_team_id = self._overridden_team_id if self._overridden_team_id is not None else self.draft_order[self.current_pick_idx]
        
        # Find the player object
        drafted_player = self.player_map.get(player_id)
        if not drafted_player or drafted_player.player_id not in self.available_players_ids:
            raise ValueError(f"Player with ID {player_id} is not available to be drafted.")

        # Validate that the team can draft this position
        if not self._can_team_draft_position(current_team_id, drafted_player.position):
            raise ValueError(f"Team {current_team_id} cannot draft a {drafted_player.position} as the roster is full for that position.")

        # Store the state before making the pick for the undo history
        was_override = (self._overridden_team_id is not None)
        
        # Record the pick in history BEFORE making changes
        self._draft_history.append({
            'player_id': drafted_player.player_id,
            'team_id': current_team_id,
            'is_manual_pick': True,
            'previous_pick_idx': self.current_pick_idx,
            'previous_pick_number': self.current_pick_number,
            'previous_overridden_team_id': self._overridden_team_id,
            'was_override': was_override
        })

        # Update roster and available players
        self.teams_rosters[current_team_id]['PLAYERS'].append(drafted_player)
        self.available_players_ids.remove(drafted_player.player_id)
        self._update_roster_counts(current_team_id, drafted_player)

        # Advance pick index and number
        self.current_pick_idx += 1
        self.current_pick_number += 1
        
        # Reset override after it's used for a pick
        self._overridden_team_id = None


    def undo_last_pick(self):
        """
        Reverts the last pick made in the draft.
        """
        if not self._draft_history:
            raise ValueError("No picks to undo.")

        last_pick_info = self._draft_history.pop() # Get the last pick info
        player_id = last_pick_info['player_id']
        team_id = last_pick_info['team_id']
        is_manual_pick = last_pick_info['is_manual_pick']

        # Re-add player to available pool
        self.available_players_ids.add(player_id)
        player = self.player_map[player_id]

        # Remove player from team roster and update counts
        # Find the player object in the team's roster and remove it
        player_removed = False
        for i, p in enumerate(self.teams_rosters[team_id]['PLAYERS']):
            if p.player_id == player_id:
                self.teams_rosters[team_id]['PLAYERS'].pop(i);
                player_removed = True
                break
        
        if not player_removed:
            print(f"Warning: Player {player.name} (ID: {player_id}) not found in Team {team_id}'s roster during undo.")

        # Decrement roster counts (reverse of _update_roster_counts)
        # This logic needs to be careful with FLEX spots. For simplicity, we'll just decrement the position count.
        # A more robust undo might need to store how the player filled the spot (e.g., QB, RB, WR, TE, or FLEX)
        # For now, we assume the player filled their primary position or a flex if primary was full.
        # This is a simplification and might not perfectly restore complex flex scenarios.
        if self.teams_rosters[team_id][player.position] > 0:
            self.teams_rosters[team_id][player.position] -= 1
        elif player.position in ['RB', 'WR', 'TE'] and self.teams_rosters[team_id]['FLEX'] > 0:
            self.teams_rosters[team_id]['FLEX'] -= 1
        else:
            print(f"Warning: Could not decrement roster count for {player.name} ({player.position}) on Team {team_id} during undo.")

        # Revert pick index and number and overridden team state
        self.current_pick_idx = last_pick_info['previous_pick_idx']
        self.current_pick_number = last_pick_info['previous_pick_number']
        self._overridden_team_id = last_pick_info['previous_overridden_team_id']

        

        print(f"Undo successful. Current pick: {self.current_pick_number}, Team on clock: {self.draft_order[self.current_pick_idx]}")

    def set_current_team_picking(self, team_id: int):
        """
        Sets the current team on the clock to the specified team_id for the *next* pick only.
        This is primarily for manual override from the frontend.
        """
        if team_id not in range(1, self.config.NUM_TEAMS + 1):
            raise ValueError(f"Invalid team ID: {team_id}. Must be between 1 and {self.config.NUM_TEAMS}.")
        
        # Check if the team has any picks left in the draft order
        if team_id not in self.draft_order[self.current_pick_idx:]:
            raise ValueError(f"Team {team_id} has no more picks remaining in the draft order.")

        self._overridden_team_id = team_id
        print(f"Next pick will be overridden for Team {team_id}.")

    def simulate_single_pick(self):
        """
        Simulates a single pick by the current team on the clock.
        This is used for 'Manual Step' mode.
        """
        if self.current_pick_idx >= len(self.draft_order):
            raise ValueError("The draft has already concluded. No more picks can be made.")

        current_sim_team_id = self._overridden_team_id if self._overridden_team_id is not None else self.draft_order[self.current_pick_idx]

        # Reset override after it's used
        self._overridden_team_id = None

        if current_sim_team_id in self.manual_draft_teams:
            raise ValueError("It is a manual team's turn. Cannot simulate pick.")

        sim_drafted_player = self._simulate_competing_pick(current_sim_team_id)

        if sim_drafted_player:
            self.teams_rosters[current_sim_team_id]['PLAYERS'].append(sim_drafted_player)
            self.available_players_ids.remove(sim_drafted_player.player_id)
            self._update_roster_counts(current_sim_team_id, sim_drafted_player)
            # Record the simulated pick in history, marking it as not manual
            self._draft_history.append({
                'player_id': sim_drafted_player.player_id,
                'team_id': current_sim_team_id,
                'is_manual_pick': False,
                'previous_pick_idx': self.current_pick_idx, # Store state before this sim pick
                'previous_pick_number': self.current_pick_number,
                'previous_overridden_team_id': None, # Sim picks don't have overrides
                'was_override': False
            })
        else:
            raise ValueError(f"Competing team {current_sim_team_id} could not make a valid pick.")

        self.current_pick_idx += 1
        self.current_pick_number += 1

        print(f"Simulated pick for Team {current_sim_team_id}. Current pick: {self.current_pick_number}")

    def get_ai_suggestion(self):
        """Gets the AI's suggested action for the current state."""
        if self.current_pick_idx >= len(self.draft_order):
            return "Draft is over."

        if not self.agent_model:
            return "AI model not loaded."

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
            action_probs = self.agent_model.get_action_probabilities(state_tensor, action_mask=action_mask)
            action_chosen = torch.argmax(action_probs).item()

        return self.action_to_position[action_chosen]

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
        for pos_type, count_needed in self.config.ROSTER_STRUCTURE.items():
            if pos_type == 'FLEX':
                roster_summary.append(f"{pos_type}: {agent_roster_data['FLEX']}/{count_needed}")
            else:
                bench_max = self.config.BENCH_MAXES.get(pos_type, 0)
                current_total = agent_roster_data[pos_type]
                roster_summary.append(f"{pos_type}: {current_total}/{count_needed + bench_max}")
        print("  " + " | ".join(roster_summary))
        
        # Optionally, list players
        # print("  Players:", [p.name for p in agent_roster_data['PLAYERS']])

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
            if 'opponent_std_dev_applied' in info and info['opponent_std_dev_applied']:
                print(f"Opponent score std dev penalty applied: {info['std_dev_penalty_amount']:.2f}")
            
            # Print final rosters for all teams
            for team_id in range(1, config.NUM_TEAMS + 1):
                roster_data = env.teams_rosters[team_id]
                team_score = sum(p.projected_points for p in roster_data['PLAYERS'])
                print(f"\nTeam {team_id} Roster (Total Raw Points: {team_score:.2f}):")
                for player in sorted(roster_data['PLAYERS'], key=lambda p: p.projected_points, reverse=True):
                    print(f"  - {player.name} ({player.position}, {player.projected_points:.1f} pts)")

    env.close()