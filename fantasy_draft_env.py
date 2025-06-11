import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import math
import random
from typing import Optional, Dict, List, Tuple

from config import Config
from data_utils import load_player_data, Player

class FantasyFootballDraftEnv(gym.Env):
    """
    Custom OpenAI Gym Environment for a Fantasy Football Draft.
    The agent learns to pick optimal positions to maximize projected team points.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config: Config):
        super(FantasyFootballDraftEnv, self).__init__()
        self.config = config

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
            "best_available_qb_points": self._get_best_available_qb_points,
            "best_available_rb_points": self._get_best_available_rb_points,
            "best_available_wr_points": self._get_best_available_wr_points,
            "best_available_te_points": self._get_best_available_te_points,
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

        # Pre-calculate total roster size for done check
        self.total_roster_size_per_team = sum(self.config.ROSTER_STRUCTURE.values()) + sum(self.config.BENCH_MAXES.values())
        # The sum of required spots plus bench spots for each position.
        # Flex players reduce available flex slots but increase RB/WR/TE counts,
        # so this is complex. For now, ensure we don't overfill total_roster_size.


    def _get_best_available_qb_points(self):
        best_qb = self._get_best_available_player_by_pos('QB')
        return best_qb.projected_points if best_qb else 0

    def _get_best_available_rb_points(self):
        best_rb = self._get_best_available_player_by_pos('RB')
        return best_rb.projected_points if best_rb else 0

    def _get_best_available_wr_points(self):
        best_wr = self._get_best_available_player_by_pos('WR')
        return best_wr.projected_points if best_wr else 0

    def _get_best_available_te_points(self):
        best_te = self._get_best_available_player_by_pos('TE')
        return best_te.projected_points if best_te else 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Reset all internal game state variables
        self.available_players_ids = {p.player_id for p in self.all_players_data}
        self.teams_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []})
        self.current_pick_idx = 0
        self.current_pick_number = 1 # Start at pick 1

        # Generate draft order (snake draft)
        self.draft_order = self._generate_snake_draft_order(self.config.NUM_TEAMS, self.total_roster_size_per_team)

        # IMPORTANT: Advance the draft until it's the agent's turn
        # Simulate picks by other teams until the agent's turn arrives or the draft ends.
        while self.current_pick_idx < len(self.draft_order) and \
              self.draft_order[self.current_pick_idx] != self.agent_team_id:

            current_sim_team_id = self.draft_order[self.current_pick_idx]
            sim_drafted_player = self._simulate_competing_pick(current_sim_team_id)

            if sim_drafted_player:
                self.teams_rosters[current_sim_team_id]['PLAYERS'].append(sim_drafted_player)
                self.available_players_ids.remove(sim_drafted_player.player_id)
                self._update_roster_counts(current_sim_team_id, sim_drafted_player)
            else:
                # If a competing team cannot make a pick (e.g., no players left that fit their needs/roster)
                # and no players are available globally, the draft may effectively end early.
                if not self.available_players_ids:
                    print(f"[{self.current_pick_number}] No players left in pool. Ending draft early during reset advance.")
                    break # End the reset advance loop if no more players to pick
                # If there are still players, but the sim team couldn't pick (e.g., specific needs not met),
                # we just advance the pick counter and hope for other teams.
                # This could indicate a need for more robust sim team logic or draft end conditions.
                print(f"[{self.current_pick_number}] Warning: Competing team {current_sim_team_id} could not make a valid pick.")


            self.current_pick_idx += 1
            self.current_pick_number += 1
        
        # After this loop, it's either the agent's turn, or the draft has already completed
        # or there are no more players available at all.

        observation = self._get_state()
        info = self._get_info() 
        
        # If the draft ended before the agent even got its first pick (e.g., very late draft position in a short draft)
        if self.current_pick_idx >= len(self.draft_order) and len(self.teams_rosters[self.agent_team_id]['PLAYERS']) == 0:
            info['episode_ended_before_agent_first_pick'] = True
            # The agent will get a 0 reward for this episode if it couldn't make any picks.

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
            if 'draft_complete' in info and self.config.BONUS_FOR_FULL_ROSTER > 0:
                reward += self.config.BONUS_FOR_FULL_ROSTER

            final_projected_points = sum(p.projected_points for p in self.teams_rosters[self.agent_team_id]['PLAYERS'])
            reward += final_projected_points
            info['final_score'] = final_projected_points

        # --- 5. Get Next State and Action Mask ---
        observation = self._get_state()
        info['action_mask'] = self.get_action_mask()

        return observation, reward, done, False, info


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
        return {
            'current_pick_number': self.current_pick_number,
            'current_team_picking': self.draft_order[self.current_pick_idx] if self.current_pick_idx < len(self.draft_order) else None,
            'agent_roster_size': len(self.teams_rosters[self.agent_team_id]['PLAYERS']),
            'available_players_count': len(self.available_players_ids)
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
                normalized_state.append(0.0)
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

    def _get_best_available_player_by_pos(self, position: str) -> Optional[Player]:
        """Returns the available player with the highest projected points for a given position."""
        best_player = None
        highest_points = -1

        for player_id in self.available_players_ids:
            player = self.player_map[player_id]
            if player.position == position:
                if player.projected_points > highest_points:
                    highest_points = player.projected_points
                    best_player = player
        return best_player

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
        randomness_factor = opponent_strategy['randomness_factor']
        suboptimal_strategy = opponent_strategy['suboptimal_strategy']
        positional_priority = opponent_strategy['positional_priority'] # Used for HEURISTIC logic

        # Define positions that our environment and roster structure can handle.
        handled_positions = set(self.config.ROSTER_STRUCTURE.keys()) - {'FLEX'}
        flex_eligible_positions = {'RB', 'WR', 'TE'}
        all_draftable_positions = handled_positions.union(flex_eligible_positions)

        # Get all players that are available AND whose positions are handled by our logic
        available_players_filtered = [self.player_map[pid] 
                                      for pid in self.available_players_ids 
                                      if self.player_map[pid].position in all_draftable_positions]

        # Filter these further to only include players that the current team can actually draft
        eligible_players_for_pick = []
        for player in available_players_filtered:
            if self._can_team_draft_position(team_id, player.position):
                eligible_players_for_pick.append(player)
        
        if not eligible_players_for_pick:
            return None # No eligible player found for this team

        # --- Determine the pick based on opponent's strategy ---
        chosen_player = None

        if logic == 'RANDOM':
            # 'RANDOM' logic directly picks a random eligible player
            chosen_player = random.choice(eligible_players_for_pick)
        else:
            # Determine the "best" pick based on the configured logic (ADP or HEURISTIC)
            best_pick_by_logic = None
            if logic == 'ADP':
                # Sort by ADP (ascending)
                sorted_by_logic = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                best_pick_by_logic = sorted_by_logic[0] if sorted_by_logic else None
            elif logic == 'HEURISTIC':
                # Heuristic: Prioritize by position need and then projected points, using custom priority
                
                # Helper to get best player for a position among eligible
                def get_best_for_pos(pos_list: List[str]):
                    for pos in pos_list:
                        # Check starting spots for that position first
                        if roster_counts[pos] < self.config.ROSTER_STRUCTURE.get(pos, 0): 
                            eligible = [p for p in eligible_players_for_pick if p.position == pos]
                            if eligible: return sorted(eligible, key=lambda p: p.projected_points, reverse=True)[0]
                    return None

                # 1. Attempt to fill core starting spots based on positional_priority
                best_pick_by_logic = get_best_for_pos(positional_priority)
                
                if not best_pick_by_logic: # If no starter spot filled, try bench
                    for pos in positional_priority: # Use positional_priority for bench as well
                        max_pos_count_total = self.config.ROSTER_STRUCTURE.get(pos, 0) + self.config.BENCH_MAXES.get(pos, 0)
                        if roster_counts[pos] < max_pos_count_total:
                            eligible = [p for p in eligible_players_for_pick if p.position == pos]
                            if eligible:
                                best_pick_by_logic = sorted(eligible, key=lambda p: p.projected_points, reverse=True)[0]
                                if best_pick_by_logic: break # Found a bench player
                
                if not best_pick_by_logic: # If no bench spot filled, try FLEX
                    if roster_counts['FLEX'] < self.config.ROSTER_STRUCTURE['FLEX']:
                         eligible_flex_players = [p for p in eligible_players_for_pick if p.position in ['RB', 'WR', 'TE']]
                         if eligible_flex_players:
                             best_pick_by_logic = sorted(eligible_flex_players, key=lambda p: p.projected_points, reverse=True)[0]
            
            else:
                print(f"Warning: Unknown competing team logic: {logic}. Falling back to ADP.")
                sorted_by_logic = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                best_pick_by_logic = sorted_by_logic[0] if sorted_by_logic else None

            if not best_pick_by_logic:
                return None # Should not happen if eligible_players_for_pick was not empty and logic is sound

            # Apply randomness if not a 'RANDOM' logic
            if random.random() < randomness_factor:
                # Make a suboptimal pick
                if suboptimal_strategy == 'RANDOM_ELIGIBLE':
                    chosen_player = random.choice(eligible_players_for_pick)
                elif suboptimal_strategy == 'NEXT_BEST_ADP':
                    # Exclude the very best ADP player, then pick the next best
                    temp_list = sorted(eligible_players_for_pick, key=lambda p: p.adp)
                    if len(temp_list) > 1:
                        chosen_player = temp_list[1] # Return the second best ADP player
                    else:
                        chosen_player = temp_list[0] # Fallback to best if only one option
                elif suboptimal_strategy == 'NEXT_BEST_HEURISTIC':
                    # This is a simplification: pick a random other eligible player.
                    if len(eligible_players_for_pick) > 1:
                        other_eligible = [p for p in eligible_players_for_pick if p != best_pick_by_logic]
                        if other_eligible:
                            chosen_player = random.choice(other_eligible)
                        else:
                            chosen_player = best_pick_by_logic # Fallback if no other distinct options
                    else:
                        chosen_player = best_pick_by_logic # Fallback if only one option
                else:
                    print(f"Warning: Unknown suboptimal pick strategy: {suboptimal_strategy}. Falling back to best pick.")
                    chosen_player = best_pick_by_logic
            else:
                # Make the optimal pick based on the configured logic
                chosen_player = best_pick_by_logic

        return chosen_player

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
        """Clean up resources."""
        pass