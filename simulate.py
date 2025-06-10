import torch
import os
import datetime
from collections import defaultdict
from typing import List, Dict, Optional
import numpy as np

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from policy_network import PolicyNetwork
from data_utils import Player

def calculate_roster_scores(team_roster: List[Player], roster_structure: Dict, bench_maxes: Dict) -> Dict:
    """
    Calculates projected points for starters, bench, and combined based on a team's drafted players.
    This logic attempts to fill starting positions first, then bench.
    FLEX players are assigned after primary positions are filled.
    """
    starters = defaultdict(list)
    bench = defaultdict(list)
    flex_eligible = [] # Players eligible for FLEX if primary spots are full

    # Sort players by projected points (descending) for optimal placement
    sorted_players = sorted(team_roster, key=lambda p: p.projected_points, reverse=True)

    temp_pos_counts = defaultdict(int) # Tracks players in specific position slots (QB, RB, WR, TE)
    current_flex_fill = 0 # Tracks players filling FLEX slots

    for player in sorted_players:
        pos = player.position
        
        # 1. Try to fill a primary starting spot
        if temp_pos_counts[pos] < roster_structure.get(pos, 0):
            starters[pos].append(player)
            temp_pos_counts[pos] += 1
        # 2. Try to fill a primary bench spot
        elif temp_pos_counts[pos] < (roster_structure.get(pos, 0) + bench_maxes.get(pos, 0)):
            bench[pos].append(player)
            temp_pos_counts[pos] += 1
        # 3. If RB/WR/TE and primary/bench spots are full, try for FLEX
        elif pos in ['RB', 'WR', 'TE'] and current_flex_fill < roster_structure.get('FLEX', 0):
            flex_eligible.append(player)
            current_flex_fill += 1
        else:
            # This player couldn't be optimally assigned. This can happen if, for example,
            # a team drafts too many players of a certain position beyond all their starter,
            # bench, AND flex needs. For scoring, we'll just put them on the bench.
            # This indicates the agent might be drafting sub-optimally or roster size is complex.
            bench[pos].append(player) # Fallback to bench
            print(f"Warning: Player {player.name} ({player.position}) could not be optimally assigned to a starter/flex slot. Placed on bench.")


    # Calculate total points for each segment
    starter_points = sum(p.projected_points for pos_list in starters.values() for p in pos_list)
    bench_points = sum(p.projected_points for pos_list in bench.values() for p in pos_list)
    flex_points = sum(p.projected_points for p in flex_eligible)
    
    combined_total_points = starter_points + bench_points + flex_points
    
    # Calculate average points
    num_starters = sum(len(lst) for lst in starters.values()) + len(flex_eligible)
    num_bench = sum(len(lst) for lst in bench.values())

    avg_starter_points = starter_points / num_starters if num_starters > 0 else 0
    avg_bench_points = bench_points / num_bench if num_bench > 0 else 0
    avg_combined_points = combined_total_points / (num_starters + num_bench) if (num_starters + num_bench) > 0 else 0

    return {
        'starters_total_points': starter_points,
        'bench_total_points': bench_points,
        'flex_total_points': flex_points,
        'combined_total_points': combined_total_points,
        'starters_avg_points': avg_starter_points,
        'bench_avg_points': avg_bench_points,
        'combined_avg_points': avg_combined_points,
        'roster_size': len(team_roster)
    }


def simulate_drafts(config: Config, num_runs: int):
    """
    Loads a trained agent and simulates multiple fantasy football drafts, logging all picks.
    """
    print(f"\n--- Starting {num_runs} Draft Simulations ---")
    
    # 1. Load the trained Policy Network
    # Need to know the input_dim and output_dim, which are from the environment/config
    # Temporarily instantiate env to get action space dimensions safely
    temp_env_for_dims = FantasyFootballDraftEnv(config)
    input_dim = len(config.ENABLED_STATE_FEATURES)
    output_dim = len(temp_env_for_dims.action_to_position) 
    
    policy_network = PolicyNetwork(input_dim, output_dim, config.HIDDEN_DIM)
    
    try:
        policy_network.load_state_dict(torch.load(config.MODEL_PATH_TO_LOAD))
        policy_network.eval() # Set to evaluation mode (e.g., disables dropout layers)
        print(f"Successfully loaded model from: {config.MODEL_PATH_TO_LOAD}")
    except FileNotFoundError:
        print(f"Error: Model not found at {config.MODEL_PATH_TO_LOAD}. Please ensure the path is correct.")
        print("Aborting simulation.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Aborting simulation.")
        return

    # 2. Prepare Environment for Simulation
    env = FantasyFootballDraftEnv(config) # Main environment instance

    # This loop will now run pick-by-pick directly, not using env.step's internal opponent simulation
    all_simulation_results = []

    for run_idx in range(num_runs):
        print(f"\n--- Simulation Run {run_idx + 1}/{num_runs} ---")
        
        # Reset environment for a new draft. This also pre-simulates up to agent's first pick.
        # We need the state *after* the initial reset, to be ready for the agent's pick.
        state, info = env.reset() 
        
        # Ensure the draft log for this run starts with any picks made during env.reset() advance
        # Note: env.reset() prints warnings about competing team failing to pick if needed.
        
        # We need a way to get the full log of picks *made so far* by env.reset()
        # The current env.reset() does not expose this cleanly.
        # To get the full draft log, we'll manually simulate pick by pick here instead
        # of relying on env.reset() to advance. This is more explicit for logging.

        # Re-reset and manage all picks in simulate.py for comprehensive logging
        env.available_players_ids = {p.player_id for p in env.all_players_data}
        env.teams_rosters = defaultdict(lambda: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0, 'PLAYERS': []})
        env.current_pick_idx = 0
        env.current_pick_number = 1 
        
        draft_order_for_run = env._generate_snake_draft_order(env.config.NUM_TEAMS, env.total_roster_size_per_team)
        # Ensure draft_order_for_run is set for this simulation
        env.draft_order = draft_order_for_run # Update env's internal draft order
        
        draft_log = [] # To store details of each pick in this simulation
        
        done = False
        while not done:
            if env.current_pick_idx >= len(env.draft_order):
                done = True # All picks in the planned draft order are exhausted
                break
            
            current_team_id = env.draft_order[env.current_pick_idx]
            player_picked = None
            action_taken = None # Store action for agent picks

            if current_team_id == env.agent_team_id:
                # Agent's turn: Use trained policy
                state_for_agent = env._get_state() # Get the current state for the agent
                state_tensor = torch.from_numpy(state_for_agent).float().unsqueeze(0)
                with torch.no_grad():
                    action_probs = policy_network.get_action_probabilities(state_tensor)
                    action_taken = torch.argmax(action_probs).item() # Greedy pick

                is_valid, drafted_player_obj = env._try_select_player_for_team(
                    current_team_id, env.action_to_position[action_taken], env.available_players_ids
                )
                if is_valid:
                    player_picked = drafted_player_obj
                else:
                    # Agent made an invalid pick. In simulation, we just log this and move on.
                    # It doesn't get a penalty here, and the episode continues to fill up the roster.
                    print(f"  Sim Warning: Agent (Team {current_team_id}) made an invalid pick at Pick {env.current_pick_number}. Attempted {env.action_to_position[action_taken]}.")
                    # No player_picked, so no changes to rosters/available_players for this turn.

            else:
                # Competing team's turn: Use their configured logic
                player_picked = env._simulate_competing_pick(current_team_id)
                if not player_picked:
                    # If a competing team can't make a valid pick (e.g., no players left to fit needs)
                    print(f"  Sim Warning: Competing team {current_team_id} could not make a pick at Pick {env.current_pick_number}.")
                    # No player_picked, so no changes to rosters/available_players for this turn.

            # --- Log and Update Environment for the current pick ---
            if player_picked:
                draft_log.append({
                    'pick_number': env.current_pick_number,
                    'team_id': current_team_id,
                    'player_id': player_picked.player_id,
                    'name': player_picked.name,
                    'position': player_picked.position,
                    'projected_points': player_picked.projected_points,
                    'adp': player_picked.adp,
                    'status': 'Valid Pick'
                })
                # Update environment state manually for this single pick
                env.teams_rosters[current_team_id]['PLAYERS'].append(player_picked)
                env.available_players_ids.remove(player_picked.player_id)
                env._update_roster_counts(current_team_id, player_picked)
            else:
                # Log a skipped pick (due to invalid action or no valid player found)
                draft_log.append({
                    'pick_number': env.current_pick_number,
                    'team_id': current_team_id,
                    'player_id': None,
                    'name': 'NO_PICK',
                    'position': 'N/A',
                    'projected_points': 0.0,
                    'adp': 0.0,
                    'status': 'Skipped/Invalid Pick'
                })

            # Advance to the next pick in the draft order
            env.current_pick_idx += 1
            env.current_pick_number += 1

            # Check for done condition:
            # The draft is done if all picks in the order are exhausted OR
            # all teams have filled their rosters OR no players are left
            all_rosters_full = True
            for team_id_check in range(1, config.NUM_TEAMS + 1):
                if len(env.teams_rosters[team_id_check]['PLAYERS']) < env.total_roster_size_per_team:
                    all_rosters_full = False
                    break
            
            if all_rosters_full or env.current_pick_idx >= len(env.draft_order) or not env.available_players_ids:
                done = True
                
        # --- End of Simulation Run ---
        run_results = {
            'draft_log': draft_log,
            'final_team_rosters': {team_id: env.teams_rosters[team_id]['PLAYERS'] for team_id in range(1, config.NUM_TEAMS + 1)}
        }
        all_simulation_results.append(run_results)

    # --- Print Summary of Simulations ---
    print("\n--- Simulation Summary ---")
    avg_agent_scores = []
    
    # Store individual team scores for average across runs
    all_runs_team_scores = defaultdict(lambda: defaultdict(float)) # team_id -> {total, starters, bench, flex}

    for run_idx, result in enumerate(all_simulation_results):
        print(f"\nRun {run_idx + 1}:")
        
        # Display draft log
        print("\n  Draft Log:")
        for pick in result['draft_log']:
            # Ensure 'name' is handled for 'NO_PICK' scenarios
            player_name_display = pick['name'] if pick['name'] != 'NO_PICK' else pick['status']
            print(f"    Pick {pick['pick_number']:<3} | Team {pick['team_id']:<2} | {pick['position']:<3} {player_name_display:<25} | {pick['projected_points']:.1f} pts | ADP {pick['adp']:.1f} | {pick['status']}")

        # Calculate and display final team scores
        print("\n  Final Team Scores:")
        for team_id, players in result['final_team_rosters'].items():
            scores = calculate_roster_scores(players, config.ROSTER_STRUCTURE, config.BENCH_MAXES)
            print(f"    Team {team_id}: Roster Size={scores['roster_size']}/{env.total_roster_size_per_team} | Total Combined={scores['combined_total_points']:.1f} | Avg Combined={scores['combined_avg_points']:.1f}")
            print(f"      Starters: {scores['starters_total_points']:.1f} total, {scores['starters_avg_points']:.1f} avg")
            print(f"      Bench:    {scores['bench_total_points']:.1f} total, {scores['bench_avg_points']:.1f} avg")
            print(f"      FLEX:     {scores['flex_total_points']:.1f} total") # Display FLEX points explicitly

            # Aggregate scores for overall averages
            all_runs_team_scores[team_id]['combined_total'] += scores['combined_total_points']
            all_runs_team_scores[team_id]['starters_total'] += scores['starters_total_points']
            all_runs_team_scores[team_id]['bench_total'] += scores['bench_total_points']
            all_runs_team_scores[team_id]['flex_total'] += scores['flex_total_points']
            all_runs_team_scores[team_id]['roster_size_total'] += scores['roster_size']
            all_runs_team_scores[team_id]['count'] += 1

    print("\n--- Overall Averages Across All Runs ---")
    overall_avg_agent_score = 0
    overall_avg_opponent_score = 0
    
    for team_id in sorted(all_runs_team_scores.keys()):
        team_data = all_runs_team_scores[team_id]
        runs_count = team_data['count']
        
        avg_combined = team_data['combined_total'] / runs_count
        avg_starters = team_data['starters_total'] / runs_count
        avg_bench = team_data['bench_total'] / runs_count
        avg_flex = team_data['flex_total'] / runs_count
        avg_roster_size = team_data['roster_size_total'] / runs_count

        if team_id == config.AGENT_START_POSITION:
            overall_avg_agent_score = avg_combined
            print(f"Agent (Team {team_id}):")
            print(f"  Avg Roster Size: {avg_roster_size:.1f}/{env.total_roster_size_per_team}")
            print(f"  Avg Total Combined Points: {avg_combined:.1f}")
            print(f"  Avg Starters Points: {avg_starters:.1f}")
            print(f"  Avg Bench Points: {avg_bench:.1f}")
            print(f"  Avg FLEX Points: {avg_flex:.1f}")
        else:
            overall_avg_opponent_score += avg_combined
            print(f"Opponent Team {team_id}:")
            print(f"  Avg Roster Size: {avg_roster_size:.1f}/{env.total_roster_size_per_team}")
            print(f"  Avg Total Combined Points: {avg_combined:.1f}")
            print(f"  Avg Starters Points: {avg_starters:.1f}")
            print(f"  Avg Bench Points: {avg_bench:.1f}")
            print(f"  Avg FLEX Points: {avg_flex:.1f}")

    if config.NUM_TEAMS > 1: # Only calculate average opponent if there are opponents
        overall_avg_opponent_score /= (config.NUM_TEAMS - 1)
        print(f"\nOverall Average Opponent Total Score (across all opponent teams and runs): {overall_avg_opponent_score:.1f}")
    
    if overall_avg_agent_score > overall_avg_opponent_score:
        print("\nAgent's average score is HIGHER than opponents' average score. Good job!")
    elif overall_avg_agent_score < overall_avg_opponent_score:
        print("\nAgent's average score is LOWER than opponents' average score. Keep training!")
    else:
        print("\nAgent's average score is EQUAL to opponents' average score.")


if __name__ == "__main__":
    config = Config()
    simulate_drafts(config, config.NUM_SIMULATION_RUNS)