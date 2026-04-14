import datetime
import os
import pandas as pd
import tqdm
import numpy as np

from draft_buddy.config import Config
from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
from draft_buddy.data_pipeline import get_simulation_dfs
from draft_buddy.utils.season_simulation_fast import simulate_season_fast

def run_full_season_simulation(config: Config, num_simulations: int):
    """
    Runs a series of fantasy football draft and season simulations.
    """
    print(f"--- Starting {num_simulations} Full Season Simulations ---")

    # --- Data Setup ---
    season = 2024
    ps_start_year = 2021
    
    # Use centralized configuration data
    team_abbreviations = config.draft.TEAM_ABBREVIATIONS
    team_bye_weeks_2024 = config.draft.TEAM_BYE_WEEKS_2024
    
    bye_weeks_2024 = {}
    for week, team_list in team_bye_weeks_2024.items():
        for team in team_list:
            abbr = team_abbreviations.get(team)
            if abbr:
                bye_weeks_2024[abbr] = week

    # --- Get Player Data ---
    print("Fetching player data for simulation...")
    _, weekly_projections = get_simulation_dfs(
        season, ps_start_year, measure_of_center='median', custom_bye_weeks=bye_weeks_2024, custom_roster=season
    )
    print("Player data loaded.")

    # --- Matchup Data ---
    print("Loading matchup data...")
    # Prefer size-specific matchup file if available
    default_matchups_filename = 'red_league_matchups_2025.csv'
    size_specific_filename = f"red_league_matchups_2025_{config.draft.NUM_TEAMS}_team.csv"
    candidates = [f"data/{size_specific_filename}", f"data/{default_matchups_filename}"]
    matchups_path = None
    for p in candidates:
        if os.path.exists(p):
            matchups_path = p
            break
    if matchups_path is None:
        matchups_path = f"data/{default_matchups_filename}"
    try:
        matchups_df = pd.read_csv(matchups_path)
        # Basic preprocessing, this can be improved
        matchups_df = matchups_df.loc[matchups_df['Week'] < 15]
    except FileNotFoundError:
        print(f"ERROR: Matchups file not found at {matchups_path}.")
        return

    # Use manager mapping from config for flexible team sizes
    draft_order_dict = config.draft.TEAM_MANAGER_MAPPING

    # --- Simulation Loop ---
    win_dict = {name: 0 for name in draft_order_dict.values()}
    draft_env = FantasyFootballDraftEnv(config)

    for _ in tqdm.tqdm(range(num_simulations)):
        # 1. Simulate a draft
        draft_env.reset()
        done = False
        while not done:
            if draft_env.current_pick_idx >= len(draft_env.draft_order):
                done = True
                continue

            current_team_id = draft_env.draft_order[draft_env.current_pick_idx]
            
            # In a full simulation, we'd use the agent or heuristics.
            # For now, we'll just use the simulation logic from the environment.
            if current_team_id == draft_env.agent_team_id:
                # In a real scenario, you might use the loaded agent model to make a pick
                # For this simulation, we can let the environment's internal logic handle it
                # or use a simplified approach.
                action_mask = draft_env.get_action_mask()
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break # No valid actions
                action = np.random.choice(valid_actions)
                _, _, step_done, _, _ = draft_env.step(action)
                if step_done:
                    done = True
            else:
                draft_env.simulate_single_pick()

            if draft_env.current_pick_idx >= len(draft_env.draft_order):
                done = True

        # 2. Prepare rosters for season simulation
        rosters = {}
        for team_id, data in draft_env.teams_rosters.items():
            manager_name = draft_order_dict.get(team_id)
            if manager_name:
                rosters[manager_name] = [p.player_id for p in data['PLAYERS']]

        # 3. Simulate the season
        _, _, _, _, winner = simulate_season_fast(
            weekly_projections,
            matchups_df.copy(deep=True), # Use a copy to avoid modification issues
            rosters,
            season,
            output_file_prefix=f'./simulated_data/sim_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
            save_data=False, # Disable saving for now
            num_playoff_teams=int(getattr(config, 'REGULAR_SEASON_REWARD', {}).get('NUM_PLAYOFF_TEAMS', 6))
        )
        if winner:
            win_dict[winner] += 1

    print("\n--- Simulation Complete ---")
    print("Win Distribution:")
    print(win_dict)

if __name__ == '__main__':
    config = Config()
    run_full_season_simulation(config, num_simulations=10)
