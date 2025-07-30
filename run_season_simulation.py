import datetime
import pandas as pd
import tqdm
import numpy as np

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from utils import player_data_utils, season_simulation_utils

def run_full_season_simulation(config: Config, num_simulations: int):
    """
    Runs a series of fantasy football draft and season simulations.
    """
    print(f"--- Starting {num_simulations} Full Season Simulations ---")

    # --- Data Setup ---
    # This part can be further refactored to be more aligned with the new config-driven approach
    season = 2024
    ps_start_year = 2021
    team_abbreviations = {
        "Detroit Lions": "DET", "Los Angeles Chargers": "LAC", "Philadelphia Eagles": "PHI",
        "Tennessee Titans": "TEN", "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA",
        "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "Chicago Bears": "CHI",
        "Dallas Cowboys": "DAL", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Cleveland Browns": "CLE", "Green Bay Packers": "GB", "Las Vegas Raiders": "LV",
        "Seattle Seahawks": "SEA", "Arizona Cardinals": "ARI", "Carolina Panthers": "CAR",
        "New York Giants": "NYG", "Tampa Bay Buccaneers": "TB", "Atlanta Falcons": "ATL",
        "Buffalo Bills": "BUF", "Cincinnati Bengals": "CIN", "Jacksonville Jaguars": "JAX",
        "New Orleans Saints": "NO", "New York Jets": "NYJ", "Baltimore Ravens": "BAL",
        "Denver Broncos": "DEN", "Houston Texans": "HOU", "Indianapolis Colts": "IND",
        "New England Patriots": "NE", "Washington Commanders": "WAS"
    }
    team_bye_weeks_2024 = {
        5: ["Detroit Lions", "Los Angeles Chargers", "Philadelphia Eagles", "Tennessee Titans"],
        6: ["Kansas City Chiefs", "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings"],
        7: ["Chicago Bears", "Dallas Cowboys"],
        9: ["Pittsburgh Steelers", "San Francisco 49ers"],
        10: ["Cleveland Browns", "Green Bay Packers", "Las Vegas Raiders", "Seattle Seahawks"],
        11: ["Arizona Cardinals", "Carolina Panthers", "New York Giants", "Tampa Bay Buccaneers"],
        12: ["Atlanta Falcons", "Buffalo Bills", "Cincinnati Bengals", "Jacksonville Jaguars", "New Orleans Saints", "New York Jets"],
        14: ["Baltimore Ravens", "Denver Broncos", "Houston Texans", "Indianapolis Colts", "New England Patriots", "Washington Commanders"]
    }
    bye_weeks_2024 = {}
    for week, team_list in team_bye_weeks_2024.items():
        for team in team_list:
            bye_weeks_2024[team_abbreviations[team]] = week

    # --- Get Player Data ---
    print("Fetching player data for simulation...")
    _, wtw_pts_dict = player_data_utils.get_simulation_dfs(
        season, ps_start_year, measure_of_center='median', custom_bye_weeks=bye_weeks_2024, custom_roster=season
    )
    print("Player data loaded.")

    # --- Matchup Data ---
    print("Loading matchup data...")
    try:
        matchups_df = pd.read_csv('data/red_league_matchups_2025.csv')
        # Basic preprocessing, this can be improved
        matchups_df = matchups_df.rename(columns={'Away Manager(s)': 'Away Team Manager(s)', 'Home Manager(s)': 'Home Team Manager(s)'})
        matchups_df['Away Team Manager(s)'] = matchups_df['Away Team Manager(s)'].apply(lambda x: x.split(' ')[0].lower())
        matchups_df['Home Team Manager(s)'] = matchups_df['Home Team Manager(s)'].apply(lambda x: x.split(' ')[0].lower())
        matchups_df = matchups_df.loc[matchups_df['Week'] < 15]
    except FileNotFoundError:
        print("ERROR: 'data/red_league_matchups_2025.csv' not found.")
        return

    draft_order_2024 = [
        'michael', 'paul', 'ryan', 'val', 'shane', 'noah', 'jake', 'scott', 'sean', 'lucas'
    ]
    draft_order_dict = {i + 1: name for i, name in enumerate(draft_order_2024)}

    # --- Simulation Loop ---
    win_dict = {name: 0 for name in draft_order_2024}
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
            rosters[draft_order_dict[team_id]] = [p.player_id for p in data['PLAYERS']]

        # 3. Simulate the season
        _, _, _, _, winner = season_simulation_utils.simulate_season(
            wtw_pts_dict,
            matchups_df.copy(deep=True), # Use a copy to avoid modification issues
            rosters,
            season,
            output_file_prefix=f'./simulated_data/sim_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
            save_data=False # Disable saving for now
        )
        if winner:
            win_dict[winner] += 1

    print("\n--- Simulation Complete ---")
    print("Win Distribution:")
    print(win_dict)

if __name__ == '__main__':
    config = Config()
    run_full_season_simulation(config, num_simulations=10)
