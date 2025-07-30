from utils.data_processor import FantasyDataProcessor
import pandas as pd

pd.set_option('display.max_columns', None)

# --- Step 1: Generate your core player data ---
print("--- Running Player Data Processor for 2025 Season ---")

bye_weeks_2025 = {
    5: ["PIT", "CHI", "GB", "ATL"],
    6: ["HOU", "MIN"],
    7: ["BAL", "BUF"],
    8: ["JAX", "LV", "DET", "ARI", "SEA", "LA"],
    9: ["PHI", "CLE", "NYJ", "TB"],
    10: ["KC", "CIN", "TEN", "DAL"],
    11: ["IND", "NO"],
    12: ["MIA", "DEN", "LAC", "WAS"],
    14: ["NYG", "NE", "CAR", "SF"]
}

# Initialize the processor for a historical analysis of the 2025 season
processor = FantasyDataProcessor(
    project_rookies=True,
    bye_weeks_override=bye_weeks_2025,
    start_year=2023 # Use data from 2023-2025 for career averages
)
computed_players_df, _ = processor.process_draft_data(draft_year=2025)

print(computed_players_df.head(10))

# --- Step 2: Merge the ADP data using the new method ---
adp_file = './data/FantasyPros_2025_Overall_ADP_Rankings.csv'

# The processor instance can be reused for other tasks
merged_df, unmatched_df, borderline_df = processor.merge_adp_data(
    computed_df=computed_players_df,
    adp_filepath=adp_file,
    match_threshold=85
)

# --- Step 3: Analyze the results ---
if not merged_df.empty:
    print("\n\n--- Successfully Merged Data (Top 10) ---")
    display_cols = ['Rank', 'Player', 'Team', 'Pos', 'AVG', 'player_display_name', 'recent_team', 'total_pts', 'match_score']
    print(merged_df[display_cols].head(10))
    
    merged_df.to_csv('./data/final_data_with_adp.csv', index=False)
    print("\n✅ Saved final merged data to 'final_data_with_adp.csv'")

if not borderline_df.empty:
    print("\n\n--- Borderline Cases for Manual Review ---")
    print(borderline_df[['Player', 'Team', 'Pos', 'match_score']].head(10))
    
    borderline_df.to_csv('./data/borderline_adp_matches.csv', index=False)
    print("\n✅ Saved borderline cases to 'borderline_adp_matches.csv'")