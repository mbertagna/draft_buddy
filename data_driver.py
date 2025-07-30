import argparse
import pandas as pd
from utils.data_processor import FantasyDataProcessor
from config import Config

def main(output_path, draft_year):
    """
    Main function to run the data processing and merging pipeline.
    """
    pd.set_option('display.max_columns', None)

    print(f"--- Running Player Data Processor for {draft_year} Season ---")

    bye_weeks = {
        2024: {
            5: ['DET', 'LAC', 'PHI', 'TEN'],
            6: ['KC', 'LAR', 'MIA', 'MIN'],
            7: ['CHI', 'DAL'],
            9: ['PIT', 'SF'],
            10: ['CLE', 'GB', 'LV', 'SEA'],
            11: ['ARI', 'CAR', 'NYG', 'TB'],
            12: ['BUF', 'CIN', 'JAX', 'NYJ', 'HOU', 'DEN'],
            14: ['ATL', 'BAL', 'IND', 'NE', 'NO', 'WAS']
        },
        2025: {
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
    }

    processor = FantasyDataProcessor(
        project_rookies=True,
        bye_weeks_override=bye_weeks.get(draft_year, {}),
        start_year=draft_year - 2,
        positions=['QB', 'RB', 'WR', 'TE']
    )
    computed_players_df, _ = processor.process_draft_data(draft_year=draft_year)

    adp_file = f'./data/FantasyPros_{draft_year}_Overall_ADP_Rankings.csv'
    
    merged_df, unmatched_df, borderline_df = processor.merge_adp_data(
        computed_df=computed_players_df,
        adp_filepath=adp_file,
        match_threshold=85
    )

    merged_df = merged_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points', 'AVG': 'adp'})
    unmatched_df = unmatched_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points', 'AVG': 'adp'})
    borderline_df = borderline_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points', 'AVG': 'adp'})
    # Clean existing player_ids and assign new ones for missing
    def get_max_id(dfs):
        max_id = 0
        for df in dfs:
            if 'player_id' in df.columns:
                df['player_id'] = df['player_id'].str.replace(r'[^0-9]', '', regex=True)
                df_max = df['player_id'].astype(float).max()
                if pd.notna(df_max) and df_max > max_id:
                    max_id = df_max
        return int(max_id)

    next_id = get_max_id([merged_df, unmatched_df, borderline_df]) + 1

    # Assign new IDs to players missing them
    for df in [merged_df, unmatched_df, borderline_df]:
        if 'player_id' in df.columns:
            missing_mask = df['player_id'].isna()
            df.loc[missing_mask, 'player_id'] = [str(id).zfill(9) for id in range(next_id, next_id + missing_mask.sum())]
            next_id += missing_mask.sum()

    if not merged_df.empty:
        print(f"\n\n--- Successfully Merged Data for {draft_year} ---")
        display_cols = ['Rank', 'Player', 'Team', 'Pos', 'name', 'recent_team', 'projected_points', 'match_score', 'adp']
        print(merged_df[display_cols].head(10))
        
        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved final merged data to '{output_path}'")

    if not borderline_df.empty:
        borderline_path = './data/borderline_adp_matches.csv'
        borderline_df.to_csv(borderline_path, index=False)
        print(f"\n✅ Saved borderline cases to '{borderline_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process fantasy football data for a given year.')
    parser.add_argument('--year', type=int, default=2025, help='The draft year to process data for.')
    
    args = parser.parse_args()
    
    config = Config()
    output_file_path = config.PLAYER_DATA_CSV
    
    main(output_path=output_file_path, draft_year=args.year)