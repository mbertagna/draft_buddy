"""
Entry point: fetch stats, score players, merge ADP, and write generated player CSV.

Run from repo root with PYTHONPATH including ``src`` (see docker-compose ``data-prep``).
"""

import argparse

import pandas as pd

from draft_buddy.config import Config
from draft_buddy.data_pipeline.data_processor import FantasyDataProcessor


def main(output_path, draft_year, rookie_projection_method):
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

    half_ppr_sleeper_scoring_rules = {
        # Passing
        "passing_yards": 0.04,          # 1 point per 25 yards
        "passing_touchdowns": 6,
        "interceptions": -3,
        "sack_fumbles_lost": -3,        # turnovers
        "two_point_conversions": 2,     # includes passing 2PCs if tracked

        # Rushing
        "rushing_yards": 0.1,           # 1 point per 10 yards
        "rushing_touchdowns": 6,
        "rushing_fumbles_lost": -3,

        # Receiving
        "receptions": 1,                # full PPR
        "receiving_yards": 0.1,
        "receiving_touchdowns": 6,
        "receiving_fumbles_lost": -3,

        # Kicking (no distance buckets — just flat yardage scoring)
        "fg_made_yards": 0.1,           # 1 point per 10 yards of FG distance
        "fg_missed": -1,
        "xp_made": 1,
        "xp_missed": -1,

        # Defense / Special Teams
        "sacks": 1,
        "def_interceptions": 2,
        "def_fumbles_recovered": 2,
        "safeties": 2,
        "def_touchdowns": 6,
        "kick_return_touchdowns": 6,
        "punt_return_touchdowns": 6,
        "blocked_kicks": 2,

        # Negative scoring
        "points_allowed": None,         # handled separately as a tiered rule
    }

    processor = FantasyDataProcessor(
        scoring_rules=half_ppr_sleeper_scoring_rules,
        project_rookies=True,
        bye_weeks_override=bye_weeks.get(draft_year, {}),
        start_year=draft_year - 2,
        positions=['QB', 'RB', 'WR', 'TE'],
        rookie_projection_method=rookie_projection_method,
    )

    # ADP file path for the given season
    adp_file = f'./data/FantasyPros_{draft_year}_Overall_ADP_Rankings.csv'

    computed_players_df, _ = processor.process_draft_data(
        draft_year=draft_year,
        adp_filepath=adp_file,
    )

    merged_df, unmatched_df, borderline_df = processor.merge_adp_data(
        computed_df=computed_players_df,
        adp_filepath=adp_file,
        match_threshold=85
    )

    # Normalize column names for outputs
    for df in [merged_df, unmatched_df, borderline_df]:
        if not df.empty:
            if 'AVG' in df.columns and 'adp' not in df.columns:
                df.rename(columns={'AVG': 'adp'}, inplace=True)
            # Ensure single record per player_id in output CSVs where relevant
            if 'player_id' in df.columns:
                df.drop_duplicates(subset=['player_id'], keep='first', inplace=True)

    merged_df = merged_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})
    unmatched_df = unmatched_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})
    borderline_df = borderline_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})

    # Extra diagnostic: unmatched highest-ADP per position (ADP worst values)
    if not unmatched_df.empty:
        adp_col = 'adp' if 'adp' in unmatched_df.columns else ('Rank' if 'Rank' in unmatched_df.columns else None)
        if adp_col and 'Pos' in unmatched_df.columns:
            tmp = unmatched_df[['Pos', 'Player', 'Team', adp_col]].copy()
            tmp[adp_col] = pd.to_numeric(tmp[adp_col], errors='coerce')
            tmp = tmp[pd.notna(tmp[adp_col])]
            if not tmp.empty:
                # Normalize positions like WR1 -> WR
                tmp['PosBase'] = tmp['Pos'].astype(str).str.extract(r'([A-Za-z]+)')[0]
                print("\nUnmatched Highest-ADP per Position:")
                for position_value, g in tmp.groupby('PosBase'):
                    r = g.sort_values(by=adp_col, ascending=False).iloc[0]
                    print(f"- {position_value}: {r['Player']} (Team: {r.get('Team', 'N/A')}, ADP: {r[adp_col]})")
    # ID normalization and assignment now handled inside FantasyDataProcessor.
    # No additional ID manipulation required here.

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
    parser.add_argument('--rookie_projection_method', type=str, default='draft', choices=['draft', 'adp', 'hybrid'],
                        help='Method to project rookie points: draft (slot scaling), adp (ADP interpolation), or hybrid (average).')

    args = parser.parse_args()

    config = Config()
    output_file_path = config.paths.PLAYER_DATA_CSV

    main(output_path=output_file_path, draft_year=args.year, rookie_projection_method=args.rookie_projection_method)
