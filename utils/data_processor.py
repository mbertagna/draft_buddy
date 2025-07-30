import os
import pandas as pd
import numpy as np
import requests
import re
from fuzzywuzzy import process, fuzz
from io import StringIO

# Default scoring rules, can be overridden during class initialization
DEFAULT_SCORING_RULES = {
    "passing_yards": 0.04,
    "passing_tds": 4, # Changed from 6 to a more standard 4
    "interceptions": -2,
    "passing_2pt_conversions": 2,
    'passing_yards_300_399_game': 2,
    'passing_yards_400_plus_game': 6,
    "receiving_yards": 0.1,
    "receptions": 1, # PPR default
    "receiving_tds": 6,
    "receiving_2pt_conversions": 2,
    'receiving_yards_100_199_game': 3,
    'receiving_yards_200_plus_game': 6,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "rushing_2pt_conversions": 2,
    'rushing_yards_100_199_game': 3,
    'rushing_yards_200_plus_game': 6,
    "pat_made": 1,
    "fg_missed": -1,
    "fg_made_0_39": 3,
    "fg_made_40_49": 4,
    "fg_made_50_59": 5,
    "fg_made_60_": 6,
    "total_fumbles_lost": -2,
}

def download_file_from_link(url: str, file_path: str, chunk_size=8192):
    """
    Downloads a file from a URL and saves it to a local path.
    Includes a simple progress indicator.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                total_size = int(r.headers.get('content-length', 0))
                print(f"Downloading {os.path.basename(file_path)} ({total_size/1e6:.2f} MB)...")
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        print(f"Successfully downloaded to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        raise

class FantasyDataProcessor:
    """
    A class to process nflverse data for fantasy football analysis.
    """

    def __init__(self,
                 scoring_rules: dict = None,
                 positions: list = None,
                 cache_dir: str = './data',
                 bye_weeks_override: dict = None,
                 project_rookies: bool = True,
                 rookie_projection_params: dict = None,
                 start_year: int = 1999):
        """
        Initializes the FantasyDataProcessor with configuration settings.

        Args:
            scoring_rules (dict, optional): Dictionary defining fantasy scoring. Defaults to a standard PPR format.
            positions (list, optional): List of player positions to include. Defaults to ['QB', 'RB', 'WR', 'TE', 'K'].
            cache_dir (str, optional): Directory to cache downloaded data. Defaults to './data'.
            bye_weeks_override (dict, optional): The ONLY source for bye week data. Must be in the format
                                                 {week_num: [teams]}. If not provided, bye weeks will be empty.
            project_rookies (bool, optional): Whether to use current rosters and project rookies, or use historical data.
            rookie_projection_params (dict, optional): Parameters for rookie projection.
            start_year (int, optional): The first year of historical data to consider for player averages.
        """
        self.scoring_rules = scoring_rules if scoring_rules is not None else DEFAULT_SCORING_RULES
        self.positions = positions if positions is not None else ['QB', 'RB', 'WR', 'TE', 'K']
        self.cache_dir = cache_dir
        self.bye_weeks_override = bye_weeks_override
        self.project_rookies = project_rookies
        self.rookie_projection_params = rookie_projection_params if rookie_projection_params is not None \
            else {'scale_min': 5, 'scale_max': 95, 'udfa_percentile': 75}
        self.start_year = start_year

        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_team_bye_weeks(self) -> dict:
        """
        Processes the user-provided bye week dictionary. This is the only source for bye weeks.
        Returns a dictionary mapping a team abbreviation to its bye week.
        """
        if not self.bye_weeks_override:
            print("Warning: No 'bye_weeks_override' data provided. 'bye_week' column will be empty.")
            return {}

        print("Processing provided bye week data.")
        # Invert the dictionary from {week: [teams]} to {team: week} for easy mapping
        inverted_byes = {team: week for week, teams in self.bye_weeks_override.items() for team in teams}
        return inverted_byes

    def _download_data(self, file_name: str, url: str) -> pd.DataFrame:
        """
        Downloads a data file if it doesn't exist in the cache, then loads it into a DataFrame.
        """
        file_path = os.path.join(self.cache_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            download_file_from_link(url, file_path)
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def _get_player_stats_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Fetches, merges, and prepares the core player statistics data.
        """
        # Download and load main stats
        stats_url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv'
        ps_df = self._download_data('player_stats.csv', stats_url)

        # Download and load kicking stats
        kicking_url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_kicking.csv'
        psk_df = self._download_data('player_stats_kicking.csv', kicking_url)
        
        # Merge the two dataframes
        merge_cols = list(set(psk_df.columns).intersection(set(ps_df.columns)))
        merged_df = ps_df.merge(psk_df, how='outer', on=merge_cols)

        # Filter by year and position
        merged_df = merged_df[merged_df['season'].between(start_year, end_year)]
        merged_df = merged_df[merged_df['position'].isin(self.positions)]

        return merged_df.reset_index(drop=True)

    def _apply_fantasy_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the configured fantasy scoring rules to the DataFrame.
        """
        # Create composite stats needed for scoring
        df['passing_yards_300_399_game'] = df['passing_yards'].between(300, 399).astype(int)
        df['passing_yards_400_plus_game'] = (df['passing_yards'] >= 400).astype(int)
        df['receiving_yards_100_199_game'] = df['receiving_yards'].between(100, 199).astype(int)
        df['receiving_yards_200_plus_game'] = (df['receiving_yards'] >= 200).astype(int)
        df['rushing_yards_100_199_game'] = df['rushing_yards'].between(100, 199).astype(int)
        df['rushing_yards_200_plus_game'] = (df['rushing_yards'] >= 200).astype(int)
        df['total_fumbles_lost'] = df['sack_fumbles_lost'] + df['rushing_fumbles_lost'] + df['receiving_fumbles_lost']
        df['fg_made_0_39'] = df['fg_made_0_19'] + df['fg_made_20_29'] + df['fg_made_30_39']

        # Calculate total points
        df['total_pts'] = 0
        for stat, points in self.scoring_rules.items():
            if stat in df.columns:
                df['total_pts'] += df[stat].fillna(0) * points
        
        return df

    def _calculate_fraction_career_in(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the fraction of possible games a player has played in throughout their career.

        Args:
            historical_df (pd.DataFrame): The DataFrame with historical week-to-week stats.

        Returns:
            pd.DataFrame: A DataFrame with 'player_id' and the calculated 'fraction_career_in'.
        """
        if historical_df.empty:
            return pd.DataFrame(columns=['player_id', 'fraction_career_in'])

        # Calculate the number of games each team played in each season
        team_games_per_season = historical_df.groupby(['season', 'recent_team'])['week'].nunique().reset_index()
        team_games_per_season.rename(columns={'week': 'num_team_games'}, inplace=True)

        # Get the seasons and teams for each player
        player_seasons = historical_df[['player_id', 'season', 'recent_team']].drop_duplicates()
        
        # Merge to find the total number of team games for each player's seasons
        player_team_games = player_seasons.merge(team_games_per_season, on=['season', 'recent_team'], how='left')
        total_team_games = player_team_games.groupby('player_id')['num_team_games'].sum()

        # Get the total number of games each player actually played
        total_player_games = historical_df.groupby('player_id').size()

        # Calculate the fraction
        fraction_career_in = (total_player_games / total_team_games).reset_index(name='fraction_career_in')
        
        return fraction_career_in

    def _estimate_rookie_points(self, rookies_df: pd.DataFrame, veterans_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates fantasy points for rookies based on their draft number and position.

        Args:
            rookies_df (pd.DataFrame): DataFrame containing rookies with missing point values.
            veterans_df (pd.DataFrame): DataFrame containing veterans with existing point values.

        Returns:
            pd.DataFrame: The rookies DataFrame with estimated 'total_pts'.
        """
        projected_rookies_list = []

        # Get the scaling parameters from the class configuration
        scale_min = self.rookie_projection_params['scale_min']
        scale_max = self.rookie_projection_params['scale_max']
        udfa_percentile = self.rookie_projection_params['udfa_percentile']

        for position in rookies_df['position'].unique():
            pos_rookies_df = rookies_df[rookies_df['position'] == position].copy()
            pos_veterans_df = veterans_df[veterans_df['position'] == position]

            if pos_veterans_df.empty:
                # If there are no veterans for this position, we can't project. Assign 0.
                pos_rookies_df['total_pts'] = 0
                projected_rookies_list.append(pos_rookies_df)
                continue

            # Get the point distribution for veterans at this position
            vet_pts_dist = pos_veterans_df['total_pts'].values

            # --- Scaling Logic ---
            draft_numbers = pos_rookies_df['draft_number'].dropna()
            
            min_draft_num, max_draft_num = (0, 0)
            if not draft_numbers.empty:
                min_draft_num = draft_numbers.min()
                max_draft_num = draft_numbers.max()

            def scale_draft_number_to_percentile(x):
                if pd.isna(x):
                    # For undrafted free agents (UDFAs), use the default percentile
                    return udfa_percentile
                
                # If there's no range of draft numbers (e.g., only one drafted player),
                # assign the highest percentile.
                if max_draft_num == min_draft_num:
                    return 100 - scale_min

                # Scale draft number: a lower draft number (e.g., 1) is better (higher percentile)
                # We subtract from 100 because np.percentile(dist, 95) gives a top value.
                scaled_val = (((scale_max - scale_min) * (x - min_draft_num)) / (max_draft_num - min_draft_num)) + scale_min
                return 100 - scaled_val

            pos_rookies_df['draft_percentile'] = pos_rookies_df['draft_number'].apply(scale_draft_number_to_percentile)
            
            # --- Point Projection ---
            # Use the calculated percentile to get a point value from the veteran distribution
            pos_rookies_df['total_pts'] = pos_rookies_df['draft_percentile'].apply(
                lambda p: np.percentile(vet_pts_dist, p)
            )
            
            projected_rookies_list.append(pos_rookies_df)

        return pd.concat(projected_rookies_list, ignore_index=True) if projected_rookies_list else pd.DataFrame()

    def process_draft_data(self, draft_year: int, measure_of_center: str = 'median') -> tuple:
        print("Fetching historical player data...")
        historical_df = self._get_player_stats_data(start_year=self.start_year, end_year=draft_year - 1)
        historical_df = self._apply_fantasy_scoring(historical_df)

        agg_func = 'median' if measure_of_center == 'median' else 'mean'
        legacy_stats_df = historical_df.groupby('player_id').agg(total_pts=('total_pts', agg_func)).reset_index()

        # Calculate fraction_career_in and merge it
        fraction_df = self._calculate_fraction_career_in(historical_df)
        legacy_stats_df = legacy_stats_df.merge(fraction_df, on='player_id', how='left')

        print(f"Fetching player pool for {draft_year}...")
        
        if self.project_rookies:
            roster_url = f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{draft_year}.csv'
            draft_pool_df = self._download_data(f'roster_{draft_year}.csv', roster_url)
            draft_pool_df = draft_pool_df.rename(columns={'gsis_id': 'player_id', 'team': 'recent_team'})
            
            # Standardize player name column
            draft_pool_df['player_display_name'] = draft_pool_df['full_name']
            
            draft_pool_df = draft_pool_df[draft_pool_df['position'].isin(self.positions)]
            draft_players_df = draft_pool_df.merge(legacy_stats_df, on='player_id', how='left')
            veterans_df = draft_players_df[draft_players_df['total_pts'].notna()].copy()
            rookies_df = draft_players_df[draft_players_df['total_pts'].isna()].copy()
            if not rookies_df.empty:
                print(f"Estimating points for {len(rookies_df)} rookies...")
                projected_rookies_df = self._estimate_rookie_points(rookies_df, veterans_df)
                draft_players_df = pd.concat([veterans_df, projected_rookies_df], ignore_index=True)
        else:
            draft_year_df = self._get_player_stats_data(start_year=draft_year, end_year=draft_year)
            draft_year_df = self._apply_fantasy_scoring(draft_year_df)
            draft_pool_ids = draft_year_df[['player_id', 'player_display_name', 'position', 'recent_team']].drop_duplicates(subset=['player_id'])
            draft_players_df = draft_pool_ids.merge(legacy_stats_df, on='player_id', how='left')

        team_bye_weeks = self._get_team_bye_weeks()
        draft_players_df['bye_week'] = draft_players_df['recent_team'].map(team_bye_weeks)

        wtw_pts_dict = {}
        for _, row in draft_players_df.iterrows():
            player_id = row['player_id']
            avg_pts = row['total_pts'] if pd.notna(row['total_pts']) else 0 
            bye = row['bye_week']
            wtw_pts_dict[player_id] = {'position': row['position']}
            for week in range(1, 19):
                wtw_pts_dict[player_id][week] = 0 if week == bye else avg_pts

        draft_players_df = draft_players_df.sort_values(by='total_pts', ascending=False).reset_index(drop=True)
        final_cols = [
            'player_id', 'player_display_name', 'position', 'recent_team', 
            'total_pts', 'fraction_career_in', 'bye_week'
        ]
        final_cols_exist = [col for col in final_cols if col in draft_players_df.columns]
        draft_players_df = draft_players_df[final_cols_exist]

        print("Processing complete.")
        return draft_players_df, wtw_pts_dict

    def _clean_adp_content(self, content: str) -> str:
        """Applies specific regex fixes to raw ADP file content."""
        # This function isolates the brittle cleaning steps.
        content = re.sub(r'","N","12 O","', '","NO","12","', content)
        content = re.sub(r'","","\'', '\'', content)
        content = re.sub(r'","III","', ' III","",', content)
        content = re.sub(r'","II","', ' II","",', content)
        content = re.sub(r'","Gay","', ' Gay","","', content)
        content = re.sub(r'","Ali","', ' Ali","","', content)
        content = re.sub(r'","HU","14 O","', '","HOU","14","', content)
        return content

    def merge_adp_data(self,
                       computed_df: pd.DataFrame,
                       adp_filepath: str,
                       match_threshold: int = 85,
                       adp_col_map: dict = None) -> tuple:
        """
        Merges external ADP data with the computed player data using a weighted fuzzy matching score.
        ... (docstring is the same) ...
        """
        if adp_col_map is None:
            adp_col_map = {'Player': 'Player', 'Team': 'Team', 'POS': 'POS'}

        print(f"Loading and cleaning ADP data from {adp_filepath}...")
        try:
            with open(adp_filepath, 'r', encoding='utf-8') as file:
                content = self._clean_adp_content(file.read())
            adp_df = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"Error loading ADP file: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # --- 1. Standardize DataFrames ---
        if adp_col_map['Team'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['Team']: 'Team'}, inplace=True)
            adp_df['Team'] = adp_df['Team'].replace({'JAC': 'JAX', 'LA': 'LAR'})
        if adp_col_map['POS'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['POS']: 'Pos'}, inplace=True)
            adp_df['Pos'] = adp_df['Pos'].str.replace('RB', 'RB').replace('WR', 'WR').replace('TE', 'TE').replace('QB', 'QB')
        if adp_col_map['Player'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['Player']: 'Player'}, inplace=True)

        def standardize_name(name):
            if pd.isna(name): return name
            name = re.sub(r'\s+(jr|sr|iv|iii|ii)\.?$', '', name.lower())
            name = re.sub(r'[^a-z0-9\s]', '', name)
            return re.sub(r'\s+', ' ', name).strip()

        computed_df['std_name'] = computed_df['player_display_name'].apply(standardize_name)
        adp_df['std_name'] = adp_df['Player'].apply(standardize_name)

        # --- 2. Perform Weighted Fuzzy Matching (Manual Loop) ---
        print("Performing fuzzy match with weighted scoring...")
        roster_choices = computed_df.to_dict('records')
        adp_df['matched_name'] = pd.NA
        adp_df['match_score'] = 0

        for adp_idx, adp_row in adp_df.iterrows():
            if pd.isna(adp_row['std_name']): continue

            best_score = -1
            best_match_name = None

            for roster_player in roster_choices:
                name_score = fuzz.token_sort_ratio(adp_row['std_name'], roster_player['std_name'])
                
                team_bonus = 15 if pd.notna(adp_row.get('Team')) and (adp_row['Team'] == roster_player['recent_team']) else 0
                pos_bonus = 5 if pd.notna(adp_row.get('Pos')) and (adp_row['Pos'] in roster_player['position']) else 0
                
                current_score = min(100, name_score + team_bonus + pos_bonus)

                if current_score > best_score:
                    best_score = current_score
                    best_match_name = roster_player['std_name']
            
            if best_score >= match_threshold:
                adp_df.at[adp_idx, 'matched_name'] = best_match_name
            adp_df.at[adp_idx, 'match_score'] = best_score
        
        # --- 3. Merge and Create Final DataFrames ---
        matched_mask = adp_df['matched_name'].notna()
        merged_df = pd.merge(
            adp_df[matched_mask], computed_df,
            left_on='matched_name', right_on='std_name',
            how='left', suffixes=('_adp', '_roster')
        )
        
        unmatched_df = adp_df[~matched_mask].sort_values('match_score', ascending=False)
        borderline_mask = (adp_df['match_score'] < match_threshold) & (adp_df['match_score'] >= match_threshold - 10)
        borderline_df = adp_df[borderline_mask].sort_values('match_score', ascending=False)

        # --- 4. Print Diagnostics ---
        print("\n--- ADP Merge Diagnostics ---")
        print(f"Total ADP Players: {len(adp_df)}")
        print(f"Successfully Matched: {len(merged_df)} ({len(merged_df) / len(adp_df):.2%})")
        print(f"Borderline Cases ({match_threshold-10}-{match_threshold}): {len(borderline_df)}")
        print(f"Unmatched: {len(unmatched_df)}")
        
        print("\nUnmatched Players with Highest Potential Scores:")
        for _, row in unmatched_df.head(5).iterrows():
            print(f"- {row['Player']} (Team: {row.get('Team', 'N/A')}, Top Score: {row['match_score']:.0f})")

        return merged_df, unmatched_df, borderline_df