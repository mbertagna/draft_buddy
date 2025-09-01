import os
import pandas as pd
import numpy as np
import requests
import re
import unicodedata
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
                 rookie_projection_method: str = 'draft',
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
        # 'draft' -> scale by draft slot; 'adp' -> interpolate by ADP; 'hybrid' -> average both
        self.rookie_projection_method = rookie_projection_method
        self.rookie_projection_params = rookie_projection_params if rookie_projection_params is not None \
            else {'scale_min': 5, 'scale_max': 80, 'udfa_percentile': 75}
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

    def _calculate_games_played_frac(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the fraction of possible games a player has played in throughout their career.

        Args:
            historical_df (pd.DataFrame): The DataFrame with historical week-to-week stats.

        Returns:
            pd.DataFrame: A DataFrame with 'player_id' and the calculated 'games_played_frac'.
        """
        if historical_df.empty:
            return pd.DataFrame(columns=['player_id', 'games_played_frac'])

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
        games_played_frac = (total_player_games / total_team_games).reset_index(name='games_played_frac')
        
        return games_played_frac

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

    def _project_rookies_with_adp(self,
                                   draft_players_df: pd.DataFrame,
                                   adp_filepath: str,
                                   match_threshold: int = 85,
                                   adp_col_map: dict | None = None) -> pd.DataFrame:
        """
        Projects rookie fantasy points using ADP-based interpolation among nearby veteran neighbors.

        Steps:
        - Map ADP rows to roster using fuzzy matching (existing merge logic)
        - Within each position, sort by ADP (lower is better)
        - For each rookie with ADP, interpolate points from the closest veteran above and below by ADP
        - Fallbacks: if only one veteran neighbor, copy that neighbor's points; if none, use position median
        - If ADP missing for a rookie, leave as-is (caller can fallback)
        """
        if not adp_filepath:
            # Nothing to do without ADP
            return draft_players_df

        # Ensure we track which entries are rookies (no prior points pre-projection)
        draft_players_df = draft_players_df.copy()
        draft_players_df['is_rookie'] = draft_players_df['total_pts'].isna()

        merged_df, _, _ = self.merge_adp_data(
            computed_df=draft_players_df,
            adp_filepath=adp_filepath,
            match_threshold=match_threshold,
            adp_col_map=adp_col_map,
        )

        if merged_df.empty:
            return draft_players_df

        # Determine ADP numeric column
        adp_val_col = 'AVG' if 'AVG' in merged_df.columns else ('Rank' if 'Rank' in merged_df.columns else None)
        if adp_val_col is None:
            return draft_players_df

        # Use roster points for veterans
        merged_df = merged_df.copy()
        # Standardize column names used below
        pos_col = 'Pos' if 'Pos' in merged_df.columns else ('position' if 'position' in merged_df.columns else None)
        if pos_col is None:
            return draft_players_df

        # Prepare lookups for fallback medians by position
        veteran_mask = (merged_df['is_rookie'] == False) & merged_df['total_pts'].notna()
        pos_to_vet_median = merged_df[veteran_mask].groupby(pos_col)['total_pts'].median().to_dict()

        # Compute interpolated points for rookies
        predicted_points_by_player_id = {}

        for position_value, group in merged_df.groupby(pos_col):
            group_sorted = group.sort_values(by=adp_val_col, ascending=True).reset_index(drop=True)

            # Indices of veterans in this position
            vet_indices = group_sorted[(group_sorted['is_rookie'] == False) & group_sorted['total_pts'].notna()].index.tolist()
            if not vet_indices:
                continue

            vet_idx_set = set(vet_indices)

            for idx, row in group_sorted.iterrows():
                if not bool(row.get('is_rookie', False)):
                    continue

                adp_val = row.get(adp_val_col)
                if pd.isna(adp_val):
                    # Cannot interpolate without ADP
                    continue

                # Find nearest veteran above (lower ADP number) and below (higher ADP number)
                lower_idx = None
                upper_idx = None

                # Search downward for lower (better ADP)
                for i in range(idx - 1, -1, -1):
                    if i in vet_idx_set:
                        lower_idx = i
                        break

                # Search upward for upper (worse ADP)
                for j in range(idx + 1, len(group_sorted)):
                    if j in vet_idx_set:
                        upper_idx = j
                        break

                predicted = None
                if (lower_idx is not None) and (upper_idx is not None):
                    low_row = group_sorted.loc[lower_idx]
                    high_row = group_sorted.loc[upper_idx]
                    x0 = low_row[adp_val_col]
                    x1 = high_row[adp_val_col]
                    y0 = low_row['total_pts']
                    y1 = high_row['total_pts']
                    if pd.notna(x0) and pd.notna(x1) and (x1 > x0) and pd.notna(y0) and pd.notna(y1):
                        # Linear interpolation in ADP space
                        t = (adp_val - x0) / (x1 - x0)
                        predicted = y0 + t * (y1 - y0)
                elif lower_idx is not None:
                    predicted = group_sorted.loc[lower_idx]['total_pts']
                elif upper_idx is not None:
                    predicted = group_sorted.loc[upper_idx]['total_pts']

                if predicted is None or pd.isna(predicted):
                    # Fallback to position median of veterans
                    predicted = pos_to_vet_median.get(position_value, np.nan)

                # Store by player_id to merge back
                pid = row.get('player_id')
                if pd.notna(predicted) and pd.notna(pid):
                    predicted_points_by_player_id[pid] = float(predicted)

        if not predicted_points_by_player_id:
            return draft_players_df

        # Apply predictions back to rookies in the draft_players_df
        mask = draft_players_df['player_id'].isin(predicted_points_by_player_id.keys()) & draft_players_df['is_rookie']
        draft_players_df.loc[mask, 'total_pts'] = draft_players_df.loc[mask, 'player_id'].map(predicted_points_by_player_id)

        draft_players_df.drop(columns=['is_rookie'], inplace=True)
        return draft_players_df

    def process_draft_data(self,
                           draft_year: int,
                           measure_of_center: str = 'median',
                           adp_filepath: str | None = None,
                           adp_match_threshold: int = 85,
                           adp_col_map: dict | None = None) -> tuple:
        print("Fetching historical player data...")
        historical_df = self._get_player_stats_data(start_year=self.start_year, end_year=draft_year - 1)
        historical_df = self._apply_fantasy_scoring(historical_df)

        agg_func = 'median' if measure_of_center == 'median' else 'mean'
        legacy_stats_df = historical_df.groupby('player_id').agg(total_pts=('total_pts', agg_func)).reset_index()

        # Calculate games_played_frac and merge it
        fraction_df = self._calculate_games_played_frac(historical_df)
        legacy_stats_df = legacy_stats_df.merge(fraction_df, on='player_id', how='left')

        # Normalize legacy player_id to integer form (strip non-digits, drop empties)
        if 'player_id' in legacy_stats_df.columns:
            legacy_stats_df['player_id'] = (
                legacy_stats_df['player_id']
                .astype(str)
                .str.replace(r'[^0-9]', '', regex=True)
            )
            # Preserve rows even if empty; convert to pandas nullable integer
            legacy_stats_df['player_id'] = legacy_stats_df['player_id'].replace('', pd.NA).astype('Int64')

        print(f"Fetching player pool for {draft_year}...")
        
        if self.project_rookies:
            roster_url = f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{draft_year}.csv'
            draft_pool_df = self._download_data(f'roster_{draft_year}.csv', roster_url)
            draft_pool_df = draft_pool_df.rename(columns={'team': 'recent_team'})
            draft_pool_df['player_id'] = draft_pool_df.index.astype(int)
            # Normalize roster player_id to integer form for consistent merging
            if 'player_id' in draft_pool_df.columns:
                draft_pool_df['player_id'] = (
                    draft_pool_df['player_id']
                    .astype(str)
                    .str.replace(r'[^0-9]', '', regex=True)
                )
                draft_pool_df['player_id'] = draft_pool_df['player_id'].replace('', pd.NA).astype('Int64')
            
            # Standardize player name column
            draft_pool_df['player_display_name'] = draft_pool_df['full_name']
            
            draft_pool_df = draft_pool_df[draft_pool_df['position'].isin(self.positions)]
            draft_players_df = draft_pool_df.merge(legacy_stats_df, on='player_id', how='left')

            # Immediately assign new unique integer IDs to rookies (no legacy stats)
            if 'player_id' in draft_players_df.columns:
                rookie_mask_after_merge = draft_players_df['total_pts'].isna()
                if rookie_mask_after_merge.any():
                    # Determine the max existing legacy id to start sequencing
                    max_legacy_id = None
                    try:
                        max_legacy_id = int(pd.to_numeric(legacy_stats_df['player_id'], errors='coerce').max())
                    except Exception:
                        max_legacy_id = None
                    start_id = (max_legacy_id + 1) if max_legacy_id is not None and not pd.isna(max_legacy_id) else 1
                    num_new = int(rookie_mask_after_merge.sum())
                    if num_new > 0:
                        new_ids = pd.Series(
                            range(start_id, start_id + num_new),
                            index=draft_players_df[rookie_mask_after_merge].index,
                            dtype='Int64'
                        )
                        draft_players_df.loc[rookie_mask_after_merge, 'player_id'] = new_ids

            # Ensure uniqueness by player_id to avoid downstream duplication (after ID assignment)
            if 'player_id' in draft_players_df.columns:
                draft_players_df = draft_players_df.drop_duplicates(subset=['player_id'], keep='first')
            veterans_df = draft_players_df[draft_players_df['total_pts'].notna()].copy()
            rookies_df = draft_players_df[draft_players_df['total_pts'].isna()].copy()
            if not rookies_df.empty:
                print(f"Estimating points for {len(rookies_df)} rookies using method='{self.rookie_projection_method}'...")
                if self.rookie_projection_method == 'adp':
                    draft_players_df = self._project_rookies_with_adp(
                        draft_players_df=draft_players_df,
                        adp_filepath=adp_filepath,
                        match_threshold=adp_match_threshold,
                        adp_col_map=adp_col_map,
                    )
                    # If some rookies remain without projection, fill with draft-based as fallback
                    remaining_rookies = draft_players_df[draft_players_df['total_pts'].isna()].copy()
                    if not remaining_rookies.empty:
                        fallback_proj = self._estimate_rookie_points(remaining_rookies, veterans_df)
                        draft_players_df = pd.concat([
                            draft_players_df[draft_players_df['total_pts'].notna()],
                            fallback_proj
                        ], ignore_index=True)
                elif self.rookie_projection_method == 'hybrid':
                    # Compute both, then average
                    adp_version = self._project_rookies_with_adp(
                        draft_players_df=draft_players_df,
                        adp_filepath=adp_filepath,
                        match_threshold=adp_match_threshold,
                        adp_col_map=adp_col_map,
                    )
                    draft_based = pd.concat([
                        veterans_df,
                        self._estimate_rookie_points(rookies_df, veterans_df)
                    ], ignore_index=True)
                    # Average only for rookies present in both versions
                    merged_versions = draft_based[['player_id', 'total_pts']].merge(
                        adp_version[['player_id', 'total_pts']], on='player_id', how='outer', suffixes=('_draft', '_adp')
                    )
                    merged_versions['total_pts_final'] = merged_versions[['total_pts_draft', 'total_pts_adp']].mean(axis=1, skipna=True)
                    # Apply back to base frame
                    draft_players_df = draft_players_df.merge(
                        merged_versions[['player_id', 'total_pts_final']], on='player_id', how='left'
                    )
                    draft_players_df['total_pts'] = draft_players_df['total_pts'].combine_first(draft_players_df['total_pts_final'])
                    draft_players_df.drop(columns=['total_pts_final'], inplace=True)
                    # Fallbacks if still missing
                    remaining_rookies = draft_players_df[draft_players_df['total_pts'].isna()].copy()
                    if not remaining_rookies.empty:
                        fallback_proj = self._estimate_rookie_points(remaining_rookies, veterans_df)
                        draft_players_df = pd.concat([
                            draft_players_df[draft_players_df['total_pts'].notna()],
                            fallback_proj
                        ], ignore_index=True)
                else:
                    projected_rookies_df = self._estimate_rookie_points(rookies_df, veterans_df)
                    draft_players_df = pd.concat([veterans_df, projected_rookies_df], ignore_index=True)
        else:
            draft_year_df = self._get_player_stats_data(start_year=draft_year, end_year=draft_year)
            draft_year_df = self._apply_fantasy_scoring(draft_year_df)
            draft_pool_ids = draft_year_df[['player_id', 'player_display_name', 'position', 'recent_team']].drop_duplicates(subset=['player_id'])
            draft_players_df = draft_pool_ids.merge(legacy_stats_df, on='player_id', how='left')

        team_bye_weeks = self._get_team_bye_weeks()
        draft_players_df['bye_week'] = draft_players_df['recent_team'].map(team_bye_weeks)

        # For rookies (players without historical data), games_played_frac will be NaN.
        # Fill this with 'R' to signify they are a rookie.
        if 'games_played_frac' in draft_players_df.columns:
            draft_players_df['games_played_frac'].fillna('R', inplace=True)

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
            'total_pts', 'games_played_frac', 'bye_week'
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
            adp_df['Pos'] = adp_df['Pos'].astype(str)
            adp_df['Pos'] = adp_df['Pos'].str.extract(r'([A-Za-z]+)')
        if adp_col_map['Player'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['Player']: 'Player'}, inplace=True)

        def standardize_name(name):
            if pd.isna(name):
                return name
            s = str(name)
            # Normalize to a consistent Unicode form
            s = unicodedata.normalize('NFKC', s)
            # Remove any non-printable or zero-width characters (e.g., BOM)
            # Re-using the existing logic but being explicit about what's being removed
            s = ''.join(ch for ch in s if unicodedata.category(ch) not in ('Cf', 'Cc'))
            # Remove diacritics
            s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.category(ch).startswith('M'))
            # Case fold for robust lowercasing
            s = s.casefold()
            # Drop common suffixes at end (jr/sr/ii/iii/iv)
            s = re.sub(r'\b(jr|sr|iv|iii|ii)\b\.?$', '', s).strip()
            # Keep only letters, numbers, and whitespace. This is where the issue is.
            # Let's try to remove only what's absolutely necessary.
            # s = re.sub(r'[^a-z0-9\s]', '', s) # This line might be too aggressive
            # A more targeted approach would be to only replace specific known issues, or to
            # make sure the input is clean before this step.
            
            # Given your debugging, the issue is not with the name itself, but with a hidden
            # character or encoding problem. The best place to fix this is earlier.
            # Let's trust the `_clean_adp_content` to fix the source file.
            
            # Collapse spaces
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        computed_df['std_name'] = computed_df['player_display_name'].apply(standardize_name)
        adp_df['std_name'] = adp_df['Player'].apply(standardize_name)

        # Normalize position letters for safer comparisons and bonuses (e.g., WR8 -> WR)
        def _pos_base(val):
            if pd.isna(val):
                return None
            m = re.search(r'([A-Za-z]+)', str(val))
            return m.group(1).upper() if m else None
        if 'Pos' in adp_df.columns:
            adp_df['PosBase'] = adp_df['Pos'].apply(_pos_base)
        else:
            adp_df['PosBase'] = None
        if 'position' in computed_df.columns:
            computed_df['PosBase'] = computed_df['position'].astype(str).apply(_pos_base)
        else:
            computed_df['PosBase'] = None

        # Temporary diagnostics for specific problematic names (e.g., TreVeyon Henderson)
        def _debug_fuzzy_for_target(adp_df_local, computed_df_local, target_name='TreVeyon Henderson'):
            try:
                print("\n--- Debug: Fuzzy check for target name ---")
                # Collect raw candidates from both datasets
                adp_candidates = adp_df_local[adp_df_local['Player'].astype(str).str.contains(target_name, na=False, regex=False)]
                roster_candidates = computed_df_local[computed_df_local['player_display_name'].astype(str).str.contains(target_name, na=False, regex=False)]

                if adp_candidates.empty and roster_candidates.empty:
                    print(f"No raw candidates found for '{target_name}' in either dataset.")
                    return

                for _, a_row in adp_candidates.iterrows():
                    a_raw = a_row['Player']
                    a_std = standardize_name(a_raw)
                    print(f"ADP raw: {repr(a_raw)} | std: {repr(a_std)}")
                    for _, r_row in roster_candidates.iterrows():
                        r_raw = r_row['player_display_name']
                        r_std = standardize_name(r_raw)
                        name_score_raw = fuzz.token_sort_ratio(str(a_raw), str(r_raw))
                        name_score_std = fuzz.token_sort_ratio(a_std, r_std)
                        print(f"Roster raw: {repr(r_raw)} | std: {repr(r_std)} | fuzz_raw={name_score_raw} | fuzz_std={name_score_std}")
            except Exception as _e:
                # Non-fatal diagnostics
                print(f"Debug error: {_e}")

        _debug_fuzzy_for_target(adp_df, computed_df, target_name='TreVeyon Henderson')

        # Deduplicate ADP rows by standardized name and position, keeping best ADP (lowest numeric AVG/Rank)
        adp_rank_col = 'AVG' if 'AVG' in adp_df.columns else ('Rank' if 'Rank' in adp_df.columns else None)
        if adp_rank_col is not None:
            adp_df[adp_rank_col] = pd.to_numeric(adp_df[adp_rank_col], errors='coerce')
            # Sort by ADP ascending so best appears first
            adp_df = adp_df.sort_values(by=[adp_rank_col], ascending=True)
        # Use Team/Pos if available to reduce cross-position duplicates
        dedupe_keys = ['std_name']
        if 'Team' in adp_df.columns:
            dedupe_keys.append('Team')
        if 'Pos' in adp_df.columns:
            dedupe_keys.append('Pos')
        adp_df = adp_df.drop_duplicates(subset=dedupe_keys, keep='first').reset_index(drop=True)

        # --- 2. Perform Weighted Fuzzy Matching (with exact-match short-circuit) ---
        print("Performing fuzzy match with weighted scoring...")
        # Ensure computed_df is unique by player_id to prevent fan-out merges
        if 'player_id' in computed_df.columns:
            computed_df = computed_df.drop_duplicates(subset=['player_id'], keep='first')
        roster_choices = computed_df.to_dict('records')
        # Index roster by standardized name to short-circuit perfect matches
        stdname_to_roster = {}
        for rp in roster_choices:
            key = rp.get('std_name')
            if isinstance(key, str):
                stdname_to_roster.setdefault(key, []).append(rp)
        adp_df['matched_name'] = pd.NA
        adp_df['matched_player_id'] = pd.NA
        adp_df['match_score'] = 0

        for adp_idx, adp_row in adp_df.iterrows():
            if pd.isna(adp_row['std_name']): continue

            best_score = -1
            best_match_name = None
            best_match_player_id = None
            # Targeted debug for TreVeyon Henderson
            is_target = False
            try:
                is_target = (
                    str(adp_row.get('Player', '')) == 'TreVeyon Henderson' or
                    adp_row.get('std_name') == 'treveyon henderson'
                )
            except Exception:
                is_target = False
            target_candidates = [] if is_target else None

            # 2a) Deterministic: exact standardized-name match
            exact_candidates = stdname_to_roster.get(adp_row['std_name'], [])
            if exact_candidates:
                candidates = exact_candidates
                if len(candidates) > 1 and adp_row.get('PosBase') is not None:
                    filtered = [c for c in candidates if c.get('PosBase') == adp_row.get('PosBase')]
                    if len(filtered) == 1:
                        candidates = filtered
                if len(candidates) == 1:
                    rp = candidates[0]
                    team_bonus = 15 if pd.notna(adp_row.get('Team')) and (adp_row['Team'] == rp.get('recent_team')) else 0
                    pos_bonus = 5 if adp_row.get('PosBase') is not None and (adp_row.get('PosBase') == rp.get('PosBase')) else 0
                    best_score = min(100, 100 + team_bonus + pos_bonus)
                    best_match_name = rp.get('std_name')
                    best_match_player_id = rp.get('player_id')
                    if is_target:
                        try:
                            print("\n--- Debug (Exact-Match Short-Circuit): TreVeyon Henderson ---")
                            print(f"ADP: raw={repr(adp_row.get('Player'))}, std={repr(adp_row.get('std_name'))}, Team={adp_row.get('Team')}, PosBase={adp_row.get('PosBase')}")
                            print(f"Roster chosen: raw={repr(rp.get('player_display_name'))}, std={repr(rp.get('std_name'))}, Team={rp.get('recent_team')}, PosBase={rp.get('PosBase')}")
                            print(f"Bonuses: team_bonus={team_bonus}, pos_bonus={pos_bonus}; final_score={best_score}")
                        except Exception:
                            pass
                    adp_df.at[adp_idx, 'matched_name'] = best_match_name
                    adp_df.at[adp_idx, 'matched_player_id'] = best_match_player_id
                    adp_df.at[adp_idx, 'match_score'] = best_score
                    continue

            # 2b) Fuzzy fallback
            for roster_player in roster_choices:
                name_score = fuzz.token_sort_ratio(adp_row['std_name'], roster_player['std_name'])
                
                team_bonus = 15 if pd.notna(adp_row.get('Team')) and (adp_row['Team'] == roster_player.get('recent_team')) else 0
                pos_bonus = 5 if adp_row.get('PosBase') is not None and (adp_row.get('PosBase') == roster_player.get('PosBase')) else 0
                
                current_score = min(100, name_score + team_bonus + pos_bonus)

                if current_score > best_score:
                    best_score = current_score
                    best_match_name = roster_player['std_name']
                    best_match_player_id = roster_player.get('player_id')
                if is_target:
                    try:
                        target_candidates.append({
                            'roster_raw': roster_player.get('player_display_name'),
                            'roster_std': roster_player.get('std_name'),
                            'roster_team': roster_player.get('recent_team'),
                            'roster_posbase': roster_player.get('PosBase'),
                            'name_score': name_score,
                            'team_bonus': team_bonus,
                            'pos_bonus': pos_bonus,
                            'current_score': current_score,
                        })
                    except Exception:
                        pass
            
            if best_score >= match_threshold:
                adp_df.at[adp_idx, 'matched_name'] = best_match_name
                adp_df.at[adp_idx, 'matched_player_id'] = best_match_player_id
            adp_df.at[adp_idx, 'match_score'] = best_score
            if is_target and target_candidates is not None:
                try:
                    print("\n--- Debug (Fuzzy Details): TreVeyon Henderson ---")
                    print(f"ADP: raw={repr(adp_row.get('Player'))}, std={repr(adp_row.get('std_name'))}, Team={adp_row.get('Team')}, PosBase={adp_row.get('PosBase')}")
                    # Show top 10 by current_score
                    top = sorted(target_candidates, key=lambda x: x['current_score'], reverse=True)[:10]
                    for i, cand in enumerate(top, 1):
                        print(
                            f"{i:02d}) score={cand['current_score']} (name={cand['name_score']}, team={cand['team_bonus']}, pos={cand['pos_bonus']}) "
                            f"-> roster_raw={repr(cand['roster_raw'])}, roster_std={repr(cand['roster_std'])}, "
                            f"team={cand['roster_team']}, posbase={cand['roster_posbase']}"
                        )
                    print(f"Selected best_match_std={best_match_name}, best_score={best_score}")
                except Exception:
                    pass
        
        # --- 3. Merge and Create Final DataFrames ---
        matched_mask = adp_df['matched_player_id'].notna()
        # Merge by unique player_id to avoid duplication when names collide
        merge_left = adp_df[matched_mask].copy()
        # Keep only one ADP row per matched player_id, preferring best ADP (lowest AVG/Rank)
        adp_sort_col = None
        if 'AVG' in merge_left.columns:
            adp_sort_col = 'AVG'
        elif 'Rank' in merge_left.columns:
            adp_sort_col = 'Rank'
        if adp_sort_col is not None:
            merge_left[adp_sort_col] = pd.to_numeric(merge_left[adp_sort_col], errors='coerce')
            merge_left = merge_left.sort_values(by=adp_sort_col, ascending=True)
        merge_left = merge_left.drop_duplicates(subset=['matched_player_id'], keep='first')
        merged_df = pd.merge(
            merge_left, computed_df,
            left_on='matched_player_id', right_on='player_id',
            how='left', suffixes=('_adp', '_roster')
        )
        
        unmatched_df = adp_df[~matched_mask].sort_values('match_score', ascending=False)
        borderline_mask = (adp_df['match_score'] < match_threshold) & (adp_df['match_score'] >= match_threshold - 10)
        borderline_df = adp_df[borderline_mask].sort_values('match_score', ascending=False)

        # --- 4. Print Diagnostics ---
        print("\n--- ADP Merge Diagnostics ---")
        total_adp_rows = len(adp_df) if len(adp_df) > 0 else 1
        matched_count = int(matched_mask.sum())
        print(f"Total ADP Players: {total_adp_rows}")
        print(f"Successfully Matched: {matched_count} ({matched_count / total_adp_rows:.2%})")
        print(f"Borderline Cases ({match_threshold-10}-{match_threshold}): {len(borderline_df)}")
        print(f"Unmatched: {len(unmatched_df)}")
        
        print("\nUnmatched Players with Highest Potential Scores:")
        for _, row in unmatched_df.head(5).iterrows():
            print(f"- {row['Player']} (Team: {row.get('Team', 'N/A')}, Top Score: {row['match_score']:.0f})")

        # Extra: show highest ADP (worst) unmatched per position
        adp_val_col = 'AVG' if 'AVG' in adp_df.columns else ('Rank' if 'Rank' in adp_df.columns else None)
        pos_col = 'Pos' if 'Pos' in adp_df.columns else None
        if adp_val_col and pos_col and not unmatched_df.empty and adp_val_col in unmatched_df.columns:
            tmp = unmatched_df[[pos_col, 'Player', 'Team', adp_val_col]].copy()
            tmp[adp_val_col] = pd.to_numeric(tmp[adp_val_col], errors='coerce')
            tmp = tmp[pd.notna(tmp[adp_val_col])]
            if not tmp.empty:
                # Normalize position grouping (e.g., WR1 -> WR)
                tmp['PosBase'] = tmp[pos_col].astype(str).str.extract(r'([A-Za-z]+)')[0]
                print("\nUnmatched Highest-ADP per Position:")
                for position_value, g in tmp.groupby('PosBase'):
                    # Highest ADP is numerically largest
                    g_sorted = g.sort_values(by=adp_val_col, ascending=False)
                    r = g_sorted.iloc[0]
                    print(f"- {position_value}: {r['Player']} (Team: {r.get('Team', 'N/A')}, ADP: {r[adp_val_col]})")

        return merged_df, unmatched_df, borderline_df