"""
ADP matcher service for fuzzy-matching external ADP data to roster players.

Encapsulates ADP file loading, content cleaning, name standardization,
and weighted fuzzy matching logic.
"""

import re
import unicodedata
from io import StringIO
from typing import Optional, Tuple

import pandas as pd
from fuzzywuzzy import fuzz


def _standardize_name(name) -> Optional[str]:
    """
    Normalize a player name for consistent fuzzy matching.

    Parameters
    ----------
    name : str or Any
        Raw player name (may contain diacritics, suffixes, etc.).

    Returns
    -------
    str or None
        Standardized name string, or None if input is NaN.
    """
    if pd.isna(name):
        return None
    s = str(name)
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) not in ('Cf', 'Cc'))
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.category(ch).startswith('M'))
    s = s.casefold()
    s = re.sub(r'\b(jr|sr|iv|iii|ii)\b\.?$', '', s).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


class AdpMatcher:
    """
    Merges external ADP data with computed player data using weighted fuzzy matching.
    """

    def _clean_adp_content(self, content: str) -> str:
        """
        Apply regex fixes to raw ADP file content.

        Parameters
        ----------
        content : str
            Raw file content from the ADP CSV.

        Returns
        -------
        str
            Cleaned content suitable for CSV parsing.
        """
        content = re.sub(r'","N","12 O","', '","NO","12","', content)
        content = re.sub(r'","","\'', '\'', content)
        content = re.sub(r'","III","', ' III","",', content)
        content = re.sub(r'","II","', ' II","",', content)
        content = re.sub(r'","Gay","', ' Gay","","', content)
        content = re.sub(r'","Ali","', ' Ali","","', content)
        content = re.sub(r'","HU","14 O","', '","HOU","14","', content)
        return content

    def merge_adp_data(
        self,
        computed_df: pd.DataFrame,
        adp_filepath: str,
        match_threshold: int = 85,
        adp_col_map: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Merge external ADP data with computed player data using weighted fuzzy matching.

        Parameters
        ----------
        computed_df : pd.DataFrame
            Player roster with columns player_id, player_display_name, position, recent_team, etc.
        adp_filepath : str
            Path to the ADP CSV file.
        match_threshold : int, optional
            Minimum fuzzy match score (0-100) to accept a match. Default 85.
        adp_col_map : dict, optional
            Column mapping for ADP file: {'Player': str, 'Team': str, 'POS': str}.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (merged_df, unmatched_df, borderline_df).
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

        if adp_col_map['Team'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['Team']: 'Team'}, inplace=True)
            adp_df['Team'] = adp_df['Team'].replace({'JAC': 'JAX', 'LA': 'LAR'})
        if adp_col_map['POS'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['POS']: 'Pos'}, inplace=True)
            adp_df['Pos'] = adp_df['Pos'].astype(str)
            adp_df['Pos'] = adp_df['Pos'].str.extract(r'([A-Za-z]+)')
        if adp_col_map['Player'] in adp_df.columns:
            adp_df.rename(columns={adp_col_map['Player']: 'Player'}, inplace=True)

        computed_df = computed_df.copy()
        if 'player_id' not in computed_df.columns:
            computed_df['player_id'] = computed_df.index.astype(int)
        computed_df['std_name'] = computed_df['player_display_name'].apply(_standardize_name)
        adp_df['std_name'] = adp_df['Player'].apply(_standardize_name)

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

        adp_rank_col = 'AVG' if 'AVG' in adp_df.columns else ('Rank' if 'Rank' in adp_df.columns else None)
        if adp_rank_col is not None:
            adp_df[adp_rank_col] = pd.to_numeric(adp_df[adp_rank_col], errors='coerce')
            adp_df = adp_df.sort_values(by=[adp_rank_col], ascending=True)
        dedupe_keys = ['std_name']
        if 'Team' in adp_df.columns:
            dedupe_keys.append('Team')
        if 'Pos' in adp_df.columns:
            dedupe_keys.append('Pos')
        adp_df = adp_df.drop_duplicates(subset=dedupe_keys, keep='first').reset_index(drop=True)

        print("Performing fuzzy match with weighted scoring...")
        if 'player_id' in computed_df.columns:
            computed_df = computed_df.drop_duplicates(subset=['player_id'], keep='first')
        roster_choices = computed_df.to_dict('records')
        stdname_to_roster = {}
        for rp in roster_choices:
            key = rp.get('std_name')
            if isinstance(key, str):
                stdname_to_roster.setdefault(key, []).append(rp)
        adp_df['matched_name'] = pd.NA
        adp_df['matched_player_id'] = pd.NA
        adp_df['match_score'] = 0

        for adp_idx, adp_row in adp_df.iterrows():
            if pd.isna(adp_row['std_name']):
                continue

            best_score = -1
            best_match_name = None
            best_match_player_id = None

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
                    adp_df.at[adp_idx, 'matched_name'] = best_match_name
                    adp_df.at[adp_idx, 'matched_player_id'] = best_match_player_id
                    adp_df.at[adp_idx, 'match_score'] = best_score
                    continue

            for roster_player in roster_choices:
                name_score = fuzz.token_sort_ratio(adp_row['std_name'], roster_player['std_name'])
                team_bonus = 15 if pd.notna(adp_row.get('Team')) and (adp_row['Team'] == roster_player.get('recent_team')) else 0
                pos_bonus = 5 if adp_row.get('PosBase') is not None and (adp_row.get('PosBase') == roster_player.get('PosBase')) else 0
                current_score = min(100, name_score + team_bonus + pos_bonus)

                if current_score > best_score:
                    best_score = current_score
                    best_match_name = roster_player['std_name']
                    best_match_player_id = roster_player.get('player_id')

            if best_score >= match_threshold:
                adp_df.at[adp_idx, 'matched_name'] = best_match_name
                adp_df.at[adp_idx, 'matched_player_id'] = best_match_player_id
            adp_df.at[adp_idx, 'match_score'] = best_score

        matched_mask = adp_df['matched_player_id'].notna()
        merge_left = adp_df[matched_mask].copy()
        adp_sort_col = 'AVG' if 'AVG' in merge_left.columns else ('Rank' if 'Rank' in merge_left.columns else None)
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

        adp_val_col = 'AVG' if 'AVG' in adp_df.columns else ('Rank' if 'Rank' in adp_df.columns else None)
        pos_col = 'Pos' if 'Pos' in adp_df.columns else None
        if adp_val_col and pos_col and not unmatched_df.empty and adp_val_col in unmatched_df.columns:
            tmp = unmatched_df[[pos_col, 'Player', 'Team', adp_val_col]].copy()
            tmp[adp_val_col] = pd.to_numeric(tmp[adp_val_col], errors='coerce')
            tmp = tmp[pd.notna(tmp[adp_val_col])]
            if not tmp.empty:
                tmp['PosBase'] = tmp[pos_col].astype(str).str.extract(r'([A-Za-z]+)')[0]
                print("\nUnmatched Highest-ADP per Position:")
                for position_value, g in tmp.groupby('PosBase'):
                    g_sorted = g.sort_values(by=adp_val_col, ascending=False)
                    r = g_sorted.iloc[0]
                    print(f"- {position_value}: {r['Player']} (Team: {r.get('Team', 'N/A')}, ADP: {r[adp_val_col]})")

        return merged_df, unmatched_df, borderline_df
