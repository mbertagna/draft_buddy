"""
Utilities for loading player data for season simulations.

Provides get_simulation_dfs which fetches draft data and converts it to
the format expected by simulate_season_fast.
"""

from typing import Dict, Tuple

import pandas as pd

from utils.data_processor import FantasyDataProcessor


def _convert_to_simulation_format(weekly_projections: Dict) -> Dict:
    """
    Convert data_processor output format to simulate_season_fast format.

    data_processor returns: {player_id: {'position': str, 1: pts, 2: pts, ... 18: pts}}
    simulate_season_fast expects: {player_id: {'pos': str, 'pts': List[float]}}

    Parameters
    ----------
    weekly_projections : dict
        Output from FantasyDataProcessor.process_draft_data (second element).

    Returns
    -------
    dict
        Format compatible with simulate_season_fast.
    """
    result = {}
    for player_id, data in weekly_projections.items():
        pos = data.get('position', data.get('pos', 'WR'))
        pts_list = [data.get(w, 0) for w in range(1, 19)]
        result[player_id] = {'pos': pos, 'pts': pts_list}
    return result


def get_simulation_dfs(
    season: int,
    ps_start_year: int,
    measure_of_center: str = 'median',
    custom_bye_weeks: Dict = None,
    custom_roster: int = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch player data for season simulation.

    Parameters
    ----------
    season : int
        Draft/season year.
    ps_start_year : int
        First year of historical stats.
    measure_of_center : str, optional
        'median' or 'mean' for aggregating legacy stats.
    custom_bye_weeks : dict, optional
        Mapping of team abbreviation to bye week.
    custom_roster : int, optional
        Unused; kept for API compatibility.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (draft_players_df, weekly_projections) where weekly_projections
        is in the format expected by simulate_season_fast.
    """
    bye_weeks = custom_bye_weeks or {}
    adp_file = f'./data/FantasyPros_{season}_Overall_ADP_Rankings.csv'

    processor = FantasyDataProcessor(
        project_rookies=True,
        bye_weeks_override=bye_weeks,
        start_year=ps_start_year,
        positions=['QB', 'RB', 'WR', 'TE'],
        rookie_projection_method='draft',
    )

    draft_players_df, weekly_projections_raw = processor.process_draft_data(
        draft_year=season,
        measure_of_center=measure_of_center,
        adp_filepath=adp_file,
    )

    weekly_projections = _convert_to_simulation_format(weekly_projections_raw)
    return draft_players_df, weekly_projections
