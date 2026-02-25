import os
import pandas as pd
from typing import Dict, Any, List, Optional
from draft_buddy.utils.season_simulation_fast import simulate_season_fast

class SeasonSimulationService:
    """
    Business logic for running full-season simulations.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : Config
            Application configuration.
        """
        self._config = config

    def simulate_season(self, draft_env) -> Dict[str, Any]:
        """
        Runs a full season simulation using the current draft environment state.

        Parameters
        ----------
        draft_env : FantasyFootballDraftEnv
            The draft environment with current rosters and projections.

        Returns
        -------
        dict
            Structured simulation results including records, tree, and results.
        """
        rosters = {}
        for team_id, roster_data in draft_env.teams_rosters.items():
            manager_name = draft_env.team_manager_mapping.get(team_id)
            if not manager_name:
                continue
            rosters[manager_name] = [p.player_id for p in roster_data['PLAYERS']]

        matchups_df = self._load_matchups()
        
        # Week-to-week points. Prefer env's prepared projections if available
        weekly_projections = getattr(draft_env, 'weekly_projections', None)
        if weekly_projections is None:
            weekly_projections = {
                p.player_id: {'pts': [p.projected_points] * 18, 'pos': p.position} 
                for p in draft_env.all_players_data
            }

        num_playoff_teams = int(self._config.reward.REGULAR_SEASON_REWARD.get('NUM_PLAYOFF_TEAMS', 6))
        
        _, regular_records, playoff_results_df, playoffs_tree, winner = simulate_season_fast(
            weekly_projections, matchups_df, rosters, 2025, '', False, num_playoff_teams
        )

        return {
            'regular_season_records': regular_records,
            'playoff_tree': playoffs_tree,
            'playoff_results': self._format_playoff_results(playoff_results_df),
            'winner': winner
        }

    def _load_matchups(self) -> pd.DataFrame:
        """Loads the appropriate matchups CSV based on team count."""
        default_matchups_filename = 'red_league_matchups_2025.csv'
        size_specific_filename = f"red_league_matchups_2025_{self._config.draft.NUM_TEAMS}_team.csv"
        
        candidates = [
            os.path.join(self._config.paths.DATA_DIR, size_specific_filename),
            os.path.join(self._config.paths.DATA_DIR, default_matchups_filename),
        ]
        
        matchups_path = None
        for p in candidates:
            if os.path.exists(p):
                matchups_path = p
                break
        
        if matchups_path is None:
            matchups_path = os.path.join(self._config.paths.DATA_DIR, default_matchups_filename)
            
        return pd.read_csv(matchups_path)

    def _format_playoff_results(self, playoff_results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converts playoff results DataFrame to a UI-friendly list of dicts."""
        playoff_results = []
        try:
            for _, row in playoff_results_df.iterrows():
                playoff_results.append({
                    'week': int(row['Week']),
                    'matchup': int(row['Matchup']),
                    'away_manager': None if pd.isna(row['Away Manager(s)']) else str(row['Away Manager(s)']),
                    'away_score': None if pd.isna(row['Away Score']) else float(row['Away Score']),
                    'home_manager': None if pd.isna(row['Home Manager(s)']) else str(row['Home Manager(s)']),
                    'home_score': None if pd.isna(row['Home Score']) else float(row['Home Score'])
                })
        except Exception:
            pass
        return playoff_results
