import os
from typing import Optional, Tuple

import pandas as pd

from scoring_service import ScoringService
from data_downloader import NflverseCsvDownloader
from rookie_projector import RookieProjector
from adp_matcher import AdpMatcher

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

class FantasyDataProcessor:
    """
    Orchestrates the fantasy football data pipeline.

    Delegates network I/O, scoring, and rookie projection to injected services.
    """

    def __init__(
        self,
        scoring_rules: dict = None,
        positions: list = None,
        cache_dir: str = "./data",
        bye_weeks_override: dict = None,
        project_rookies: bool = True,
        rookie_projection_method: str = "draft",
        rookie_projection_params: dict = None,
        start_year: int = 1999,
        data_downloader=None,
        scoring_service: Optional[ScoringService] = None,
        rookie_projector: Optional[RookieProjector] = None,
        adp_matcher: Optional[AdpMatcher] = None,
    ):
        """
        Initialize the processor with configuration and optional service injections.

        Parameters
        ----------
        scoring_rules : dict, optional
            Fantasy scoring rules.
        positions : list, optional
            Positions to include.
        cache_dir : str, optional
            Cache directory for downloads.
        bye_weeks_override : dict, optional
            Bye week data {week: [teams]}.
        project_rookies : bool, optional
            Whether to project rookies.
        rookie_projection_method : str, optional
            'draft', 'adp', or 'hybrid'.
        rookie_projection_params : dict, optional
            Parameters for rookie projection.
        start_year : int, optional
            First year of historical data.
        data_downloader : DataDownloader, optional
            Injected downloader. Defaults to NflverseCsvDownloader.
        scoring_service : ScoringService, optional
            Injected scoring service. Defaults to ScoringService(scoring_rules).
        rookie_projector : RookieProjector, optional
            Injected rookie projector. Defaults to RookieProjector from params.
        adp_matcher : AdpMatcher, optional
            Injected ADP matcher for fuzzy matching. Defaults to AdpMatcher().
        """
        self.scoring_rules = scoring_rules if scoring_rules is not None else DEFAULT_SCORING_RULES
        self.positions = positions if positions is not None else ["QB", "RB", "WR", "TE", "K"]
        self.cache_dir = cache_dir
        self.bye_weeks_override = bye_weeks_override
        self.project_rookies = project_rookies
        self.rookie_projection_method = rookie_projection_method
        self.rookie_projection_params = rookie_projection_params or {
            "scale_min": 5,
            "scale_max": 80,
            "udfa_percentile": 75,
        }
        self.start_year = start_year

        self._downloader = data_downloader or NflverseCsvDownloader(cache_dir)
        self._scoring_service = scoring_service or ScoringService(self.scoring_rules)
        self._adp_matcher = adp_matcher or AdpMatcher()
        rp_params = self.rookie_projection_params
        self._rookie_projector = rookie_projector or RookieProjector(
            scale_min=rp_params.get("scale_min", 5),
            scale_max=rp_params.get("scale_max", 80),
            udfa_percentile=rp_params.get("udfa_percentile", 75),
            adp_matcher=self._adp_matcher,
        )

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

    def process_draft_data(self,
                           draft_year: int,
                           measure_of_center: str = 'median',
                           adp_filepath: str | None = None,
                           adp_match_threshold: int = 85,
                           adp_col_map: dict | None = None) -> tuple:
        """
        Orchestrate the data pipeline: fetch, score, aggregate, merge, project rookies.

        Parameters
        ----------
        draft_year : int
            The draft year.
        measure_of_center : str, optional
            'median' or 'mean' for legacy stats aggregation.
        adp_filepath : str, optional
            Path to ADP CSV for adp/hybrid rookie projection.
        adp_match_threshold : int, optional
            Fuzzy match threshold for ADP.
        adp_col_map : dict, optional
            ADP column mapping.

        Returns
        -------
        tuple
            (draft_players_df, weekly_projections).
        """
        print("Fetching player pool and historical data...")
        draft_pool_df, legacy_stats_df, draft_year_stats_df = self._downloader.fetch_player_pool(
            draft_year=draft_year,
            positions=self.positions,
            start_year=self.start_year,
            end_year=draft_year,
        )

        scored_historical = self._scoring_service.apply_scoring(legacy_stats_df)
        legacy_stats_df = self._scoring_service.aggregate_legacy_stats(
            scored_historical, measure_of_center
        )

        if self.project_rookies:
            draft_players_df = self._scoring_service.merge_roster_with_legacy(
                draft_pool_df, legacy_stats_df
            )
            rookies_df = draft_players_df[draft_players_df['is_rookie_original'] == True]
            if not rookies_df.empty:
                print(f"Estimating points for {len(rookies_df)} rookies using method='{self.rookie_projection_method}'...")
                draft_players_df = self._rookie_projector.project_rookies(
                    draft_players_df,
                    method=self.rookie_projection_method,
                    adp_filepath=adp_filepath,
                    match_threshold=adp_match_threshold,
                    adp_col_map=adp_col_map,
                )
        else:
            draft_year_scored = self._scoring_service.apply_scoring(draft_year_stats_df)
            draft_players_df = self._scoring_service.merge_draft_year_with_legacy(
                draft_year_scored, legacy_stats_df
            )

        team_bye_weeks = self._get_team_bye_weeks()
        draft_players_df = draft_players_df.copy()
        draft_players_df['bye_week'] = draft_players_df['recent_team'].map(team_bye_weeks)

        draft_players_df = self._scoring_service.apply_rookie_metadata(draft_players_df)
        weekly_projections = self._scoring_service.generate_weekly_projections(draft_players_df)
        draft_players_df = self._scoring_service.finalize_draft_players(draft_players_df)

        print("Processing complete.")
        return draft_players_df, weekly_projections

    def merge_adp_data(self,
                       computed_df: pd.DataFrame,
                       adp_filepath: str,
                       match_threshold: int = 85,
                       adp_col_map: dict = None) -> tuple:
        """
        Merges external ADP data with the computed player data using a weighted fuzzy matching score.

        Delegates to the injected AdpMatcher service.

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
        return self._adp_matcher.merge_adp_data(
            computed_df=computed_df,
            adp_filepath=adp_filepath,
            match_threshold=match_threshold,
            adp_col_map=adp_col_map,
        )