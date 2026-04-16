"""Season simulation service in simulator boundary."""

import os
from typing import Any, Dict, List

import pandas as pd

from draft_buddy.core import DraftState, PlayerCatalog
from draft_buddy.simulator.evaluator import simulate_season_fast


class SeasonSimulationService:
    """Business service for running season simulations."""

    def __init__(self, config):
        """Initialize service with runtime config.

        Parameters
        ----------
        config : Config
            Application configuration object.
        """
        self._config = config

    def simulate_season(
        self,
        draft_state: DraftState,
        player_catalog: PlayerCatalog,
        team_manager_mapping: Dict[int, str],
        weekly_projections: Dict[int, Dict[str, object]] | None = None,
    ) -> Dict[str, Any]:
        """Simulate a season using explicit draft inputs.

        Parameters
        ----------
        draft_state : DraftState
            Current draft state.
        player_catalog : PlayerCatalog
            Shared player catalog.
        team_manager_mapping : Dict[int, str]
            Team id to manager name mapping.
        weekly_projections : Dict[int, Dict[str, object]], optional
            Weekly projection map. When omitted, uses season projection repeats.

        Returns
        -------
        Dict[str, Any]
            Structured simulation results.
        """
        rosters = {}
        for team_id, team_roster in draft_state.team_rosters.items():
            manager_name = team_manager_mapping.get(team_id)
            if manager_name:
                rosters[manager_name] = list(team_roster.player_ids)

        if weekly_projections is None:
            weekly_projections = player_catalog.to_weekly_projections()

        matchups_df = self._load_matchups()
        num_playoff_teams = int(self._config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6))
        _, regular_records, playoff_results_df, playoffs_tree, winner = simulate_season_fast(
            weekly_projections, matchups_df, rosters, 2025, "", False, num_playoff_teams
        )
        return {
            "regular_season_records": regular_records,
            "playoff_tree": playoffs_tree,
            "playoff_results": self._format_playoff_results(playoff_results_df),
            "winner": winner,
        }

    def _load_matchups(self) -> pd.DataFrame:
        """Load matchup schedule for configured team count."""
        default_matchups_filename = "red_league_matchups_2025.csv"
        size_specific_filename = f"red_league_matchups_2025_{self._config.draft.NUM_TEAMS}_team.csv"
        candidates = [
            os.path.join(self._config.paths.DATA_DIR, size_specific_filename),
            os.path.join(self._config.paths.DATA_DIR, default_matchups_filename),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return pd.read_csv(candidate)
        return pd.read_csv(os.path.join(self._config.paths.DATA_DIR, default_matchups_filename))

    def _format_playoff_results(self, playoff_results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert playoff dataframe to JSON-ready dictionaries."""
        playoff_results: List[Dict[str, Any]] = []
        for _, row in playoff_results_df.iterrows():
            playoff_results.append(
                {
                    "week": int(row["Week"]),
                    "matchup": int(row["Matchup"]),
                    "away_manager": None
                    if pd.isna(row["Away Manager(s)"])
                    else str(row["Away Manager(s)"]),
                    "away_score": None if pd.isna(row["Away Score"]) else float(row["Away Score"]),
                    "home_manager": None
                    if pd.isna(row["Home Manager(s)"])
                    else str(row["Home Manager(s)"]),
                    "home_score": None if pd.isna(row["Home Score"]) else float(row["Home Score"]),
                }
            )
        return playoff_results
