"""Season simulation service in simulator boundary."""

import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from draft_buddy.core import DraftState, PlayerCatalog
from draft_buddy.simulator.evaluator import simulate_season_fast

_TEAM_LABEL_PATTERN = re.compile(r"^Team (\d+)$")


def team_label(team_id: int) -> str:
    """Return the canonical UI team label for a team id."""
    return f"Team {team_id}"


def parse_team_id(label: str | None) -> Optional[int]:
    """Parse a team id from a ``Team {n}`` label."""
    if not label:
        return None
    match = _TEAM_LABEL_PATTERN.match(str(label).strip())
    if not match:
        return None
    return int(match.group(1))


def manager_to_label_map(team_manager_mapping: Dict[int, str]) -> Dict[str, str]:
    """Build manager-name to ``Team {id}`` lookup from team mapping."""
    return {
        manager_name: team_label(team_id)
        for team_id, manager_name in team_manager_mapping.items()
        if manager_name
    }


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
        label_by_manager = manager_to_label_map(team_manager_mapping)
        rosters: Dict[str, List[int]] = {}
        for team_id, team_roster in draft_state.team_rosters.items():
            label = label_by_manager.get(team_manager_mapping.get(team_id, ""))
            if label:
                rosters[label] = list(team_roster.player_ids)

        if weekly_projections is None:
            weekly_projections = player_catalog.to_weekly_projections()

        matchups_df = self._translate_matchup_labels(self._load_matchups(), label_by_manager)
        num_playoff_teams = int(self._config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6))
        regular_results, regular_records, playoff_results_df, playoffs_tree, winner = (
            simulate_season_fast(
                weekly_projections, matchups_df, rosters, 2025, "", False, num_playoff_teams
            )
        )
        winner_label = str(winner) if winner is not None else None
        return {
            "regular_season_records": self._format_regular_records(regular_records),
            "regular_season_matchups": self._format_regular_matchups(regular_results),
            "playoff_tree": playoffs_tree,
            "playoff_results": self._format_playoff_results(playoff_results_df),
            "winner": winner_label,
            "winner_team_id": parse_team_id(winner_label),
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

    def _translate_matchup_labels(
        self, matchups_df: pd.DataFrame, label_by_manager: Dict[str, str]
    ) -> pd.DataFrame:
        """Remap manager-name matchup columns to ``Team {id}`` labels."""
        if matchups_df.empty:
            return matchups_df
        translated = matchups_df.copy()
        for column in ("Away Manager(s)", "Home Manager(s)"):
            if column not in translated.columns:
                continue
            translated[column] = translated[column].apply(
                lambda value: label_by_manager.get(str(value), value)
                if not pd.isna(value)
                else value
            )
        return translated

    def _format_regular_records(self, regular_records: list) -> List[Dict[str, Any]]:
        """Convert standings tuples to JSON-ready team records."""
        formatted: List[Dict[str, Any]] = []
        for team_name, record in regular_records:
            label = str(team_name)
            formatted.append(
                {
                    "team_id": parse_team_id(label),
                    "team": label,
                    "W": record["W"],
                    "L": record["L"],
                    "T": record["T"],
                    "pts": record["pts"],
                }
            )
        return formatted

    def _format_regular_matchups(self, regular_results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert regular-season matchup dataframe to JSON-ready dictionaries."""
        if regular_results is None or regular_results.empty:
            return []
        matchups: List[Dict[str, Any]] = []
        for _, row in regular_results.iterrows():
            away_team = None if pd.isna(row["Away Manager(s)"]) else str(row["Away Manager(s)"])
            home_team = None if pd.isna(row["Home Manager(s)"]) else str(row["Home Manager(s)"])
            away_score = None if pd.isna(row["Away Score"]) else float(row["Away Score"])
            home_score = None if pd.isna(row["Home Score"]) else float(row["Home Score"])
            matchups.append(
                {
                    "week": int(row["Week"]),
                    "matchup": int(row["Matchup"]),
                    "away_team_id": parse_team_id(away_team),
                    "home_team_id": parse_team_id(home_team),
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_score": away_score,
                    "home_score": home_score,
                }
            )
        return matchups

    def _format_playoff_results(self, playoff_results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert playoff dataframe to JSON-ready dictionaries."""
        playoff_results: List[Dict[str, Any]] = []
        for _, row in playoff_results_df.iterrows():
            away_team = None if pd.isna(row["Away Manager(s)"]) else str(row["Away Manager(s)"])
            home_team = None if pd.isna(row["Home Manager(s)"]) else str(row["Home Manager(s)"])
            playoff_results.append(
                {
                    "week": int(row["Week"]),
                    "matchup": int(row["Matchup"]),
                    "away_team_id": parse_team_id(away_team),
                    "home_team_id": parse_team_id(home_team),
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_score": None if pd.isna(row["Away Score"]) else float(row["Away Score"]),
                    "home_score": None if pd.isna(row["Home Score"]) else float(row["Home Score"]),
                }
            )
        return playoff_results
