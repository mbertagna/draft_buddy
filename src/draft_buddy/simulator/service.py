"""Season simulation service in simulator boundary."""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from draft_buddy.core import DraftState, PlayerCatalog
from draft_buddy.simulator.evaluator import generate_round_robin_schedule, simulate_season_fast
from draft_buddy.simulator.team_identity import (
    display_name_to_team_label_map,
    parse_team_id,
    team_label,
    team_labels_for_league,
)

__all__ = [
    "SeasonSimulationService",
    "display_name_to_team_label_map",
    "manager_to_label_map",
    "parse_team_id",
    "team_label",
]


class SeasonSimulationService:
    """Business service for running season simulations by team id."""

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
        team_display_names: Dict[int, str] | None = None,
        weekly_projections: Dict[int, Dict[str, object]] | None = None,
    ) -> Dict[str, Any]:
        """Simulate a season using team-id identity throughout compute.

        Parameters
        ----------
        draft_state : DraftState
            Current draft state.
        player_catalog : PlayerCatalog
            Shared player catalog.
        team_display_names : Dict[int, str], optional
            Cosmetic id-to-name map. Used only to translate legacy name-keyed
            matchup CSVs and to decorate API response labels.
        weekly_projections : Dict[int, Dict[str, object]], optional
            Weekly projection map. When omitted, uses season projection repeats.

        Returns
        -------
        Dict[str, Any]
            Structured simulation results keyed by team id, with optional
            display names for UI.
        """
        display_names = dict(team_display_names or {})
        rosters = self._build_team_label_rosters(draft_state)
        if weekly_projections is None:
            weekly_projections = player_catalog.to_weekly_projections()

        matchups_df = self.resolve_matchups(display_names)
        num_playoff_teams = int(self._config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6))
        regular_results, regular_records, playoff_results_df, playoffs_tree, winner = (
            simulate_season_fast(
                weekly_projections, matchups_df, rosters, 2025, "", False, num_playoff_teams
            )
        )
        winner_label = str(winner) if winner is not None else None
        winner_team_id = parse_team_id(winner_label)
        return {
            "regular_season_records": self._format_regular_records(
                regular_records, display_names
            ),
            "regular_season_matchups": self._format_regular_matchups(
                regular_results, display_names
            ),
            "playoff_tree": playoffs_tree,
            "playoff_results": self._format_playoff_results(
                playoff_results_df, display_names
            ),
            "winner": self._display_team_name(winner_team_id, winner_label, display_names),
            "winner_team_id": winner_team_id,
        }

    def resolve_matchups(
        self, team_display_names: Dict[int, str] | None = None
    ) -> pd.DataFrame:
        """Return a schedule whose identity columns use ``Team {id}`` labels.

        Parameters
        ----------
        team_display_names : Dict[int, str], optional
            Cosmetic names used only when translating legacy CSV schedules.

        Returns
        -------
        pd.DataFrame
            Matchup schedule keyed by compute team labels.
        """
        team_labels = team_labels_for_league(self._config.draft.NUM_TEAMS)
        if self._config.reward.USE_RANDOM_MATCHUPS:
            num_weeks = int(self._config.reward.NUM_REGULAR_SEASON_WEEKS)
            return generate_round_robin_schedule(team_labels, num_weeks)
        label_by_name = display_name_to_team_label_map(team_display_names or {})
        return self._translate_matchup_labels(self._load_matchups(), label_by_name)

    def _build_team_label_rosters(self, draft_state: DraftState) -> Dict[str, List[int]]:
        """Build ``Team {id}``-keyed roster map for every configured team."""
        return {
            team_label(team_id): list(draft_state.roster_for_team(team_id).player_ids)
            for team_id in range(1, self._config.draft.NUM_TEAMS + 1)
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
        return pd.DataFrame()

    def _translate_matchup_labels(
        self, matchups_df: pd.DataFrame, label_by_name: Dict[str, str]
    ) -> pd.DataFrame:
        """Remap legacy schedule name columns to ``Team {id}`` labels.

        Translates ``Away Manager(s)`` / ``Home Manager(s)`` using cosmetic
        display names. When those cells remain unrecognized, falls back to
        ``Away Team`` / ``Home Team`` nickname columns.
        """
        if matchups_df.empty:
            return matchups_df
        translated = matchups_df.copy()

        def _remap(value: object) -> object:
            if pd.isna(value):
                return value
            text = str(value)
            return label_by_name.get(text, text)

        for column in ("Away Manager(s)", "Home Manager(s)"):
            if column not in translated.columns:
                continue
            translated[column] = translated[column].apply(_remap)

        team_column_pairs = (
            ("Away Manager(s)", "Away Team"),
            ("Home Manager(s)", "Home Team"),
        )
        for manager_column, team_column in team_column_pairs:
            if manager_column not in translated.columns or team_column not in translated.columns:
                continue
            unresolved = ~translated[manager_column].astype(str).str.match(r"^Team \d+$", na=False)
            translated.loc[unresolved, manager_column] = translated.loc[
                unresolved, team_column
            ].map(_remap)
        return translated

    def _display_team_name(
        self,
        team_id: Optional[int],
        fallback_label: str | None,
        display_names: Dict[int, str],
    ) -> Optional[str]:
        """Return a cosmetic team name for API responses."""
        if team_id is not None and display_names.get(team_id):
            return display_names[team_id]
        return fallback_label

    def _format_regular_records(
        self, regular_records: list, display_names: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """Convert standings tuples to JSON-ready team records."""
        formatted: List[Dict[str, Any]] = []
        for team_name, record in regular_records:
            label = str(team_name)
            team_id = parse_team_id(label)
            formatted.append(
                {
                    "team_id": team_id,
                    "team": self._display_team_name(team_id, label, display_names),
                    "W": record["W"],
                    "L": record["L"],
                    "T": record["T"],
                    "pts": record["pts"],
                }
            )
        return formatted

    def _format_regular_matchups(
        self, regular_results: pd.DataFrame, display_names: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """Convert regular-season matchup dataframe to JSON-ready dictionaries."""
        if regular_results is None or regular_results.empty:
            return []
        matchups: List[Dict[str, Any]] = []
        for _, row in regular_results.iterrows():
            away_label = None if pd.isna(row["Away Manager(s)"]) else str(row["Away Manager(s)"])
            home_label = None if pd.isna(row["Home Manager(s)"]) else str(row["Home Manager(s)"])
            away_team_id = parse_team_id(away_label)
            home_team_id = parse_team_id(home_label)
            away_score = None if pd.isna(row["Away Score"]) else float(row["Away Score"])
            home_score = None if pd.isna(row["Home Score"]) else float(row["Home Score"])
            matchups.append(
                {
                    "week": int(row["Week"]),
                    "matchup": int(row["Matchup"]),
                    "away_team_id": away_team_id,
                    "home_team_id": home_team_id,
                    "away_team": self._display_team_name(away_team_id, away_label, display_names),
                    "home_team": self._display_team_name(home_team_id, home_label, display_names),
                    "away_score": away_score,
                    "home_score": home_score,
                }
            )
        return matchups

    def _format_playoff_results(
        self, playoff_results_df: pd.DataFrame, display_names: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """Convert playoff dataframe to JSON-ready dictionaries."""
        playoff_results: List[Dict[str, Any]] = []
        for _, row in playoff_results_df.iterrows():
            away_label = None if pd.isna(row["Away Manager(s)"]) else str(row["Away Manager(s)"])
            home_label = None if pd.isna(row["Home Manager(s)"]) else str(row["Home Manager(s)"])
            away_team_id = parse_team_id(away_label)
            home_team_id = parse_team_id(home_label)
            playoff_results.append(
                {
                    "week": int(row["Week"]),
                    "matchup": int(row["Matchup"]),
                    "away_team_id": away_team_id,
                    "home_team_id": home_team_id,
                    "away_team": self._display_team_name(away_team_id, away_label, display_names),
                    "home_team": self._display_team_name(home_team_id, home_label, display_names),
                    "away_score": None if pd.isna(row["Away Score"]) else float(row["Away Score"]),
                    "home_score": None if pd.isna(row["Home Score"]) else float(row["Home Score"]),
                }
            )
        return playoff_results


# Backwards-compatible aliases for older imports/tests.
manager_to_label_map = display_name_to_team_label_map
