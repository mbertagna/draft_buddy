"""Select distinct NFL teams for team-level insight enrichment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from draft_buddy.data.insights.team_context import InsightTeamContext


class InsightTeamSelector:
    """Load and select distinct NFL teams from the canonical generated player CSV."""

    def __init__(self, csv_path: str, draft_year: int) -> None:
        """
        Parameters
        ----------
        csv_path : str
            Path to ``generated_player_data.csv``.
        draft_year : int
            Draft year used in search query templates.
        """
        self._csv_path = csv_path
        self._draft_year = draft_year

    def select(
        self,
        start_index: int = 0,
        max_teams: int | None = None,
    ) -> list[InsightTeamContext]:
        """Return every distinct NFL team as an insight context.

        Parameters
        ----------
        start_index : int, optional
            Zero-based offset into the alphabetically sorted team list.
        max_teams : int, optional
            Optional cap on how many teams to return after ``start_index``.

        Returns
        -------
        list[InsightTeamContext]
            Selected team contexts, sorted alphabetically by abbreviation.

        Raises
        ------
        FileNotFoundError
            When the CSV path does not exist.
        ValueError
            When the CSV has no usable team column.
        """
        path = Path(self._csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Player data CSV not found at '{self._csv_path}'. "
                "Run 'docker compose run --rm data' first."
            )

        dataframe = pd.read_csv(path)
        team_column = "recent_team" if "recent_team" in dataframe.columns else "team"
        if team_column not in dataframe.columns:
            raise ValueError("Player data CSV is missing a team column ('recent_team' or 'team').")

        team_abbrs = sorted(
            {str(value).strip() for value in dataframe[team_column].dropna() if str(value).strip()}
        )

        if start_index > 0:
            team_abbrs = team_abbrs[start_index:]
        if max_teams is not None:
            team_abbrs = team_abbrs[:max_teams]

        return [
            InsightTeamContext(team_abbr=team_abbr, draft_year=self._draft_year)
            for team_abbr in team_abbrs
        ]
