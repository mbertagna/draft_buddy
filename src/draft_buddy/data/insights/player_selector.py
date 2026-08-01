"""Select top-N players by ADP for insight enrichment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from draft_buddy.data.insights.player_context import InsightPlayerContext


class InsightPlayerSelector:
    """Load and select players from the canonical generated player CSV."""

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
        top_n: int,
        start_index: int = 0,
        max_players: int | None = None,
    ) -> list[InsightPlayerContext]:
        """Return the top-N players by ADP as insight contexts.

        Parameters
        ----------
        top_n : int
            Number of players to select by ADP rank.
        start_index : int, optional
            Zero-based offset into the ADP-sorted list.
        max_players : int, optional
            Optional cap on how many players to return after ``start_index``.

        Returns
        -------
        list[InsightPlayerContext]
            Selected player contexts.

        Raises
        ------
        FileNotFoundError
            When the CSV path does not exist.
        ValueError
            When the CSV has fewer rows than requested.
        """
        path = Path(self._csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Player data CSV not found at '{self._csv_path}'. "
                "Run 'docker compose run --rm data' first."
            )

        dataframe = pd.read_csv(path)
        if "adp" not in dataframe.columns:
            raise ValueError("Player data CSV is missing required column 'adp'.")

        filtered = dataframe[dataframe["adp"].apply(lambda value: np.isfinite(float(value)))]
        filtered = filtered.sort_values("adp", ascending=True).head(top_n)

        if len(filtered) < top_n:
            raise ValueError(
                f"Player data CSV has only {len(filtered)} ADP-ranked rows; "
                f"requested top_n={top_n}."
            )

        if start_index > 0:
            filtered = filtered.iloc[start_index:]

        if max_players is not None:
            filtered = filtered.head(max_players)

        team_column = "recent_team" if "recent_team" in filtered.columns else "team"
        contexts: list[InsightPlayerContext] = []
        for _, row in filtered.iterrows():
            games_played = row.get("games_played_frac", 1.0)
            if pd.isna(games_played):
                games_played = 1.0
            elif games_played != "R":
                games_played = float(games_played)

            injury_status = row.get("sleeper_injury_status")
            if pd.isna(injury_status):
                injury_status = None

            depth_chart = row.get("sleeper_depth_chart_position")
            if pd.isna(depth_chart):
                depth_chart = None

            team_value = row.get(team_column, "")
            if pd.isna(team_value):
                team_value = ""

            contexts.append(
                InsightPlayerContext(
                    sleeper_id=str(int(row["player_id"])),
                    name=str(row["name"]),
                    position=str(row["position"]),
                    team=str(team_value),
                    adp=float(row["adp"]),
                    projected_points=float(row["projected_points"]),
                    games_played_frac=games_played,
                    draft_year=self._draft_year,
                    sleeper_injury_status=injury_status,
                    sleeper_depth_chart_position=depth_chart,
                )
            )
        return contexts
