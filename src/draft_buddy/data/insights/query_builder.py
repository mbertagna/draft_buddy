"""Search query templates for player insight enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from draft_buddy.data.insights.player_context import InsightPlayerContext


class QueryKind(str, Enum):
    """Named search query kinds for caching."""

    OUTLOOK = "outlook"
    ROLE = "role"
    INJURY_RECOVERY = "injury_recovery"


@dataclass(frozen=True, slots=True)
class InsightQuery:
    """A single search query with its cache key."""

    kind: QueryKind
    text: str


class InsightQueryBuilder:
    """Build Google CSE queries for a player insight context."""

    def build_queries(self, player: InsightPlayerContext) -> list[InsightQuery]:
        """Return all search queries to run for one player.

        Parameters
        ----------
        player : InsightPlayerContext
            Player context from the generated CSV.

        Returns
        -------
        list[InsightQuery]
            Two standard queries plus an optional injury-recovery query.
        """
        queries = [
            InsightQuery(kind=QueryKind.OUTLOOK, text=self._outlook_query(player)),
            InsightQuery(kind=QueryKind.ROLE, text=self._role_query(player)),
        ]
        if self.should_run_injury_recovery_query(player):
            queries.append(
                InsightQuery(
                    kind=QueryKind.INJURY_RECOVERY,
                    text=self._injury_recovery_query(player),
                )
            )
        return queries

    def should_run_injury_recovery_query(self, player: InsightPlayerContext) -> bool:
        """Return whether the optional injury-recovery query should run.

        Parameters
        ----------
        player : InsightPlayerContext
            Player context.

        Returns
        -------
        bool
            True when injury or low durability signals are present (rookies excluded).
        """
        if player.is_rookie():
            return False

        numeric_gp = player.numeric_games_played_frac()
        if numeric_gp is not None and numeric_gp < 0.70:
            return True

        if player.sleeper_injury_status:
            return True

        return False

    def _outlook_query(self, player: InsightPlayerContext) -> str:
        """Build the season-outlook query."""
        return (
            f'"{player.name}" {player.team} {player.position} '
            f"{player.draft_year} fantasy outlook injury role"
        )

    def _role_query(self, player: InsightPlayerContext) -> str:
        """Build the position-specific role query."""
        templates = {
            "RB": f'"{player.name}" {player.team} snap share touches committee starter',
            "WR": f'"{player.name}" {player.team} target share WR1 WR2 depth chart',
            "TE": f'"{player.name}" {player.team} routes red zone TE1',
            "QB": f'"{player.name}" {player.team} pass attempts rushing floor ceiling',
        }
        return templates.get(
            player.position,
            f'"{player.name}" {player.team} {player.position} role snaps targets',
        )

    def _injury_recovery_query(self, player: InsightPlayerContext) -> str:
        """Build the injury-recovery query."""
        return f'"{player.name}" {player.team} injury recovery cleared full practice {player.draft_year}'
