"""Search query templates for team outlook enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from draft_buddy.data.insights.team_context import InsightTeamContext


class TeamQueryKind(str, Enum):
    """Named search query kinds for team-level caching."""

    OUTLOOK = "outlook"
    OFFENSE = "offense"
    DEFENSE = "defense"
    SCHEDULE = "schedule"


@dataclass(frozen=True, slots=True)
class TeamInsightQuery:
    """A single team search query with its cache key."""

    kind: TeamQueryKind
    text: str


class TeamInsightQueryBuilder:
    """Build search queries for a team insight context."""

    def build_queries(self, team: InsightTeamContext) -> list[TeamInsightQuery]:
        """Return all search queries to run for one team.

        Parameters
        ----------
        team : InsightTeamContext
            Team context.

        Returns
        -------
        list[TeamInsightQuery]
            One query per team query kind (outlook, offense, defense, schedule).
        """
        return [
            TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text=self._outlook_query(team)),
            TeamInsightQuery(kind=TeamQueryKind.OFFENSE, text=self._offense_query(team)),
            TeamInsightQuery(kind=TeamQueryKind.DEFENSE, text=self._defense_query(team)),
            TeamInsightQuery(kind=TeamQueryKind.SCHEDULE, text=self._schedule_query(team)),
        ]

    def _outlook_query(self, team: InsightTeamContext) -> str:
        """Build the general season-outlook query."""
        return (
            f"{team.team_abbr} NFL {team.draft_year} season outlook preview coaching roster changes"
        )

    def _offense_query(self, team: InsightTeamContext) -> str:
        """Build the offensive-ability query."""
        return (
            f"{team.team_abbr} NFL {team.draft_year} offense outlook offensive line "
            "skill position talent"
        )

    def _defense_query(self, team: InsightTeamContext) -> str:
        """Build the defensive-ability query."""
        return (
            f"{team.team_abbr} NFL {team.draft_year} defense outlook pass rush "
            "secondary run defense"
        )

    def _schedule_query(self, team: InsightTeamContext) -> str:
        """Build the schedule-hardness query."""
        return (
            f"{team.team_abbr} NFL {team.draft_year} schedule strength of schedule "
            "difficulty analysis"
        )


class LeagueInsightQueryBuilder:
    """Build broad, league-wide search queries covering all 32 teams at once.

    These intentionally omit a team name so a single fetch can surface the
    same multi-team roundup articles (best/worst-case previews, division
    rankings, strength-of-schedule breakdowns) that would otherwise be
    re-fetched redundantly once per team.
    """

    def build_queries(self, draft_year: int) -> list[TeamInsightQuery]:
        """Return the four league-wide search queries for one draft year.

        Parameters
        ----------
        draft_year : int
            Draft season year.

        Returns
        -------
        list[TeamInsightQuery]
            One query per query kind (outlook, offense, defense, schedule).
        """
        return [
            TeamInsightQuery(
                kind=TeamQueryKind.OUTLOOK,
                text=f"NFL {draft_year} season outlook best worst case scenarios all 32 teams",
            ),
            TeamInsightQuery(
                kind=TeamQueryKind.OFFENSE,
                text=(
                    f"NFL {draft_year} offensive line rankings skill position talent all 32 teams"
                ),
            ),
            TeamInsightQuery(
                kind=TeamQueryKind.DEFENSE,
                text=(f"NFL {draft_year} defensive rankings pass rush secondary all 32 teams"),
            ),
            TeamInsightQuery(
                kind=TeamQueryKind.SCHEDULE,
                text=f"NFL {draft_year} strength of schedule rankings all 32 teams",
            ),
        ]
