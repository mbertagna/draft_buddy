"""Synthesize structured team outlooks from cached search snippets."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterable, Optional

from draft_buddy.data.insights.cse_gateway import SearchSnippet
from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_names import team_name_variants
from draft_buddy.data.insights.team_query_builder import TeamInsightQueryBuilder
from draft_buddy.data.insights.team_schemas import (
    TeamOutlook,
    TeamOutlooksFile,
    apply_team_outlook_post_validation,
    default_unknown_team_outlook,
    sanitize_team_synthesis_payload,
)
from draft_buddy.data.insights.team_search_cache import LEAGUE_CACHE_KEY, TeamSearchCacheStore

logger = logging.getLogger(__name__)

MAX_TEAM_SNIPPETS = 8
MAX_LEAGUE_SNIPPETS = 4
MAX_COMBINED_SNIPPETS = 10

SYSTEM_PROMPT = """You are a fantasy football research analyst covering NFL teams.
Synthesize a team outlook using ONLY the provided search snippets.
Rules:
- Output JSON matching the TeamOutlook schema exactly.
- Do not include markdown or prose outside JSON.
- Use unknown enum values and fields_unknown when snippets are insufficient.
- Every bullet must cite a snippet URL provided in the input.
- Prefer draft-year offseason evidence; ignore stale prior-season recaps when newer analysis exists.
- outlook_summary must be at most 2 sentences.
- Fill offense_tier, defense_tier, and schedule_hardness only when grounded in snippets.
- Set evidence_as_of to the newest cited bullet published_date when available.
- key_storylines should call out coaching changes, scheme changes, or notable roster moves.
- Some snippets are marked "[multi-team roundup]": they cover all 32 NFL teams, not just
  this one. For those, extract and use ONLY the portion discussing this specific team, and
  ignore content about other teams even if it appears in the same snippet.
"""


class TeamSynthesisCacheStore:
    """Read and write per-team synthesized outlook caches."""

    def __init__(self, cache_root: str) -> None:
        """
        Parameters
        ----------
        cache_root : str
            Root directory for team synthesis caches.
        """
        self._cache_root = cache_root
        os.makedirs(self._cache_root, exist_ok=True)

    def cache_path(self, team_abbr: str) -> str:
        """Return the cache file path for one team."""
        return os.path.join(self._cache_root, f"{team_abbr}.json")

    def has_cache(self, team_abbr: str) -> bool:
        """Return whether a synthesis cache exists for the team."""
        return os.path.isfile(self.cache_path(team_abbr))

    def load(self, team_abbr: str) -> TeamOutlook:
        """Load a cached team outlook."""
        with open(self.cache_path(team_abbr), encoding="utf-8") as handle:
            payload = json.load(handle)
        return TeamOutlook.model_validate(payload)

    def save(self, team_abbr: str, outlook: TeamOutlook) -> None:
        """Persist a synthesized team outlook."""
        with open(self.cache_path(team_abbr), "w", encoding="utf-8") as handle:
            json.dump(outlook.model_dump(mode="json"), handle, indent=2)


class TeamOutlookSynthesizer:
    """Build prompts and synthesize team outlooks via a structured LLM gateway."""

    def __init__(
        self,
        gateway: InsightSynthesisGateway,
        search_cache: TeamSearchCacheStore,
        synthesis_cache: TeamSynthesisCacheStore,
        league_search_cache: Optional[TeamSearchCacheStore] = None,
    ) -> None:
        """
        Parameters
        ----------
        gateway : InsightSynthesisGateway
            LLM gateway for structured synthesis.
        search_cache : TeamSearchCacheStore
            Cached team-specific search results.
        synthesis_cache : TeamSynthesisCacheStore
            Per-team synthesis cache.
        league_search_cache : TeamSearchCacheStore, optional
            Cached league-wide (all-32-teams) search results, keyed under
            ``LEAGUE_CACHE_KEY``. When provided, snippets that mention this
            team by name are pulled in alongside the team-specific ones.
            When omitted, only team-specific snippets are used.
        """
        self._gateway = gateway
        self._search_cache = search_cache
        self._synthesis_cache = synthesis_cache
        self._league_search_cache = league_search_cache
        self._query_builder = TeamInsightQueryBuilder()

    def synthesize_team(
        self,
        team: InsightTeamContext,
        force: bool = False,
    ) -> TeamOutlook:
        """Synthesize one team outlook, using cache when available.

        Parameters
        ----------
        team : InsightTeamContext
            Team context.
        force : bool, optional
            Re-synthesize even when cache exists.

        Returns
        -------
        TeamOutlook
            Structured outlook for the team.

        Raises
        ------
        FileNotFoundError
            When search cache manifest is missing.
        """
        if not self._search_cache.has_manifest(team.team_abbr):
            raise FileNotFoundError(
                f"Search cache missing for team_abbr={team.team_abbr}. "
                "Run fetch_player_insight_search.py --scope teams first."
            )

        if self._synthesis_cache.has_cache(team.team_abbr) and not force:
            return self._synthesis_cache.load(team.team_abbr)

        team_snippets = self._search_cache.load_snippets(
            team.team_abbr,
            max_snippets=MAX_TEAM_SNIPPETS,
            draft_year=team.draft_year,
        )
        league_snippets = self._load_relevant_league_snippets(team)
        snippets = _merge_snippets(team_snippets, league_snippets, max_total=MAX_COMBINED_SNIPPETS)
        league_urls = {snippet.url for snippet in league_snippets}

        queries = self._query_builder.build_queries(team)
        query_texts = [query.text for query in queries]
        allowed_urls = {snippet.url for snippet in snippets if snippet.url}

        if not snippets:
            outlook = default_unknown_team_outlook(team.team_abbr, team.draft_year, query_texts)
            self._synthesis_cache.save(team.team_abbr, outlook)
            return outlook

        user_prompt = build_team_user_prompt(team, snippets, query_texts, league_urls=league_urls)
        try:
            payload = self._gateway.generate_structured(
                SYSTEM_PROMPT, user_prompt, TeamOutlook, "team_outlook"
            )
            raw_outlook = TeamOutlook.model_validate(sanitize_team_synthesis_payload(payload))
            outlook = apply_team_outlook_post_validation(raw_outlook, allowed_urls=allowed_urls)
            outlook = outlook.model_copy(
                update={
                    "search_queries_used": query_texts,
                    "snippet_count": len(snippets),
                }
            )
        except Exception:
            logger.exception(
                "Team outlook synthesis failed for team_abbr=%s",
                team.team_abbr,
            )
            outlook = default_unknown_team_outlook(team.team_abbr, team.draft_year, query_texts)
            outlook = outlook.model_copy(update={"snippet_count": len(snippets)})

        self._synthesis_cache.save(team.team_abbr, outlook)
        return outlook

    def _load_relevant_league_snippets(self, team: InsightTeamContext) -> list[SearchSnippet]:
        """Return cached league-wide snippets that mention this team by name.

        Parameters
        ----------
        team : InsightTeamContext
            Team context.

        Returns
        -------
        list[SearchSnippet]
            Up to ``MAX_LEAGUE_SNIPPETS`` league-wide snippets mentioning
            the team, or an empty list when no league cache is configured
            or none mention this team.
        """
        if self._league_search_cache is None:
            return []
        if not self._league_search_cache.has_manifest(LEAGUE_CACHE_KEY):
            return []

        league_snippets = self._league_search_cache.load_snippets(
            LEAGUE_CACHE_KEY,
            max_snippets=20,
            draft_year=team.draft_year,
        )
        name_variants = team_name_variants(team.team_abbr)
        matching = [
            snippet for snippet in league_snippets if _snippet_mentions_team(snippet, name_variants)
        ]
        return matching[:MAX_LEAGUE_SNIPPETS]


def _snippet_mentions_team(snippet: SearchSnippet, name_variants: list[str]) -> bool:
    """Return whether a snippet's title or text mentions any team name variant."""
    haystack = f"{snippet.title} {snippet.snippet}".lower()
    return any(variant.lower() in haystack for variant in name_variants)


def _merge_snippets(
    team_snippets: list[SearchSnippet],
    league_snippets: list[SearchSnippet],
    max_total: int,
) -> list[SearchSnippet]:
    """Combine team-specific and league-wide snippets, deduped by URL.

    Team-specific snippets take priority; league-wide snippets fill any
    remaining budget up to ``max_total``.
    """
    merged: list[SearchSnippet] = []
    seen_urls: set[str] = set()
    for snippet in [*team_snippets, *league_snippets]:
        if snippet.url in seen_urls:
            continue
        seen_urls.add(snippet.url)
        merged.append(snippet)
    return merged[:max_total]


def build_team_user_prompt(
    team: InsightTeamContext,
    snippets: Iterable[SearchSnippet],
    query_texts: list[str],
    league_urls: Optional[set[str]] = None,
) -> str:
    """Build the LLM user prompt for one team.

    Parameters
    ----------
    team : InsightTeamContext
        Team context.
    snippets : Iterable[SearchSnippet]
        Search snippets from cache.
    query_texts : list[str]
        Queries that were executed.
    league_urls : set[str], optional
        URLs of snippets sourced from the league-wide (all-32-teams) cache,
        flagged in the prompt so the LLM extracts only this team's portion.

    Returns
    -------
    str
        Markdown user prompt.
    """
    league_urls = league_urls or set()
    snippet_lines = []
    for index, snippet in enumerate(snippets, start=1):
        tag = " [multi-team roundup]" if snippet.url in league_urls else ""
        snippet_lines.append(
            f"{index}. title={snippet.title!r}{tag}\n"
            f"   url={snippet.url}\n"
            f"   domain={snippet.domain}\n"
            f"   date={snippet.published_date}\n"
            f"   snippet={snippet.snippet!r}"
        )

    return (
        "## Team (deterministic, do not contradict without snippet evidence)\n"
        f"- team_abbr: {team.team_abbr}\n"
        f"- season: {team.draft_year}\n\n"
        "## Search queries executed\n"
        + "\n".join(f"- {query}" for query in query_texts)
        + "\n\n## Search snippets (only allowed evidence sources)\n"
        + ("\n".join(snippet_lines) if snippet_lines else "No snippets available.")
    )


def merge_team_outlooks_file(
    draft_year: int,
    model: str,
    teams: dict[str, TeamOutlook],
) -> TeamOutlooksFile:
    """Build the merged team outlooks file payload.

    Parameters
    ----------
    draft_year : int
        Draft year.
    model : str
        LLM model name used for synthesis.
    teams : dict[str, TeamOutlook]
        Outlooks keyed by team abbreviation.

    Returns
    -------
    TeamOutlooksFile
        Merged file model.
    """
    return TeamOutlooksFile(
        draft_year=draft_year,
        generated_at=datetime.now(timezone.utc),
        model=model,
        teams=teams,
    )
