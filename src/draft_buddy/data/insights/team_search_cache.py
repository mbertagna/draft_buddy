"""Search result caching for team outlook enrichment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx

from draft_buddy.data.insights.cse_gateway import (
    QuotaExceededError,
    SearchGateway,
    SearchSnippet,
    snippet_mentions_year,
    snippet_rank_score,
)
from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_query_builder import TeamInsightQuery, TeamQueryKind
from draft_buddy.data.insights.team_search_windows import (
    team_start_date_for_query_kind,
    valyu_instructions_for_team_query_kind,
)

LEAGUE_CACHE_KEY = "LEAGUE"
"""Pseudo team abbreviation used to key league-wide (all-32-teams) search cache entries."""


@dataclass(frozen=True, slots=True)
class TeamSearchCacheManifest:
    """Manifest describing cached search results for one team."""

    team_abbr: str
    draft_year: int
    queries: list[str]
    fetched_at: str
    provider: str = "google"

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a JSON-friendly dictionary."""
        return {
            "team_abbr": self.team_abbr,
            "draft_year": self.draft_year,
            "queries": self.queries,
            "fetched_at": self.fetched_at,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamSearchCacheManifest":
        """Build a manifest from serialized data."""
        return cls(
            team_abbr=str(payload["team_abbr"]),
            draft_year=int(payload["draft_year"]),
            queries=[str(value) for value in payload.get("queries", [])],
            fetched_at=str(payload["fetched_at"]),
            provider=str(payload.get("provider", "google")),
        )


class TeamSearchCacheStore:
    """Read and write per-team search caches, kept separate from player caches."""

    def __init__(self, cache_root: str) -> None:
        """
        Parameters
        ----------
        cache_root : str
            Root directory for team search caches.
        """
        self._cache_root = cache_root
        os.makedirs(self._cache_root, exist_ok=True)

    def team_cache_dir(self, team_abbr: str) -> str:
        """Return the cache directory for one team."""
        return os.path.join(self._cache_root, team_abbr)

    def manifest_path(self, team_abbr: str) -> str:
        """Return the manifest file path for one team."""
        return os.path.join(self.team_cache_dir(team_abbr), "manifest.json")

    def has_manifest(self, team_abbr: str) -> bool:
        """Return whether a manifest exists for the team."""
        return os.path.isfile(self.manifest_path(team_abbr))

    def load_manifest(self, team_abbr: str) -> TeamSearchCacheManifest:
        """Load the manifest for one team."""
        with open(self.manifest_path(team_abbr), encoding="utf-8") as handle:
            return TeamSearchCacheManifest.from_dict(json.load(handle))

    def has_cached_provider(self, team_abbr: str, provider: str) -> bool:
        """Return whether a manifest exists for the team and provider.

        Parameters
        ----------
        team_abbr : str
            NFL team abbreviation.
        provider : str
            Search provider name.

        Returns
        -------
        bool
            True when cache exists for the same provider.
        """
        if not self.has_manifest(team_abbr):
            return False
        return self.load_manifest(team_abbr).provider == provider

    def save_query_result(
        self,
        team: InsightTeamContext,
        query: TeamInsightQuery,
        raw_response: dict[str, Any],
        snippets: list[SearchSnippet],
        provider: str,
    ) -> TeamSearchCacheManifest:
        """Persist one query result and update the team manifest.

        Parameters
        ----------
        team : InsightTeamContext
            Team context.
        query : TeamInsightQuery
            Query that was executed.
        raw_response : dict
            Raw search API response.
        snippets : list[SearchSnippet]
            Parsed snippets for convenience.
        provider : str
            Search provider name (``valyu`` or ``google``).

        Returns
        -------
        TeamSearchCacheManifest
            Updated manifest for the team.
        """
        team_dir = self.team_cache_dir(team.team_abbr)
        os.makedirs(team_dir, exist_ok=True)

        query_filename = f"{query.kind.value}.json"
        query_path = os.path.join(team_dir, query_filename)
        payload = {
            "query": query.text,
            "kind": query.kind.value,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": raw_response,
            "snippets": [
                {
                    "title": snippet.title,
                    "snippet": snippet.snippet,
                    "url": snippet.url,
                    "domain": snippet.domain,
                    "published_date": snippet.published_date,
                    "relevance_score": snippet.relevance_score,
                }
                for snippet in snippets
            ],
        }
        with open(query_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        queries_run = self._existing_query_kinds(team.team_abbr)
        if query.kind.value not in queries_run:
            queries_run.append(query.kind.value)

        manifest = TeamSearchCacheManifest(
            team_abbr=team.team_abbr,
            draft_year=team.draft_year,
            queries=queries_run,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
        )
        with open(self.manifest_path(team.team_abbr), "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
        return manifest

    def load_snippets(
        self,
        team_abbr: str,
        max_snippets: int = 8,
        draft_year: Optional[int] = None,
        today: Optional[date] = None,
    ) -> list[SearchSnippet]:
        """Load ranked, date-filtered snippets for one team from the search cache.

        Parameters
        ----------
        team_abbr : str
            NFL team abbreviation.
        max_snippets : int, optional
            Maximum snippets to return.
        draft_year : int, optional
            Draft year for the outlook window. When omitted, uses the
            team manifest when available.
        today : date, optional
            Reference date for the recency ranking boost; defaults to today.

        Returns
        -------
        list[SearchSnippet]
            Deduped, filtered, and ranked snippets across cached queries.
        """
        team_dir = self.team_cache_dir(team_abbr)
        if not os.path.isdir(team_dir):
            return []

        resolved_year = draft_year
        if resolved_year is None and self.has_manifest(team_abbr):
            resolved_year = self.load_manifest(team_abbr).draft_year

        window_start = (
            team_start_date_for_query_kind(TeamQueryKind.OUTLOOK, resolved_year)
            if resolved_year
            else None
        )

        seen_urls: set[str] = set()
        snippets: list[SearchSnippet] = []
        for filename in sorted(os.listdir(team_dir)):
            if not filename.endswith(".json") or filename == "manifest.json":
                continue
            with open(os.path.join(team_dir, filename), encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload.get("snippets", []):
                url = str(item.get("url", ""))
                if not url or url in seen_urls:
                    continue
                published_date = item.get("published_date")
                title = str(item.get("title", ""))
                snippet_text = str(item.get("snippet", ""))
                if not _snippet_passes_window(
                    published_date,
                    window_start,
                    draft_year=resolved_year,
                    fallback_text=f"{title} {snippet_text}",
                ):
                    continue
                seen_urls.add(url)
                relevance_raw = item.get("relevance_score")
                try:
                    relevance_score = float(relevance_raw) if relevance_raw is not None else None
                except (TypeError, ValueError):
                    relevance_score = None
                snippets.append(
                    SearchSnippet(
                        title=title,
                        snippet=snippet_text,
                        url=url,
                        domain=str(item.get("domain", "")),
                        published_date=published_date,
                        relevance_score=relevance_score,
                    )
                )

        snippets.sort(
            key=lambda snippet: snippet_rank_score(snippet, today=today or date.today()),
            reverse=True,
        )
        return snippets[:max_snippets]

    def _existing_query_kinds(self, team_abbr: str) -> list[str]:
        """Return query kinds already present in the cache."""
        if not self.has_manifest(team_abbr):
            return []
        return list(self.load_manifest(team_abbr).queries)


def _snippet_passes_window(
    published_date: Optional[str],
    window_start: Optional[str],
    draft_year: Optional[int],
    fallback_text: str,
) -> bool:
    """Return whether a snippet survives the team outlook date window.

    Snippets without a published date are only kept when the draft year
    appears in their title or snippet text, since search providers
    sometimes omit structured dates even for current-season content.
    """
    if not published_date:
        return snippet_mentions_year(fallback_text, draft_year)
    if window_start is None:
        return True
    return str(published_date)[:10] >= window_start


def execute_team_search_with_cache(
    gateway: SearchGateway,
    cache_store: TeamSearchCacheStore,
    team: InsightTeamContext,
    query: TeamInsightQuery,
    provider: str,
) -> list[SearchSnippet]:
    """Execute a search and persist the raw response to the team cache.

    Parameters
    ----------
    gateway : SearchGateway
        Search gateway.
    cache_store : TeamSearchCacheStore
        Team cache store.
    team : InsightTeamContext
        Team context.
    query : TeamInsightQuery
        Query to execute.
    provider : str
        Search provider name stored in the manifest.

    Returns
    -------
    list[SearchSnippet]
        Parsed snippets.

    Raises
    ------
    QuotaExceededError
        When the API returns a quota error.
    """
    start_date = team_start_date_for_query_kind(query.kind, team.draft_year)
    instructions = valyu_instructions_for_team_query_kind(query.kind)
    try:
        snippets, raw_payload = gateway.search_raw(
            query.text,
            start_date=start_date,
            instructions=instructions,
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 429:
            raise QuotaExceededError("Search API quota exceeded.") from error
        raise

    cache_store.save_query_result(team, query, raw_payload, snippets, provider=provider)
    return snippets
