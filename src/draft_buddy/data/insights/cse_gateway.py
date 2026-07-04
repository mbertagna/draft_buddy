"""Google Custom Search JSON API gateway for player insight enrichment."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQuery, QueryKind


CSE_BASE_URL = "https://customsearch.googleapis.com/customsearch/v1"
DEFAULT_RESULT_COUNT = 8


@dataclass(frozen=True, slots=True)
class SearchSnippet:
    """Normalized search result snippet from a search provider."""

    title: str
    snippet: str
    url: str
    domain: str
    published_date: Optional[str] = None


class SearchGateway(ABC):
    """Abstract interface for web search used in insight enrichment."""

    @abstractmethod
    def search_raw(
        self, query: str, num_results: int = DEFAULT_RESULT_COUNT
    ) -> tuple[list[SearchSnippet], dict[str, Any]]:
        """Execute a search query and return snippets plus raw API JSON.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results to return.

        Returns
        -------
        tuple[list[SearchSnippet], dict]
            Parsed snippets and raw provider response.
        """

    def search(self, query: str, num_results: int = DEFAULT_RESULT_COUNT) -> list[SearchSnippet]:
        """Execute a search query and return normalized snippets.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results to return.

        Returns
        -------
        list[SearchSnippet]
            Normalized search snippets.
        """
        snippets, _ = self.search_raw(query, num_results=num_results)
        return snippets


class GoogleCseGateway(SearchGateway):
    """Fetches search results from Google Custom Search JSON API."""

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        api_key : str
            Google API key with Custom Search enabled.
        search_engine_id : str
            Programmable Search Engine ID (``cx``).
        timeout_seconds : float, optional
            HTTP request timeout.
        """
        self._api_key = api_key
        self._search_engine_id = search_engine_id
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, num_results: int = DEFAULT_RESULT_COUNT) -> list[SearchSnippet]:
        """Execute a CSE query and normalize the response.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results (CSE max is 10).

        Returns
        -------
        list[SearchSnippet]
            Parsed snippets from the API response.
        """
        return super().search(query, num_results=num_results)

    def search_raw(
        self, query: str, num_results: int = DEFAULT_RESULT_COUNT
    ) -> tuple[list[SearchSnippet], dict[str, Any]]:
        """Execute a CSE query and return snippets plus the raw payload.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results (CSE max is 10).

        Returns
        -------
        tuple[list[SearchSnippet], dict]
            Parsed snippets and raw API JSON.
        """
        params = {
            "key": self._api_key,
            "cx": self._search_engine_id,
            "q": query,
            "num": min(num_results, 10),
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.get(CSE_BASE_URL, params=params)
            if response.status_code == 429:
                raise QuotaExceededError("Google CSE daily quota exceeded.")
            response.raise_for_status()
            payload = response.json()
        return self._parse_response(payload), payload

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> list[SearchSnippet]:
        """Parse a CSE JSON response into normalized snippets.

        Parameters
        ----------
        payload : dict
            Raw API response.

        Returns
        -------
        list[SearchSnippet]
            Parsed snippets.
        """
        items = payload.get("items", [])
        snippets: list[SearchSnippet] = []
        for item in items:
            published_date = GoogleCseGateway._extract_published_date(item)
            snippets.append(
                SearchSnippet(
                    title=str(item.get("title", "")),
                    snippet=str(item.get("snippet", "")),
                    url=str(item.get("link", "")),
                    domain=str(item.get("displayLink", "")),
                    published_date=published_date,
                )
            )
        return snippets

    @staticmethod
    def _extract_published_date(item: dict[str, Any]) -> Optional[str]:
        """Extract a published date from pagemap metatags when available."""
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [])
        for tag in metatags:
            for key in ("article:published_time", "og:updated_time", "date", "pubdate"):
                if key in tag and tag[key]:
                    return str(tag[key])[:10]
        return None


@dataclass(frozen=True, slots=True)
class SearchCacheManifest:
    """Manifest describing cached search results for one player."""

    sleeper_id: str
    name: str
    draft_year: int
    queries: list[str]
    fetched_at: str
    provider: str = "google"

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a JSON-friendly dictionary."""
        return {
            "sleeper_id": self.sleeper_id,
            "name": self.name,
            "draft_year": self.draft_year,
            "queries": self.queries,
            "fetched_at": self.fetched_at,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchCacheManifest":
        """Build a manifest from serialized data."""
        return cls(
            sleeper_id=str(payload["sleeper_id"]),
            name=str(payload["name"]),
            draft_year=int(payload["draft_year"]),
            queries=[str(value) for value in payload.get("queries", [])],
            fetched_at=str(payload["fetched_at"]),
            provider=str(payload.get("provider", "google")),
        )


class SearchCacheStore:
    """Read and write per-player search caches."""

    def __init__(self, cache_root: str) -> None:
        """
        Parameters
        ----------
        cache_root : str
            Root directory for search caches.
        """
        self._cache_root = cache_root
        os.makedirs(self._cache_root, exist_ok=True)

    def player_cache_dir(self, sleeper_id: str) -> str:
        """Return the cache directory for one player."""
        return os.path.join(self._cache_root, sleeper_id)

    def manifest_path(self, sleeper_id: str) -> str:
        """Return the manifest file path for one player."""
        return os.path.join(self.player_cache_dir(sleeper_id), "manifest.json")

    def has_manifest(self, sleeper_id: str) -> bool:
        """Return whether a manifest exists for the player."""
        return os.path.isfile(self.manifest_path(sleeper_id))

    def load_manifest(self, sleeper_id: str) -> SearchCacheManifest:
        """Load the manifest for one player."""
        with open(self.manifest_path(sleeper_id), encoding="utf-8") as handle:
            return SearchCacheManifest.from_dict(json.load(handle))

    def has_cached_provider(self, sleeper_id: str, provider: str) -> bool:
        """Return whether a manifest exists for the player and provider.

        Parameters
        ----------
        sleeper_id : str
            Sleeper player id.
        provider : str
            Search provider name.

        Returns
        -------
        bool
            True when cache exists for the same provider.
        """
        if not self.has_manifest(sleeper_id):
            return False
        return self.load_manifest(sleeper_id).provider == provider

    def save_query_result(
        self,
        player: InsightPlayerContext,
        query: InsightQuery,
        raw_response: dict[str, Any],
        snippets: list[SearchSnippet],
        provider: str,
    ) -> SearchCacheManifest:
        """Persist one query result and update the player manifest.

        Parameters
        ----------
        player : InsightPlayerContext
            Player context.
        query : InsightQuery
            Query that was executed.
        raw_response : dict
            Raw search API response.
        snippets : list[SearchSnippet]
            Parsed snippets for convenience.
        provider : str
            Search provider name (``valyu`` or ``google``).

        Returns
        -------
        SearchCacheManifest
            Updated manifest for the player.
        """
        player_dir = self.player_cache_dir(player.sleeper_id)
        os.makedirs(player_dir, exist_ok=True)

        query_filename = f"{query.kind.value}.json"
        query_path = os.path.join(player_dir, query_filename)
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
                }
                for snippet in snippets
            ],
        }
        with open(query_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        queries_run = self._existing_query_kinds(player.sleeper_id)
        if query.kind.value not in queries_run:
            queries_run.append(query.kind.value)

        manifest = SearchCacheManifest(
            sleeper_id=player.sleeper_id,
            name=player.name,
            draft_year=player.draft_year,
            queries=queries_run,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
        )
        with open(self.manifest_path(player.sleeper_id), "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
        return manifest

    def load_snippets(self, sleeper_id: str, max_snippets: int = 8) -> list[SearchSnippet]:
        """Load deduplicated snippets for one player from the search cache.

        Parameters
        ----------
        sleeper_id : str
            Sleeper player id.
        max_snippets : int, optional
            Maximum snippets to return.

        Returns
        -------
        list[SearchSnippet]
            Deduped snippets across all cached queries.
        """
        player_dir = self.player_cache_dir(sleeper_id)
        if not os.path.isdir(player_dir):
            return []

        seen_urls: set[str] = set()
        snippets: list[SearchSnippet] = []
        for filename in sorted(os.listdir(player_dir)):
            if not filename.endswith(".json") or filename == "manifest.json":
                continue
            with open(os.path.join(player_dir, filename), encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload.get("snippets", []):
                url = str(item.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                snippets.append(
                    SearchSnippet(
                        title=str(item.get("title", "")),
                        snippet=str(item.get("snippet", "")),
                        url=url,
                        domain=str(item.get("domain", "")),
                        published_date=item.get("published_date"),
                    )
                )
                if len(snippets) >= max_snippets:
                    return snippets
        return snippets

    def _existing_query_kinds(self, sleeper_id: str) -> list[str]:
        """Return query kinds already present in the cache."""
        if not self.has_manifest(sleeper_id):
            return []
        return list(self.load_manifest(sleeper_id).queries)


class QuotaExceededError(RuntimeError):
    """Raised when a search provider API quota is exceeded."""


def execute_search_with_cache(
    gateway: SearchGateway,
    cache_store: SearchCacheStore,
    player: InsightPlayerContext,
    query: InsightQuery,
    provider: str,
) -> list[SearchSnippet]:
    """Execute a search and persist the raw response to cache.

    Parameters
    ----------
    gateway : SearchGateway
        Search gateway.
    cache_store : SearchCacheStore
        Cache store.
    player : InsightPlayerContext
        Player context.
    query : InsightQuery
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
    try:
        snippets, raw_payload = gateway.search_raw(query.text)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 429:
            raise QuotaExceededError("Search API quota exceeded.") from error
        raise

    cache_store.save_query_result(player, query, raw_payload, snippets, provider=provider)
    return snippets


def sleep_between_batches(delay_ms: int) -> None:
    """Sleep between search batches when rate limiting.

    Parameters
    ----------
    delay_ms : int
        Delay in milliseconds.
    """
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
