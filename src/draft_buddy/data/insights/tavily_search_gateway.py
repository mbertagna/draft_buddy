"""Tavily Search API gateway for player insight enrichment."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from draft_buddy.data.insights.search_gateway import (
    DEFAULT_INCLUDED_SOURCES,
    DEFAULT_RESULT_COUNT,
    QuotaExceededError,
    SearchGateway,
    SearchSnippet,
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SNIPPET_CHARS = 1200
DEFAULT_TAVILY_TOPIC = "general"
DEFAULT_RELEVANCE_THRESHOLD = 0.7


def build_tavily_search_gateway() -> TavilySearchGateway:
    """Build a Tavily search gateway from environment configuration.

    Returns
    -------
    TavilySearchGateway
        Configured Tavily search gateway.

    Raises
    ------
    ValueError
        When ``TAVILY_API_KEY`` is not set.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required.")
    return TavilySearchGateway(
        api_key=api_key,
        relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
    )


class TavilySearchGateway(SearchGateway):
    """Fetches search results from the Tavily Search API."""

    def __init__(
        self,
        api_key: str,
        included_sources: tuple[str, ...] | None = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Parameters
        ----------
        api_key : str
            Tavily API key (``Authorization: Bearer`` header).
        included_sources : tuple[str, ...], optional
            Domains to restrict search results to.
        relevance_threshold : float, optional
            Minimum relevance score kept after parsing.
        timeout_seconds : float, optional
            HTTP request timeout.
        """
        self._api_key = api_key
        self._included_sources = included_sources or DEFAULT_INCLUDED_SOURCES
        self._relevance_threshold = relevance_threshold
        self._timeout_seconds = timeout_seconds

    def search_raw(
        self,
        query: str,
        num_results: int = DEFAULT_RESULT_COUNT,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        instructions: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> tuple[list[SearchSnippet], dict[str, Any]]:
        """Execute a Tavily search and return snippets plus the raw payload.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results (Tavily max 20 per request).
        start_date : str, optional
            Inclusive publication start date (``YYYY-MM-DD``).
        end_date : str, optional
            Inclusive publication end date (``YYYY-MM-DD``).
        instructions : str, optional
            Unused; accepted for ``SearchGateway`` compatibility.
        topic : str, optional
            Tavily topic (``general`` or ``news``). Defaults to ``general``.

        Returns
        -------
        tuple[list[SearchSnippet], dict]
            Parsed snippets and raw API JSON.
        """
        del instructions
        body: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "topic": topic or DEFAULT_TAVILY_TOPIC,
            "max_results": min(num_results, 20),
            "include_answer": False,
            "include_raw_content": False,
            "include_domains": list(self._included_sources),
        }
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date

        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
            )
            if response.status_code == 429:
                raise QuotaExceededError("Tavily API quota exceeded.")
            response.raise_for_status()
            payload = response.json()
        snippets = [
            snippet
            for snippet in self._parse_response(payload)
            if snippet.relevance_score is None
            or snippet.relevance_score >= self._relevance_threshold
        ]
        return snippets, payload

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> list[SearchSnippet]:
        """Parse a Tavily search JSON response into normalized snippets.

        Parameters
        ----------
        payload : dict
            Raw API response.

        Returns
        -------
        list[SearchSnippet]
            Parsed snippets.
        """
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        snippets: list[SearchSnippet] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            snippet_text = TavilySearchGateway._select_snippet_text(item)
            domain = urlparse(url).netloc or str(item.get("source") or "")
            published_date = TavilySearchGateway._normalize_date(item.get("published_date"))
            relevance_score = TavilySearchGateway._normalize_relevance(item.get("score"))
            snippets.append(
                SearchSnippet(
                    title=str(item.get("title") or ""),
                    snippet=snippet_text,
                    url=url,
                    domain=domain,
                    published_date=published_date,
                    relevance_score=relevance_score,
                )
            )
        return snippets

    @staticmethod
    def _select_snippet_text(item: dict[str, Any]) -> str:
        """Select and truncate Tavily result content.

        Parameters
        ----------
        item : dict
            One Tavily result object.

        Returns
        -------
        str
            Truncated snippet text for synthesis.
        """
        selected = str(item.get("content") or item.get("raw_content") or "").strip()
        if len(selected) > MAX_SNIPPET_CHARS:
            return selected[:MAX_SNIPPET_CHARS]
        return selected

    @staticmethod
    def _normalize_date(value: object) -> Optional[str]:
        """Normalize a publication date to an ISO date string when possible."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text[:10]

    @staticmethod
    def _normalize_relevance(value: object) -> Optional[float]:
        """Normalize a Tavily relevance score to a float when possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
