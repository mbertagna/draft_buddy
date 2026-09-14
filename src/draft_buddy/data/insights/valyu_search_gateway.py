"""Valyu Search API gateway for player insight enrichment."""

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

VALYU_SEARCH_URL = "https://api.valyu.ai/v1/search"
MAX_SNIPPET_CHARS = 1200
DEFAULT_RELEVANCE_THRESHOLD = 0.7


def build_valyu_search_gateway() -> ValyuSearchGateway:
    """Build a Valyu search gateway from environment configuration.

    Returns
    -------
    ValyuSearchGateway
        Configured Valyu search gateway.

    Raises
    ------
    ValueError
        When ``VALYU_API_KEY`` is not set.
    """
    api_key = os.environ.get("VALYU_API_KEY")
    if not api_key:
        raise ValueError("VALYU_API_KEY environment variable is required.")
    return ValyuSearchGateway(api_key=api_key, relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD)


class ValyuSearchGateway(SearchGateway):
    """Fetches search results from the Valyu Search API."""

    def __init__(
        self,
        api_key: str,
        included_sources: tuple[str, ...] | None = None,
        search_type: str = "web",
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        timeout_seconds: float = 60.0,
    ) -> None:
        """
        Parameters
        ----------
        api_key : str
            Valyu API key (``X-Api-Key`` header).
        included_sources : tuple[str, ...], optional
            Domains to restrict search results to.
        search_type : str, optional
            Valyu search type (``web``, ``news``, or ``all``).
        relevance_threshold : float, optional
            Minimum relevance score for returned results.
        timeout_seconds : float, optional
            HTTP request timeout.
        """
        self._api_key = api_key
        self._included_sources = included_sources or DEFAULT_INCLUDED_SOURCES
        self._search_type = search_type
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
        """Execute a Valyu search and return snippets plus the raw payload.

        Parameters
        ----------
        query : str
            Search query text.
        num_results : int, optional
            Maximum number of results (Valyu max 20 per request).
        start_date : str, optional
            Inclusive publication start date (``YYYY-MM-DD``).
        end_date : str, optional
            Inclusive publication end date (``YYYY-MM-DD``).
        instructions : str, optional
            Natural-language ranking instructions for Valyu.
        topic : str, optional
            Unused; accepted for ``SearchGateway`` compatibility.

        Returns
        -------
        tuple[list[SearchSnippet], dict]
            Parsed snippets and raw API JSON.
        """
        body: dict[str, Any] = {
            "query": query,
            "search_type": self._search_type,
            "max_num_results": min(num_results, 20),
            "response_length": "short",
            "relevance_threshold": self._relevance_threshold,
            "included_sources": list(self._included_sources),
        }
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        if instructions:
            body["instructions"] = instructions

        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(
                VALYU_SEARCH_URL,
                headers={"X-Api-Key": self._api_key},
                json=body,
            )
            if response.status_code == 429:
                raise QuotaExceededError("Valyu API quota exceeded.")
            response.raise_for_status()
            payload = response.json()
        return self._parse_response(payload), payload

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> list[SearchSnippet]:
        """Parse a Valyu search JSON response into normalized snippets.

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
        if results is None and isinstance(payload.get("data"), dict):
            results = payload["data"].get("results")
        if not isinstance(results, list):
            return []

        snippets: list[SearchSnippet] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            snippet_text = ValyuSearchGateway._select_snippet_text(item)
            domain = urlparse(url).netloc or str(item.get("source") or "")
            published_date = ValyuSearchGateway._normalize_date(item.get("publication_date"))
            relevance_score = ValyuSearchGateway._normalize_relevance(item.get("relevance_score"))
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
        """Prefer article body over meta description, then truncate.

        Parameters
        ----------
        item : dict
            One Valyu result object.

        Returns
        -------
        str
            Truncated snippet text for synthesis.
        """
        content = str(item.get("content") or item.get("text") or item.get("snippet") or "").strip()
        description = str(item.get("description") or "").strip()
        selected = content if content else description
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
        """Normalize a Valyu relevance score to a float when possible."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
