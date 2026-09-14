"""Tests for Tavily Search API gateway."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from draft_buddy.data.insights.tavily_search_gateway import (
    MAX_SNIPPET_CHARS,
    TavilySearchGateway,
    build_tavily_search_gateway,
)


def test_parse_response_extracts_snippets_from_results() -> None:
    """Verify Tavily parsing maps content, score, and published date."""
    payload = {
        "results": [
            {
                "title": "McCaffrey outlook",
                "url": "https://www.espn.com/fantasy/mccaffrey",
                "content": "Expected workhorse role in 2026.",
                "published_date": "2026-06-15T12:00:00Z",
                "score": 0.91,
            }
        ]
    }

    snippets = TavilySearchGateway._parse_response(payload)

    assert snippets[0].domain == "www.espn.com"
    assert snippets[0].published_date == "2026-06-15"
    assert snippets[0].relevance_score == pytest.approx(0.91)
    assert "workhorse" in snippets[0].snippet


def test_parse_response_skips_results_without_url() -> None:
    """Verify results missing a URL are dropped."""
    payload = {
        "results": [
            {"title": "No url", "content": "text", "score": 0.9},
            {
                "title": "Valid",
                "url": "https://fantasypros.com/article",
                "content": "ok",
                "score": 0.8,
            },
        ]
    }

    snippets = TavilySearchGateway._parse_response(payload)

    assert [snippet.url for snippet in snippets] == ["https://fantasypros.com/article"]


def test_parse_response_returns_empty_for_malformed_payload() -> None:
    """Verify malformed Tavily payloads yield no snippets."""
    snippets = TavilySearchGateway._parse_response({"results": "not-a-list"})

    assert snippets == []


def test_parse_response_truncates_long_content() -> None:
    """Verify long Tavily content is truncated for snippet storage."""
    payload = {
        "results": [
            {
                "title": "Long article",
                "url": "https://fantasypros.com/article",
                "content": "x" * (MAX_SNIPPET_CHARS + 500),
            }
        ]
    }

    snippets = TavilySearchGateway._parse_response(payload)

    assert len(snippets[0].snippet) == MAX_SNIPPET_CHARS


def test_search_raw_posts_to_tavily_api(monkeypatch) -> None:
    """Verify search_raw sends Tavily basic-search fields and domain allowlist."""
    gateway = TavilySearchGateway(api_key="tvly-test")
    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            response = SimpleNamespace()
            response.status_code = 200
            response.json = lambda: {"results": []}
            response.raise_for_status = lambda: None
            return response

    monkeypatch.setattr("draft_buddy.data.insights.tavily_search_gateway.httpx.Client", FakeClient)

    gateway.search_raw(
        "Christian McCaffrey fantasy outlook",
        start_date="2026-03-01",
    )

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["headers"]["Authorization"] == "Bearer tvly-test"
    assert captured["json"]["search_depth"] == "basic"
    assert captured["json"]["topic"] == "general"
    assert captured["json"]["start_date"] == "2026-03-01"
    assert "fantasypros.com" in captured["json"]["include_domains"]


def test_search_raw_forwards_topic_override(monkeypatch) -> None:
    """Verify optional topic is sent on the Tavily request body."""
    gateway = TavilySearchGateway(api_key="tvly-test")
    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, headers=None, json=None):
            captured["json"] = json
            response = SimpleNamespace()
            response.status_code = 200
            response.json = lambda: {"results": []}
            response.raise_for_status = lambda: None
            return response

    monkeypatch.setattr("draft_buddy.data.insights.tavily_search_gateway.httpx.Client", FakeClient)

    gateway.search_raw("injury query", topic="news")

    assert captured["json"]["topic"] == "news"


def test_search_raw_raises_quota_exceeded_on_429(monkeypatch) -> None:
    """Verify HTTP 429 is raised as QuotaExceededError."""
    from draft_buddy.data.insights.search_gateway import QuotaExceededError

    gateway = TavilySearchGateway(api_key="tvly-test")

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return SimpleNamespace(status_code=429, json=lambda: {}, raise_for_status=lambda: None)

    monkeypatch.setattr("draft_buddy.data.insights.tavily_search_gateway.httpx.Client", FakeClient)

    with pytest.raises(QuotaExceededError):
        gateway.search_raw("test query")


def test_search_raw_raises_on_http_error(monkeypatch) -> None:
    """Verify non-quota HTTP errors from Tavily are propagated."""
    gateway = TavilySearchGateway(api_key="tvly-test")

    def _raise_for_status():
        raise httpx.HTTPStatusError(
            "error",
            request=SimpleNamespace(),
            response=SimpleNamespace(status_code=500),
        )

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return SimpleNamespace(
                status_code=500,
                json=lambda: {},
                raise_for_status=_raise_for_status,
            )

    monkeypatch.setattr("draft_buddy.data.insights.tavily_search_gateway.httpx.Client", FakeClient)

    with pytest.raises(httpx.HTTPStatusError):
        gateway.search_raw("test query")


def test_build_tavily_search_gateway_uses_api_key(monkeypatch) -> None:
    """Verify build_tavily_search_gateway reads TAVILY_API_KEY from the environment."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    gateway = build_tavily_search_gateway()

    assert isinstance(gateway, TavilySearchGateway)


def test_build_tavily_search_gateway_requires_api_key(monkeypatch) -> None:
    """Verify missing TAVILY_API_KEY raises a clear error."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        build_tavily_search_gateway()
