"""Tests for Valyu Search API gateway."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from draft_buddy.data.insights.valyu_search_gateway import (
    MAX_SNIPPET_CHARS,
    ValyuSearchGateway,
)


def test_parse_response_extracts_snippets_from_results() -> None:
    """Verify Valyu response parsing maps content to snippet text."""
    payload = {
        "results": [
            {
                "title": "McCaffrey outlook",
                "url": "https://www.espn.com/fantasy/mccaffrey",
                "content": "Expected workhorse role in 2026.",
                "publication_date": "2026-06-15T12:00:00Z",
                "relevance_score": 0.91,
            }
        ]
    }

    snippets = ValyuSearchGateway._parse_response(payload)

    assert len(snippets) == 1
    assert snippets[0].domain == "www.espn.com"
    assert snippets[0].published_date == "2026-06-15"
    assert snippets[0].relevance_score == pytest.approx(0.91)
    assert "workhorse" in snippets[0].snippet


def test_parse_response_prefers_shorter_description() -> None:
    """Verify description is preferred when shorter than content."""
    payload = {
        "results": [
            {
                "title": "Outlook",
                "url": "https://fantasypros.com/article",
                "description": "Clean outlook summary.",
                "content": "Much longer noisy page content that includes sidebars and related links.",
            }
        ]
    }

    snippets = ValyuSearchGateway._parse_response(payload)

    assert snippets[0].snippet == "Clean outlook summary."


def test_parse_response_truncates_long_content() -> None:
    """Verify long Valyu content is truncated for snippet storage."""
    payload = {
        "results": [
            {
                "title": "Long article",
                "url": "https://fantasypros.com/article",
                "content": "x" * (MAX_SNIPPET_CHARS + 500),
            }
        ]
    }

    snippets = ValyuSearchGateway._parse_response(payload)

    assert len(snippets[0].snippet) == MAX_SNIPPET_CHARS


def test_search_raw_posts_to_valyu_api(monkeypatch) -> None:
    """Verify search_raw sends date filters and raised relevance threshold."""
    gateway = ValyuSearchGateway(api_key="test-key")
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

    monkeypatch.setattr("draft_buddy.data.insights.valyu_search_gateway.httpx.Client", FakeClient)

    gateway.search_raw(
        "Christian McCaffrey fantasy outlook",
        start_date="2026-03-01",
        instructions="Prefer season fantasy outlooks.",
    )

    assert captured["url"] == "https://api.valyu.ai/v1/search"
    assert captured["headers"]["X-Api-Key"] == "test-key"
    assert captured["json"]["query"] == "Christian McCaffrey fantasy outlook"
    assert captured["json"]["relevance_threshold"] == 0.7
    assert captured["json"]["start_date"] == "2026-03-01"
    assert captured["json"]["instructions"] == "Prefer season fantasy outlooks."
    assert "fantasypros.com" in captured["json"]["included_sources"]


def test_search_raw_raises_on_http_error(monkeypatch) -> None:
    """Verify HTTP errors from Valyu are propagated."""
    gateway = ValyuSearchGateway(api_key="test-key")

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

    monkeypatch.setattr("draft_buddy.data.insights.valyu_search_gateway.httpx.Client", FakeClient)

    with pytest.raises(httpx.HTTPStatusError):
        gateway.search_raw("test query")
