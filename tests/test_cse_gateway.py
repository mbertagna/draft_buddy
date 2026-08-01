"""Tests for Google CSE gateway and search cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from draft_buddy.data.insights.cse_gateway import (
    GoogleCseGateway,
    QuotaExceededError,
    SearchCacheStore,
    execute_search_with_cache,
)
from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQuery, QueryKind


def _player() -> InsightPlayerContext:
    """Return a test player context."""
    return InsightPlayerContext(
        sleeper_id="4034",
        name="Christian McCaffrey",
        position="RB",
        team="SF",
        adp=6.0,
        projected_points=11.65,
        games_played_frac=0.24,
        draft_year=2026,
    )


def _fake_response(payload: dict, status_code: int = 200) -> SimpleNamespace:
    """Build a fake httpx response."""
    response = SimpleNamespace()
    response.status_code = status_code
    response.json = lambda: payload
    response.raise_for_status = lambda: None
    if status_code >= 400:
        response.raise_for_status = lambda: (_ for _ in ()).throw(
            httpx.HTTPStatusError("error", request=SimpleNamespace(), response=response)
        )
    return response


def test_parse_response_extracts_snippets() -> None:
    """Verify CSE response parsing produces normalized snippets."""
    payload = {
        "items": [
            {
                "title": "McCaffrey outlook",
                "snippet": "Expected workhorse role.",
                "link": "https://espn.com/mccaffrey",
                "displayLink": "espn.com",
                "pagemap": {"metatags": [{"article:published_time": "2026-06-15T12:00:00Z"}]},
            }
        ]
    }

    snippets = GoogleCseGateway._parse_response(payload)

    assert len(snippets) == 1
    assert snippets[0].domain == "espn.com"
    assert snippets[0].published_date == "2026-06-15"


def test_search_cache_store_writes_manifest_and_query_files(tmp_path: Path) -> None:
    """Verify search cache writes manifest and per-query JSON files."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="test query")
    raw_response = {"items": []}

    store.save_query_result(player, query, raw_response, [], provider="google")

    manifest_path = tmp_path / "4034" / "manifest.json"
    outlook_path = tmp_path / "4034" / "outlook.json"
    assert manifest_path.exists()
    assert outlook_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["queries"] == ["outlook"]
    assert manifest["provider"] == "google"


def test_has_cached_provider_requires_matching_provider(tmp_path: Path) -> None:
    """Verify cache skip only applies when manifest provider matches."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="test query")
    store.save_query_result(player, query, {"items": []}, [], provider="valyu")

    assert store.has_cached_provider("4034", "valyu") is True
    assert store.has_cached_provider("4034", "google") is False


def test_load_snippets_deduplicates_urls(tmp_path: Path) -> None:
    """Verify snippet loading deduplicates by URL across query files."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="q1")
    duplicate_snippet = {
        "title": "A",
        "snippet": "text",
        "url": "https://espn.com/a",
        "domain": "espn.com",
        "published_date": "2026-06-01",
        "relevance_score": 0.8,
    }
    store.save_query_result(player, query, {"items": []}, [], provider="google")
    outlook_path = tmp_path / "4034" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [duplicate_snippet, duplicate_snippet]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("4034", draft_year=2026)

    assert len(snippets) == 1


def test_load_snippets_drops_stale_outlook_dates_and_ranks_by_score(
    tmp_path: Path,
) -> None:
    """Verify stale outlook snippets are dropped and high scores rank first."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="q1")
    store.save_query_result(player, query, {"items": []}, [], provider="google")
    outlook_path = tmp_path / "4034" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "Stale injury blurb",
            "snippet": "old",
            "url": "https://espn.com/old",
            "domain": "espn.com",
            "published_date": "2025-10-13",
            "relevance_score": 0.99,
        },
        {
            "title": "Fresh outlook B",
            "snippet": "newer lower score",
            "url": "https://espn.com/b",
            "domain": "espn.com",
            "published_date": "2026-06-01",
            "relevance_score": 0.70,
        },
        {
            "title": "Fresh outlook A",
            "snippet": "higher score",
            "url": "https://espn.com/a",
            "domain": "espn.com",
            "published_date": "2026-05-01",
            "relevance_score": 0.95,
        },
    ]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("4034", draft_year=2026, today=date(2026, 7, 16))

    assert [snippet.url for snippet in snippets] == [
        "https://espn.com/a",
        "https://espn.com/b",
    ]


def test_load_snippets_keeps_undated_snippet_mentioning_draft_year(tmp_path: Path) -> None:
    """Verify an undated outlook snippet is kept when its text mentions the draft year."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="q1")
    store.save_query_result(player, query, {"items": []}, [], provider="valyu")
    outlook_path = tmp_path / "4034" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "2026 Outlook: Christian McCaffrey",
            "snippet": "Bellcow role expected.",
            "url": "https://espn.com/undated",
            "domain": "espn.com",
            "published_date": None,
            "relevance_score": 0.8,
        }
    ]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("4034", draft_year=2026)

    assert [snippet.url for snippet in snippets] == ["https://espn.com/undated"]


def test_load_snippets_drops_undated_snippet_without_draft_year_mention(tmp_path: Path) -> None:
    """Verify an undated outlook snippet is dropped when it never mentions the draft year."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="q1")
    store.save_query_result(player, query, {"items": []}, [], provider="valyu")
    outlook_path = tmp_path / "4034" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "McCaffrey career highlights",
            "snippet": "A look back at his career.",
            "url": "https://espn.com/undated",
            "domain": "espn.com",
            "published_date": None,
            "relevance_score": 0.8,
        }
    ]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("4034", draft_year=2026)

    assert snippets == []


def test_load_snippets_keeps_undated_injury_recovery_snippet(tmp_path: Path) -> None:
    """Verify undated injury-recovery snippets are kept regardless of year mentions."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.INJURY_RECOVERY, text="q1")
    store.save_query_result(player, query, {"items": []}, [], provider="valyu")
    injury_path = tmp_path / "4034" / "injury_recovery.json"
    payload = json.loads(injury_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "McCaffrey cleared for full practice",
            "snippet": "Expected to play Sunday.",
            "url": "https://espn.com/injury",
            "domain": "espn.com",
            "published_date": None,
            "relevance_score": 0.8,
        }
    ]
    injury_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("4034", draft_year=2026)

    assert [snippet.url for snippet in snippets] == ["https://espn.com/injury"]


def test_execute_search_with_cache_raises_on_quota(monkeypatch, tmp_path: Path) -> None:
    """Verify quota errors are surfaced as QuotaExceededError."""
    gateway = GoogleCseGateway(api_key="key", search_engine_id="cx")

    def _raise_quota(_query: str, num_results: int = 8, **_kwargs):
        raise QuotaExceededError("Google CSE daily quota exceeded.")

    monkeypatch.setattr(gateway, "search_raw", _raise_quota)
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="query")

    with pytest.raises(QuotaExceededError):
        execute_search_with_cache(gateway, store, player, query, provider="google")
