"""Tests for search gateway cache and execute_search_with_cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQuery, QueryKind
from draft_buddy.data.insights.search_gateway import (
    QuotaExceededError,
    SearchCacheStore,
    SearchGateway,
    SearchSnippet,
    execute_search_with_cache,
)


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


class StubSearchGateway(SearchGateway):
    """Minimal search gateway stub for cache tests."""

    def search_raw(self, query: str, num_results: int = 8, **kwargs):
        """Return empty snippets and a minimal raw payload."""
        return [], {"results": []}


def test_search_cache_store_writes_manifest_and_query_files(tmp_path: Path) -> None:
    """Verify search cache writes manifest and per-query JSON files."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="test query")
    raw_response = {"results": []}

    store.save_query_result(player, query, raw_response, [])

    manifest_path = tmp_path / "4034" / "manifest.json"
    outlook_path = tmp_path / "4034" / "outlook.json"
    assert manifest_path.exists()
    assert outlook_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["queries"] == ["outlook"]
    assert "provider" not in manifest


def test_has_manifest_reports_cached_player(tmp_path: Path) -> None:
    """Verify has_manifest returns true after a query is cached."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="test query")
    store.save_query_result(player, query, {"results": []}, [])

    assert store.has_manifest("4034") is True
    assert store.has_manifest("9999") is False


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
    store.save_query_result(player, query, {"results": []}, [])
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
    store.save_query_result(player, query, {"results": []}, [])
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


def test_rebuild_snippets_from_raw_prefers_valyu_content(tmp_path: Path) -> None:
    """Verify rebuild rewrites meta snippets from Valyu raw article content."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    player_dir = Path(store.player_cache_dir(player.sleeper_id))
    player_dir.mkdir(parents=True)
    outlook_path = player_dir / "outlook.json"
    outlook_path.write_text(
        json.dumps(
            {
                "query": "outlook",
                "kind": "outlook",
                "fetched_at": "2026-08-06T00:00:00+00:00",
                "raw_response": {
                    "results": [
                        {
                            "title": "Bijan outlook just got complicated",
                            "url": "https://www.cbssports.com/bijan",
                            "description": "Cloudy 2026 Fantasy Football outlook after Falcons moves.",
                            "content": (
                                "Robinson is our RB2 in 2026 and is worth a top-two pick "
                                "in a one-QB league."
                            ),
                            "relevance_score": 0.9,
                        }
                    ]
                },
                "snippets": [
                    {
                        "title": "Bijan outlook just got complicated",
                        "snippet": "Cloudy 2026 Fantasy Football outlook after Falcons moves.",
                        "url": "https://www.cbssports.com/bijan",
                        "domain": "www.cbssports.com",
                        "published_date": None,
                        "relevance_score": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (player_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sleeper_id": player.sleeper_id,
                "name": player.name,
                "draft_year": 2026,
                "queries": ["outlook"],
                "fetched_at": "2026-08-06T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    rewritten = store.rebuild_snippets_from_raw(player.sleeper_id)
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))

    assert rewritten == 1
    assert "RB2" in payload["snippets"][0]["snippet"]
    assert "Cloudy" not in payload["snippets"][0]["snippet"]


def test_rebuild_snippets_from_raw_parses_tavily_score(tmp_path: Path) -> None:
    """Verify rebuild maps Tavily score and published_date onto snippets."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    player_dir = Path(store.player_cache_dir(player.sleeper_id))
    player_dir.mkdir(parents=True)
    outlook_path = player_dir / "outlook.json"
    outlook_path.write_text(
        json.dumps(
            {
                "query": "outlook",
                "kind": "outlook",
                "fetched_at": "2026-08-06T00:00:00+00:00",
                "raw_response": {
                    "results": [
                        {
                            "title": "McCaffrey outlook",
                            "url": "https://www.espn.com/fantasy/mccaffrey",
                            "content": "Expected workhorse role in 2026.",
                            "published_date": "2026-06-15T12:00:00Z",
                            "score": 0.91,
                        }
                    ]
                },
                "snippets": [
                    {
                        "title": "McCaffrey outlook",
                        "snippet": "stale",
                        "url": "https://www.espn.com/fantasy/mccaffrey",
                        "domain": "www.espn.com",
                        "published_date": None,
                        "relevance_score": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (player_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sleeper_id": player.sleeper_id,
                "name": player.name,
                "draft_year": 2026,
                "queries": ["outlook"],
                "fetched_at": "2026-08-06T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    rewritten = store.rebuild_snippets_from_raw(player.sleeper_id)
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    snippet = payload["snippets"][0]

    assert rewritten == 1
    assert snippet["relevance_score"] == pytest.approx(0.91)
    assert snippet["published_date"] == "2026-06-15"
    assert "workhorse" in snippet["snippet"]


def test_rebuild_snippets_from_raw_skips_unrecognized_payloads(tmp_path: Path) -> None:
    """Verify unrecognized raw_response payloads are left unchanged."""
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    player_dir = Path(store.player_cache_dir(player.sleeper_id))
    player_dir.mkdir(parents=True)
    outlook_path = player_dir / "outlook.json"
    outlook_path.write_text(
        json.dumps(
            {
                "query": "outlook",
                "kind": "outlook",
                "fetched_at": "2026-08-06T00:00:00+00:00",
                "raw_response": {"organic": [{"link": "https://example.com"}]},
                "snippets": [],
            }
        ),
        encoding="utf-8",
    )

    rewritten = store.rebuild_snippets_from_raw(player.sleeper_id)

    assert rewritten == 0


def test_execute_search_with_cache_raises_on_quota(tmp_path: Path) -> None:
    """Verify quota errors are surfaced as QuotaExceededError."""
    gateway = StubSearchGateway()

    def _raise_quota(_query: str, num_results: int = 8, **_kwargs):
        raise QuotaExceededError("Valyu API quota exceeded.")

    gateway.search_raw = _raise_quota  # type: ignore[method-assign]
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="query")

    with pytest.raises(QuotaExceededError):
        execute_search_with_cache(gateway, store, player, query)


def test_execute_search_with_cache_persists_results(tmp_path: Path) -> None:
    """Verify execute_search_with_cache writes snippets to the cache store."""
    gateway = StubSearchGateway()
    snippet = SearchSnippet(
        title="Outlook",
        snippet="Workhorse role expected.",
        url="https://espn.com/a",
        domain="espn.com",
    )

    def _return_snippet(_query: str, num_results: int = 8, **_kwargs):
        return [snippet], {"results": [{"url": "https://espn.com/a"}]}

    gateway.search_raw = _return_snippet  # type: ignore[method-assign]
    store = SearchCacheStore(str(tmp_path))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="query")

    result = execute_search_with_cache(gateway, store, player, query)

    assert len(result) == 1
    assert store.has_manifest(player.sleeper_id)


def test_execute_search_with_cache_forwards_tavily_topic(tmp_path: Path) -> None:
    """Verify execute_search_with_cache passes news topic for injury queries."""
    captured: dict = {}
    gateway = StubSearchGateway()

    def _capture(_query: str, num_results: int = 8, **kwargs):
        captured.update(kwargs)
        return [], {"results": []}

    gateway.search_raw = _capture  # type: ignore[method-assign]
    store = SearchCacheStore(str(tmp_path))
    query = InsightQuery(kind=QueryKind.INJURY_RECOVERY, text="injury query")

    execute_search_with_cache(gateway, store, _player(), query)

    assert captured.get("topic") == "news"


def test_execute_search_with_cache_forwards_valyu_instructions(tmp_path: Path) -> None:
    """Verify execute_search_with_cache still forwards Valyu ranking instructions."""
    captured: dict = {}
    gateway = StubSearchGateway()

    def _capture(_query: str, num_results: int = 8, **kwargs):
        captured.update(kwargs)
        return [], {"results": []}

    gateway.search_raw = _capture  # type: ignore[method-assign]
    store = SearchCacheStore(str(tmp_path))
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="outlook query")

    execute_search_with_cache(gateway, store, _player(), query)

    assert captured.get("instructions") is not None
