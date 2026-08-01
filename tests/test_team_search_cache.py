"""Tests for team search result caching."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from draft_buddy.data.insights.cse_gateway import GoogleCseGateway, QuotaExceededError
from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_query_builder import TeamInsightQuery, TeamQueryKind
from draft_buddy.data.insights.team_search_cache import (
    TeamSearchCacheStore,
    execute_team_search_with_cache,
)


def _team() -> InsightTeamContext:
    """Return a test team context."""
    return InsightTeamContext(team_abbr="SF", draft_year=2026)


def test_team_search_cache_store_writes_manifest_and_query_files(tmp_path: Path) -> None:
    """Verify team search cache writes manifest and per-query JSON files."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="test query")

    store.save_query_result(team, query, {"items": []}, [], provider="valyu")

    manifest_path = tmp_path / "SF" / "manifest.json"
    outlook_path = tmp_path / "SF" / "outlook.json"
    assert manifest_path.exists()
    assert outlook_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["queries"] == ["outlook"]
    assert manifest["provider"] == "valyu"
    assert manifest["team_abbr"] == "SF"


def test_has_cached_provider_requires_matching_provider(tmp_path: Path) -> None:
    """Verify cache skip only applies when manifest provider matches."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="test query")
    store.save_query_result(team, query, {"items": []}, [], provider="valyu")

    assert store.has_cached_provider("SF", "valyu") is True
    assert store.has_cached_provider("SF", "google") is False


def test_load_snippets_deduplicates_urls_across_query_kinds(tmp_path: Path) -> None:
    """Verify snippet loading deduplicates by URL across cached query files."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    outlook_query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    offense_query = TeamInsightQuery(kind=TeamQueryKind.OFFENSE, text="offense query")
    duplicate_snippet = {
        "title": "A",
        "snippet": "text",
        "url": "https://espn.com/a",
        "domain": "espn.com",
        "published_date": "2026-06-01",
        "relevance_score": 0.8,
    }
    store.save_query_result(team, outlook_query, {"items": []}, [], provider="valyu")
    store.save_query_result(team, offense_query, {"items": []}, [], provider="valyu")
    for filename in ("outlook.json", "offense.json"):
        query_path = tmp_path / "SF" / filename
        payload = json.loads(query_path.read_text(encoding="utf-8"))
        payload["snippets"] = [duplicate_snippet]
        query_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("SF", draft_year=2026)

    assert len(snippets) == 1


def test_load_snippets_drops_stale_dates_and_ranks_by_score(tmp_path: Path) -> None:
    """Verify stale pre-window snippets are dropped and higher scores rank first."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    store.save_query_result(team, query, {"items": []}, [], provider="valyu")
    outlook_path = tmp_path / "SF" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "Stale recap",
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

    snippets = store.load_snippets("SF", draft_year=2026, today=date(2026, 7, 16))

    assert [snippet.url for snippet in snippets] == [
        "https://espn.com/a",
        "https://espn.com/b",
    ]


def test_load_snippets_keeps_undated_snippet_mentioning_draft_year(tmp_path: Path) -> None:
    """Verify an undated snippet is kept when its text mentions the draft year."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    store.save_query_result(team, query, {"items": []}, [], provider="valyu")
    outlook_path = tmp_path / "SF" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "49ers 2026 Season Preview",
            "snippet": "Retooled offensive line expected to contend.",
            "url": "https://espn.com/undated",
            "domain": "espn.com",
            "published_date": None,
            "relevance_score": 0.8,
        }
    ]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("SF", draft_year=2026)

    assert [snippet.url for snippet in snippets] == ["https://espn.com/undated"]


def test_load_snippets_drops_undated_snippet_without_draft_year_mention(tmp_path: Path) -> None:
    """Verify an undated snippet is dropped when it never mentions the draft year."""
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    store.save_query_result(team, query, {"items": []}, [], provider="valyu")
    outlook_path = tmp_path / "SF" / "outlook.json"
    payload = json.loads(outlook_path.read_text(encoding="utf-8"))
    payload["snippets"] = [
        {
            "title": "49ers franchise history",
            "snippet": "A look back at the franchise's history.",
            "url": "https://espn.com/undated",
            "domain": "espn.com",
            "published_date": None,
            "relevance_score": 0.8,
        }
    ]
    outlook_path.write_text(json.dumps(payload), encoding="utf-8")

    snippets = store.load_snippets("SF", draft_year=2026)

    assert snippets == []


def test_execute_team_search_with_cache_raises_on_quota(monkeypatch, tmp_path: Path) -> None:
    """Verify quota errors are surfaced as QuotaExceededError."""
    gateway = GoogleCseGateway(api_key="key", search_engine_id="cx")

    def _raise_quota(_query: str, num_results: int = 8, **_kwargs):
        raise QuotaExceededError("Google CSE daily quota exceeded.")

    monkeypatch.setattr(gateway, "search_raw", _raise_quota)
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="query")

    with pytest.raises(QuotaExceededError):
        execute_team_search_with_cache(gateway, store, team, query, provider="google")


def test_execute_team_search_with_cache_persists_snippets(monkeypatch, tmp_path: Path) -> None:
    """Verify a successful search persists snippets to the team cache."""
    gateway = GoogleCseGateway(api_key="key", search_engine_id="cx")

    def _fake_search_raw(_query: str, **_kwargs):
        raw_payload = {"items": []}
        return [], raw_payload

    monkeypatch.setattr(gateway, "search_raw", _fake_search_raw)
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.DEFENSE, text="query")

    snippets = execute_team_search_with_cache(gateway, store, team, query, provider="google")

    assert snippets == []
    assert store.has_manifest("SF")


def test_execute_team_search_with_cache_raises_on_http_error(monkeypatch, tmp_path: Path) -> None:
    """Verify non-quota HTTP errors propagate unchanged."""
    gateway = GoogleCseGateway(api_key="key", search_engine_id="cx")

    def _raise_server_error(_query: str, **_kwargs):
        response = SimpleNamespace(status_code=500)
        raise httpx.HTTPStatusError("server error", request=SimpleNamespace(), response=response)

    monkeypatch.setattr(gateway, "search_raw", _raise_server_error)
    store = TeamSearchCacheStore(str(tmp_path))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="query")

    with pytest.raises(httpx.HTTPStatusError):
        execute_team_search_with_cache(gateway, store, team, query, provider="google")
