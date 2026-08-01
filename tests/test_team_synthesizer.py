"""Tests for team outlook synthesis prompt building and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import pytest
from pydantic import BaseModel

from draft_buddy.data.insights.cse_gateway import SearchSnippet
from draft_buddy.data.insights.gemini_gateway import GeminiGateway
from draft_buddy.data.insights.schemas import Confidence
from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_query_builder import TeamInsightQuery, TeamQueryKind
from draft_buddy.data.insights.team_schemas import ScheduleHardness, TeamOutlook, TeamTier
from draft_buddy.data.insights.team_search_cache import LEAGUE_CACHE_KEY, TeamSearchCacheStore
from draft_buddy.data.insights.team_synthesizer import (
    TeamOutlookSynthesizer,
    TeamSynthesisCacheStore,
    build_team_user_prompt,
)


class StubTeamGateway(GeminiGateway):
    """Structured synthesis gateway stub for team outlook tests."""

    def __init__(self, outlook: TeamOutlook) -> None:
        self._outlook = outlook
        self.calls = 0

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
    ) -> dict[str, Any]:
        """Return a fixed outlook payload and record the call."""
        self.calls += 1
        return self._outlook.model_dump(mode="json")


def _team() -> InsightTeamContext:
    """Return a test team context."""
    return InsightTeamContext(team_abbr="SF", draft_year=2026)


def _outlook(url: str) -> TeamOutlook:
    """Return a sample team outlook with one bullet."""
    return TeamOutlook(
        team_abbr="SF",
        season=2026,
        outlook_summary="Contender with a retooled offensive line.",
        offense_tier=TeamTier.HIGH,
        offense_notes="Strong receiving corps.",
        defense_tier=TeamTier.MEDIUM,
        defense_notes="Pass rush questions.",
        schedule_hardness=ScheduleHardness.AVERAGE,
        schedule_notes="Middling strength of schedule.",
        overall_confidence=Confidence.MEDIUM,
        bullets=[
            {
                "text": "New OC installs uptempo scheme.",
                "source_domain": "espn.com",
                "source_title": "Outlook",
                "source_url": url,
            }
        ],
    )


def test_build_team_user_prompt_includes_team_and_snippets() -> None:
    """Verify user prompt contains team abbreviation and snippet URLs."""
    snippets = [
        SearchSnippet(
            title="Outlook",
            snippet="Retooled offensive line expected.",
            url="https://espn.com/a",
            domain="espn.com",
        )
    ]
    prompt = build_team_user_prompt(_team(), snippets, ["query one"])

    assert "SF" in prompt
    assert "https://espn.com/a" in prompt
    assert "query one" in prompt


def test_synthesizer_uses_synthesis_cache(tmp_path: Path) -> None:
    """Verify synthesizer returns cached outlook without calling the gateway."""
    search_root = tmp_path / "search"
    synthesis_root = tmp_path / "synthesis"
    search_store = TeamSearchCacheStore(str(search_root))
    synthesis_store = TeamSynthesisCacheStore(str(synthesis_root))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(team, query, {"items": []}, [], provider="valyu")

    cached_outlook = _outlook("https://espn.com/a")
    synthesis_store.save(team.team_abbr, cached_outlook)
    gateway = StubTeamGateway(cached_outlook)
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store)

    result = synthesizer.synthesize_team(team)

    assert result.outlook_summary == "Contender with a retooled offensive line."
    assert gateway.calls == 0


def test_synthesizer_filters_bullets_to_allowed_urls(tmp_path: Path) -> None:
    """Verify post-validation removes bullets not present in snippets."""
    search_root = tmp_path / "search"
    synthesis_root = tmp_path / "synthesis"
    search_store = TeamSearchCacheStore(str(search_root))
    synthesis_store = TeamSynthesisCacheStore(str(synthesis_root))
    team = _team()
    snippet = SearchSnippet(
        title="2026 Outlook",
        snippet="Retooled offensive line expected.",
        url="https://espn.com/allowed",
        domain="espn.com",
    )
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(team, query, {"items": []}, [snippet], provider="valyu")

    gateway = StubTeamGateway(_outlook("https://blocked.com/x"))
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store)

    result = synthesizer.synthesize_team(team, force=True)

    assert result.bullets == []
    assert result.offense_tier == TeamTier.UNKNOWN


def test_synthesizer_falls_back_to_unknown_when_no_snippets(tmp_path: Path) -> None:
    """Verify synthesizer returns a default-unknown outlook when no snippets exist."""
    search_root = tmp_path / "search"
    synthesis_root = tmp_path / "synthesis"
    search_store = TeamSearchCacheStore(str(search_root))
    synthesis_store = TeamSynthesisCacheStore(str(synthesis_root))
    team = _team()
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(team, query, {"items": []}, [], provider="valyu")

    gateway = StubTeamGateway(_outlook("https://espn.com/a"))
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store)

    result = synthesizer.synthesize_team(team)

    assert result.offense_tier == TeamTier.UNKNOWN
    assert gateway.calls == 0


def test_synthesizer_raises_when_search_cache_missing(tmp_path: Path) -> None:
    """Verify synthesizer raises when the team has no search cache manifest."""
    search_store = TeamSearchCacheStore(str(tmp_path / "search"))
    synthesis_store = TeamSynthesisCacheStore(str(tmp_path / "synthesis"))
    gateway = StubTeamGateway(_outlook("https://espn.com/a"))
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store)

    with pytest.raises(FileNotFoundError, match="--scope teams"):
        synthesizer.synthesize_team(_team())


def test_team_synthesis_cache_round_trip(tmp_path: Path) -> None:
    """Verify team synthesis cache save and load."""
    store = TeamSynthesisCacheStore(str(tmp_path))
    outlook = _outlook("https://espn.com/a")
    store.save("SF", outlook)

    loaded = store.load("SF")
    assert loaded.model_dump() == outlook.model_dump()


def _save_league_snippet(league_store: TeamSearchCacheStore, snippet: SearchSnippet) -> None:
    """Save one league-wide snippet under the shared league cache key."""
    league = InsightTeamContext(team_abbr=LEAGUE_CACHE_KEY, draft_year=2026)
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="league outlook query")
    league_store.save_query_result(league, query, {"items": []}, [snippet], provider="valyu")


def test_synthesizer_pulls_in_league_snippet_mentioning_team(tmp_path: Path) -> None:
    """Verify a league-wide snippet naming this team is merged into synthesis."""
    search_store = TeamSearchCacheStore(str(tmp_path / "search"))
    league_store = TeamSearchCacheStore(str(tmp_path / "league"))
    synthesis_store = TeamSynthesisCacheStore(str(tmp_path / "synthesis"))
    team = _team()

    team_snippet = SearchSnippet(
        title="2026 Outlook",
        snippet="Retooled offensive line expected.",
        url="https://espn.com/team",
        domain="espn.com",
    )
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(team, query, {"items": []}, [team_snippet], provider="valyu")

    league_snippet = SearchSnippet(
        title="2026 NFL best/worst case: 49ers set for bounce-back",
        snippet="San Francisco 49ers among the league's most-improved rosters.",
        url="https://espn.com/league-roundup",
        domain="espn.com",
    )
    _save_league_snippet(league_store, league_snippet)

    gateway = StubTeamGateway(_outlook("https://espn.com/league-roundup"))
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store, league_store)

    result = synthesizer.synthesize_team(team, force=True)

    assert result.bullets != []
    assert result.snippet_count == 2


def test_synthesizer_excludes_league_snippet_not_mentioning_team(tmp_path: Path) -> None:
    """Verify a league-wide snippet that never names this team is left out."""
    search_store = TeamSearchCacheStore(str(tmp_path / "search"))
    league_store = TeamSearchCacheStore(str(tmp_path / "league"))
    synthesis_store = TeamSynthesisCacheStore(str(tmp_path / "synthesis"))
    team = _team()

    team_snippet = SearchSnippet(
        title="2026 Outlook",
        snippet="Retooled offensive line expected.",
        url="https://espn.com/team",
        domain="espn.com",
    )
    query = TeamInsightQuery(kind=TeamQueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(team, query, {"items": []}, [team_snippet], provider="valyu")

    league_snippet = SearchSnippet(
        title="2026 NFL best/worst case: Chiefs eye repeat",
        snippet="Kansas City Chiefs enter the season as favorites.",
        url="https://espn.com/league-roundup",
        domain="espn.com",
    )
    _save_league_snippet(league_store, league_snippet)

    gateway = StubTeamGateway(_outlook("https://espn.com/league-roundup"))
    synthesizer = TeamOutlookSynthesizer(gateway, search_store, synthesis_store, league_store)

    result = synthesizer.synthesize_team(team, force=True)

    assert result.bullets == []
    assert result.snippet_count == 1


def test_build_team_user_prompt_tags_league_snippets() -> None:
    """Verify league-sourced snippets are flagged so the LLM extracts only this team's part."""
    snippets = [
        SearchSnippet(
            title="Team-specific article",
            snippet="Details about this team.",
            url="https://espn.com/team",
            domain="espn.com",
        ),
        SearchSnippet(
            title="Roundup article",
            snippet="Covers all 32 teams.",
            url="https://espn.com/roundup",
            domain="espn.com",
        ),
    ]

    prompt = build_team_user_prompt(
        _team(), snippets, ["query one"], league_urls={"https://espn.com/roundup"}
    )

    roundup_line = next(line for line in prompt.splitlines() if "Roundup article" in line)
    assert "[multi-team roundup]" in roundup_line
    team_line = next(line for line in prompt.splitlines() if "Team-specific article" in line)
    assert "[multi-team roundup]" not in team_line
