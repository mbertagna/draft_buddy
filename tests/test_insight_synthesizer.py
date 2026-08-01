"""Tests for insight synthesis prompt building and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from draft_buddy.data.insights.cse_gateway import SearchCacheStore, SearchSnippet
from draft_buddy.data.insights.gemini_gateway import GeminiGateway
from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQuery, QueryKind
from draft_buddy.data.insights.schemas import (
    Confidence,
    DepthRole,
    EvidenceBullet,
    PlayerInsight,
    PlayingTimeTier,
    RecoveryStatus,
    RiskLevel,
)
from draft_buddy.data.insights.synthesizer import (
    InsightSynthesizer,
    SynthesisCacheStore,
    build_user_prompt,
)


class StubGeminiGateway(GeminiGateway):
    """Gemini gateway stub for synthesis tests."""

    def __init__(self, insight: PlayerInsight) -> None:
        self._insight = insight
        self.calls = 0

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
    ) -> dict[str, Any]:
        """Return a fixed insight payload and record the call."""
        self.calls += 1
        return self._insight.model_dump(mode="json")


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
        sleeper_injury_status=None,
    )


def _insight(url: str) -> PlayerInsight:
    """Return a sample insight with one bullet."""
    return PlayerInsight(
        outlook_phrase="Full-go bellcow role",
        summary="Expected to lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.MEDIUM,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.RECOVERED,
        overall_confidence=Confidence.MEDIUM,
        bullets=[
            EvidenceBullet(
                text="Practicing fully.",
                source_domain="espn.com",
                source_title="Outlook",
                source_url=url,
            )
        ],
    )


def test_build_user_prompt_includes_player_stats_and_snippets() -> None:
    """Verify user prompt contains player stats and snippet URLs."""
    snippets = [
        SearchSnippet(
            title="Outlook",
            snippet="Workhorse role expected.",
            url="https://espn.com/a",
            domain="espn.com",
        )
    ]
    prompt = build_user_prompt(_player(), snippets, ["query one"])

    assert "Christian McCaffrey" in prompt
    assert "https://espn.com/a" in prompt
    assert "query one" in prompt


def test_synthesizer_uses_synthesis_cache(tmp_path: Path) -> None:
    """Verify synthesizer returns cached insight without calling Gemini."""
    search_root = tmp_path / "search"
    synthesis_root = tmp_path / "synthesis"
    search_store = SearchCacheStore(str(search_root))
    synthesis_store = SynthesisCacheStore(str(synthesis_root))
    player = _player()
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(player, query, {"items": []}, [], provider="google")

    cached_insight = _insight("https://espn.com/a")
    synthesis_store.save(player.sleeper_id, cached_insight)
    gateway = StubGeminiGateway(cached_insight)
    synthesizer = InsightSynthesizer(gateway, search_store, synthesis_store)

    result = synthesizer.synthesize_player(player)

    assert result.outlook_phrase == "Full-go bellcow role"
    assert gateway.calls == 0


def test_synthesizer_filters_bullets_to_allowed_urls(tmp_path: Path) -> None:
    """Verify post-validation removes bullets not present in snippets."""
    search_root = tmp_path / "search"
    synthesis_root = tmp_path / "synthesis"
    search_store = SearchCacheStore(str(search_root))
    synthesis_store = SynthesisCacheStore(str(synthesis_root))
    player = _player()
    snippet = SearchSnippet(
        title="2026 Outlook",
        snippet="Workhorse role expected.",
        url="https://espn.com/allowed",
        domain="espn.com",
    )
    query = InsightQuery(kind=QueryKind.OUTLOOK, text="outlook query")
    search_store.save_query_result(player, query, {"items": []}, [snippet], provider="valyu")

    gateway = StubGeminiGateway(_insight("https://blocked.com/x"))
    synthesizer = InsightSynthesizer(gateway, search_store, synthesis_store)

    result = synthesizer.synthesize_player(player, force=True)

    assert result.bullets == []
    assert result.depth_role.value == "unknown"


def test_synthesis_cache_round_trip(tmp_path: Path) -> None:
    """Verify synthesis cache save and load."""
    store = SynthesisCacheStore(str(tmp_path))
    insight = _insight("https://espn.com/a")
    store.save("4034", insight)

    loaded = store.load("4034")
    assert loaded.model_dump() == insight.model_dump()
