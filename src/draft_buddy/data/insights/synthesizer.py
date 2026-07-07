"""Synthesize structured player insights from cached search snippets."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterable

from draft_buddy.data.insights.cse_gateway import SearchCacheStore, SearchSnippet
from draft_buddy.data.insights.gemini_gateway import InsightSynthesisGateway
from draft_buddy.data.insights.player_context import InsightPlayerContext
from draft_buddy.data.insights.query_builder import InsightQueryBuilder
from draft_buddy.data.insights.schemas import (
    PlayerInsight,
    PlayerInsightsFile,
    apply_insight_post_validation,
    default_unknown_insight,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fantasy football research analyst.
Synthesize player insights using ONLY the provided search snippets and player stats.
Rules:
- Output JSON matching the PlayerInsight schema exactly.
- Do not include markdown or prose outside JSON.
- Use unknown enum values and fields_unknown when snippets are insufficient.
- Every bullet must cite a snippet URL provided in the input.
- outlook_phrase must be at most 8 words.
- summary must be at most 2 sentences.
- tags must use the controlled vocabulary only.
- Apply injury_recovery only when snippets support a recovery narrative.
"""


class SynthesisCacheStore:
    """Read and write per-player synthesized insight caches."""

    def __init__(self, cache_root: str) -> None:
        """
        Parameters
        ----------
        cache_root : str
            Root directory for synthesis caches.
        """
        self._cache_root = cache_root
        os.makedirs(self._cache_root, exist_ok=True)

    def cache_path(self, sleeper_id: str) -> str:
        """Return the cache file path for one player."""
        return os.path.join(self._cache_root, f"{sleeper_id}.json")

    def has_cache(self, sleeper_id: str) -> bool:
        """Return whether a synthesis cache exists for the player."""
        return os.path.isfile(self.cache_path(sleeper_id))

    def load(self, sleeper_id: str) -> PlayerInsight:
        """Load a cached player insight."""
        with open(self.cache_path(sleeper_id), encoding="utf-8") as handle:
            payload = json.load(handle)
        return PlayerInsight.model_validate(payload)

    def save(self, sleeper_id: str, insight: PlayerInsight) -> None:
        """Persist a synthesized player insight."""
        with open(self.cache_path(sleeper_id), "w", encoding="utf-8") as handle:
            json.dump(insight.model_dump(mode="json"), handle, indent=2)


class InsightSynthesizer:
    """Build prompts and synthesize player insights via Gemini."""

    def __init__(
        self,
        gemini_gateway: InsightSynthesisGateway,
        search_cache: SearchCacheStore,
        synthesis_cache: SynthesisCacheStore,
    ) -> None:
        """
        Parameters
        ----------
        gemini_gateway : InsightSynthesisGateway
            LLM gateway for structured synthesis.
        search_cache : SearchCacheStore
            Cached CSE search results.
        synthesis_cache : SynthesisCacheStore
            Per-player synthesis cache.
        """
        self._gemini = gemini_gateway
        self._search_cache = search_cache
        self._synthesis_cache = synthesis_cache
        self._query_builder = InsightQueryBuilder()

    def synthesize_player(
        self,
        player: InsightPlayerContext,
        force: bool = False,
    ) -> PlayerInsight:
        """Synthesize one player insight, using cache when available.

        Parameters
        ----------
        player : InsightPlayerContext
            Player context.
        force : bool, optional
            Re-synthesize even when cache exists.

        Returns
        -------
        PlayerInsight
            Structured insight for the player.

        Raises
        ------
        FileNotFoundError
            When search cache manifest is missing.
        """
        if not self._search_cache.has_manifest(player.sleeper_id):
            raise FileNotFoundError(
                f"Search cache missing for sleeper_id={player.sleeper_id}. "
                "Run fetch_player_insight_search.py first."
            )

        if self._synthesis_cache.has_cache(player.sleeper_id) and not force:
            return self._synthesis_cache.load(player.sleeper_id)

        snippets = self._search_cache.load_snippets(player.sleeper_id, max_snippets=8)
        queries = self._query_builder.build_queries(player)
        query_texts = [query.text for query in queries]
        allowed_urls = {snippet.url for snippet in snippets if snippet.url}

        if not snippets:
            insight = default_unknown_insight(query_texts)
            self._synthesis_cache.save(player.sleeper_id, insight)
            return insight

        user_prompt = build_user_prompt(player, snippets, query_texts)
        try:
            raw_insight = self._gemini.synthesize(SYSTEM_PROMPT, user_prompt)
            insight = apply_insight_post_validation(raw_insight, allowed_urls=allowed_urls)
            insight = insight.model_copy(
                update={
                    "search_queries_used": query_texts,
                    "snippet_count": len(snippets),
                }
            )
        except Exception:
            logger.exception(
                "Gemini synthesis failed for sleeper_id=%s (%s)",
                player.sleeper_id,
                player.name,
            )
            insight = default_unknown_insight(query_texts)
            insight = insight.model_copy(update={"snippet_count": len(snippets)})

        self._synthesis_cache.save(player.sleeper_id, insight)
        return insight


def build_user_prompt(
    player: InsightPlayerContext,
    snippets: Iterable[SearchSnippet],
    query_texts: list[str],
) -> str:
    """Build the Gemini user prompt for one player.

    Parameters
    ----------
    player : InsightPlayerContext
        Player context from generated CSV.
    snippets : Iterable[SearchSnippet]
        Search snippets from cache.
    query_texts : list[str]
        Queries that were executed.

    Returns
    -------
    str
        Markdown user prompt.
    """
    snippet_lines = []
    for index, snippet in enumerate(snippets, start=1):
        snippet_lines.append(
            f"{index}. title={snippet.title!r}\n"
            f"   url={snippet.url}\n"
            f"   domain={snippet.domain}\n"
            f"   date={snippet.published_date}\n"
            f"   snippet={snippet.snippet!r}"
        )

    return (
        "## Player stats (deterministic, do not contradict without snippet evidence)\n"
        f"- name: {player.name}\n"
        f"- position: {player.position}\n"
        f"- team: {player.team}\n"
        f"- adp: {player.adp}\n"
        f"- projected_points: {player.projected_points}\n"
        f"- games_played_frac: {player.games_played_frac}\n"
        f"- sleeper_injury_status: {player.sleeper_injury_status}\n"
        f"- sleeper_depth_chart_position: {player.sleeper_depth_chart_position}\n"
        f"- draft_year: {player.draft_year}\n\n"
        "## Search queries executed\n"
        + "\n".join(f"- {query}" for query in query_texts)
        + "\n\n## Search snippets (only allowed evidence sources)\n"
        + ("\n".join(snippet_lines) if snippet_lines else "No snippets available.")
    )


def merge_insights_file(
    draft_year: int,
    model: str,
    players: dict[str, PlayerInsight],
) -> PlayerInsightsFile:
    """Build the merged insights file payload.

    Parameters
    ----------
    draft_year : int
        Draft year.
    model : str
        Gemini model name.
    players : dict[str, PlayerInsight]
        Insights keyed by sleeper id string.

    Returns
    -------
    PlayerInsightsFile
        Merged file model.
    """
    return PlayerInsightsFile(
        draft_year=draft_year,
        generated_at=datetime.now(timezone.utc),
        model=model,
        players=players,
    )
